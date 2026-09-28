"""Experiment and figure runners, one per build step (called from main.py).

The sweep engine (run_sweep) is shared by Steps 4-7. Each (sentence, noise,
SNR) case is one work unit: the mixture is built once from config.case_rng,
the noisy input is scored once, and then every method configuration of the
sweep is run and scored on it. Completed cases are appended to a
``.partial.csv`` file as they finish (a crash loses nothing, and --resume
skips rows already there); at the end all rows are sorted by their key and
written to the final CSV, so the file is byte-identical whatever order the
worker processes finished in. Per-configuration wall-clock times go to a
separate timing CSV because they cannot be reproducible.
"""

import argparse
import csv
import hashlib
import json
import logging
import math
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

import config
from src import plots, spectral_sub, wavelet, wiener
from src.data import Dataset, load_audio_file, load_dataset
from src.metrics import evaluate, find_lag, pesq_nb, seg_snr_frames
from src.mixing import make_mixture, measured_snr_db, mix_at_snr
from src.stft import istft, stft

log = logging.getLogger(__name__)

EXAMPLE_SENTENCE = 0  # sp01 (or syn01): used for single-sentence figures
EXAMPLE_SNR_DB = 5

DENOISERS: dict[str, Callable[..., np.ndarray]] = {
    "spectral_sub": spectral_sub.denoise,
    "wiener": wiener.denoise,
    "wavelet": wavelet.denoise,
}

RUN_KEY = ("sentence", "noise", "snr_in", "method", "variant", "alpha", "beta", "eta",
           "wavelet", "level")
RUN_METRICS = ("segsnr_noisy", "pesq_noisy", "stoi_noisy", "segsnr_out", "pesq_out",
               "stoi_out", "d_segsnr", "d_pesq", "d_stoi")
RUN_COLUMNS = RUN_KEY + RUN_METRICS


@dataclass(frozen=True)
class MethodConfig:
    """One denoiser configuration. Unused parameters stay None / ''."""

    method: str
    variant: str = ""
    alpha: float | None = None
    beta: float | None = None
    eta: float | None = None
    wavelet: str = ""
    level: int | None = None

    def kwargs(self) -> dict:
        if self.method == "spectral_sub":
            return {"alpha": self.alpha, "beta": self.beta}
        if self.method == "wiener":
            return {"eta": self.eta, "xi_min_db": config.WIENER_XI_MIN_DB}
        if self.method == "wavelet":
            return {"wavelet": self.wavelet, "level": self.level, "variant": self.variant}
        raise ValueError(f"unknown method {self.method!r}")

    def fields(self) -> dict:
        return {"method": self.method, "variant": self.variant,
                "alpha": "" if self.alpha is None else self.alpha,
                "beta": "" if self.beta is None else self.beta,
                "eta": "" if self.eta is None else self.eta,
                "wavelet": self.wavelet,
                "level": "" if self.level is None else self.level}

    def label(self) -> str:
        if self.method == "spectral_sub":
            return f"SS a={self.alpha:g} b={self.beta:g}"
        if self.method == "wiener":
            return f"Wiener eta={self.eta:g}"
        if self.method == "wavelet":
            return f"Wavelet {self.wavelet} L{self.level} {self.variant}"
        return self.method

    def plot_key(self) -> str:
        """Key into plots.METHOD_COLORS / METHOD_MARKERS / METHOD_LABELS."""
        return plots.method_key(self.method, self.variant)


def ss_default() -> MethodConfig:
    return MethodConfig("spectral_sub", alpha=config.SS_ALPHA, beta=config.SS_BETA)


def ss_grid() -> list[MethodConfig]:
    return [MethodConfig("spectral_sub", alpha=a, beta=b)
            for a in config.SS_ALPHA_GRID for b in config.SS_BETA_GRID]


def wiener_default() -> MethodConfig:
    return MethodConfig("wiener", eta=config.WIENER_ETA)


def wiener_grid() -> list[MethodConfig]:
    return [MethodConfig("wiener", eta=e) for e in config.WIENER_ETA_GRID]


def wavelet_default(variant: str = config.WAVELET_VARIANT) -> MethodConfig:
    return MethodConfig("wavelet", variant=variant, wavelet=config.WAVELET,
                        level=config.WAVELET_LEVEL)


def wavelet_grid() -> list[MethodConfig]:
    return [MethodConfig("wavelet", variant=v, wavelet=w, level=lev)
            for v in config.WAVELET_VARIANTS for w in config.WAVELET_FAMILIES
            for lev in config.WAVELET_LEVELS]


def default_configs() -> list[MethodConfig]:
    """The three methods at their CLAUDE.md defaults, plus the level-dependent wavelet."""
    return [ss_default(), wiener_default(), wavelet_default("universal"),
            wavelet_default("level_dependent")]


# --- worker side -------------------------------------------------------------
_WORKER: dict = {}


def _init_worker(signals: list[np.ndarray], speakers, names: list[str]) -> None:
    _WORKER.update(signals=signals, speakers=speakers, names=names)


def _run_case(
    case: tuple[int, str, int], configs: Sequence[MethodConfig], perceptual: bool
) -> tuple[list[dict], list[dict]]:
    """Score the noisy input and every config on one (sentence, noise, SNR) case."""
    si, noise_type, snr = case
    name = _WORKER["names"][si]
    m = make_mixture(_WORKER["signals"], si, noise_type, snr, _WORKER["speakers"])
    tag = f"{name}/{noise_type}/{snr}dB"
    base = evaluate(m.clean, m.noisy[m.n_lead :], perceptual=perceptual, tag=f"{tag}/noisy")
    rows, timings = [], []
    for cfg in configs:
        t_start = time.perf_counter()
        out = DENOISERS[cfg.method](m.noisy, config.FS, m.n_lead, **cfg.kwargs())
        runtime = time.perf_counter() - t_start
        res = evaluate(m.clean, out[m.n_lead :], perceptual=perceptual,
                       tag=f"{tag}/{cfg.label()}")
        key = {"sentence": name, "noise": noise_type, "snr_in": snr, **cfg.fields()}
        rows.append({
            **key,
            "segsnr_noisy": base["segsnr"], "pesq_noisy": base["pesq"],
            "stoi_noisy": base["stoi"], "segsnr_out": res["segsnr"],
            "pesq_out": res["pesq"], "stoi_out": res["stoi"],
            "d_segsnr": res["segsnr"] - base["segsnr"], "d_pesq": res["pesq"] - base["pesq"],
            "d_stoi": res["stoi"] - base["stoi"],
        })
        timings.append({**key, "runtime_s": runtime})
    return rows, timings


# --- driver side -------------------------------------------------------------
def _key_str(row: dict) -> tuple[str, ...]:
    return tuple(str(row[k]) for k in RUN_KEY)


def _sort_key(row: dict) -> tuple:
    def num(v):
        return -math.inf if v in ("", None) else float(v)

    return (row["sentence"], row["noise"], num(row["snr_in"]), row["method"], row["variant"],
            num(row["alpha"]), num(row["beta"]), num(row["eta"]), row["wavelet"],
            num(row["level"]))


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _append_rows(path: Path, rows: list[dict], columns: Sequence[str]) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        if new:
            writer.writeheader()
        writer.writerows(rows)


def _finalise(partial: Path, final: Path, columns: Sequence[str]) -> list[dict]:
    """Sort every row of ``partial`` by key, write ``final``, delete ``partial``."""
    rows = sorted(_read_rows(partial), key=_sort_key)
    keys = [_key_str(r) for r in rows]
    if len(set(keys)) != len(keys):
        raise RuntimeError(f"duplicate rows in {partial}")
    tmp = final.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(final)
    partial.unlink()
    return rows


def run_sweep(
    name: str,
    configs: Sequence[MethodConfig],
    args: argparse.Namespace,
    ds: Dataset,
    sentences: Sequence[int],
    snrs: Sequence[int],
    results_dir: Path,
) -> list[dict]:
    """Run ``configs`` on every case and write results_dir/runs_<name>.csv.

    Returns the final rows (as read back from the CSV, i.e. strings).
    """
    final = results_dir / f"runs_{name}.csv"
    partial = results_dir / f"runs_{name}.partial.csv"
    t_final = results_dir / f"timing_{name}.csv"
    t_partial = results_dir / f"timing_{name}.partial.csv"
    timing_cols = RUN_KEY + ("runtime_s",)
    if args.resume:
        # Merge the finished file and any partial one (de-duplicated by key) into
        # a fresh partial, so a crash at any point, even mid-finalise, resumes cleanly.
        for fin, par, cols in ((final, partial, RUN_COLUMNS), (t_final, t_partial, timing_cols)):
            merged = {}
            for r in _read_rows(par) + _read_rows(fin):
                merged.setdefault(_key_str(r), r)
            par.unlink(missing_ok=True)
            fin.unlink(missing_ok=True)
            if merged:
                _append_rows(par, list(merged.values()), cols)
    else:
        for p in (final, partial, t_final, t_partial):
            p.unlink(missing_ok=True)
    done = {_key_str(r) for r in _read_rows(partial)}
    if args.resume and t_partial.exists():
        # Keep timings only for rows that are done; the rest are recomputed and re-timed.
        kept = [r for r in _read_rows(t_partial) if _key_str(r) in done]
        t_partial.unlink()
        if kept:
            _append_rows(t_partial, kept, timing_cols)

    perceptual = not args.no_perceptual
    work = []
    for si in sentences:
        for noise_type in config.NOISE_TYPES:
            for snr in snrs:
                todo = [c for c in configs if _key_str(
                    {"sentence": ds.names[si], "noise": noise_type, "snr_in": snr, **c.fields()}
                ) not in done]
                if todo:
                    work.append(((si, noise_type, snr), todo))
    n_expected = len(sentences) * len(config.NOISE_TYPES) * len(snrs) * len(configs)
    log.info("sweep %s: %d configs x %d cases = %d runs (%d already done), %d workers",
             name, len(configs), len(sentences) * len(config.NOISE_TYPES) * len(snrs),
             n_expected, len(done), args.workers)

    t_start = time.perf_counter()
    init = (ds.signals, ds.speakers, ds.names)
    if args.workers == 1:
        _init_worker(*init)
        for case, todo in work:
            rows, timings = _run_case(case, todo, perceptual)
            _append_rows(partial, rows, RUN_COLUMNS)
            _append_rows(t_partial, timings, timing_cols)
    else:
        with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=init) as ex:
            futures = [ex.submit(_run_case, case, todo, perceptual) for case, todo in work]
            for i, fut in enumerate(as_completed(futures), 1):
                rows, timings = fut.result()
                _append_rows(partial, rows, RUN_COLUMNS)
                _append_rows(t_partial, timings, timing_cols)
                if i % max(1, len(futures) // 10) == 0:
                    log.info("  %s: %d/%d cases", name, i, len(futures))
    if not partial.exists():  # nothing to do and nothing resumed
        raise RuntimeError(f"sweep {name} produced no rows")
    rows = _finalise(partial, final, RUN_COLUMNS)
    _finalise(t_partial, t_final, timing_cols)
    if len(rows) != n_expected:
        raise RuntimeError(f"{final}: {len(rows)} rows, expected {n_expected}")
    log.info("sweep %s done in %.1f s: %d rows -> %s", name, time.perf_counter() - t_start,
             len(rows), final)
    if perceptual:
        n_nan = sum(math.isnan(float(r[c])) for r in rows for c in ("pesq_noisy", "pesq_out"))
        log.info("sweep %s: PESQ NaN values: %d", name, n_nan)
    else:
        log.info("sweep %s: PESQ/STOI skipped (--no-perceptual)", name)
    return rows


def _dataset(args: argparse.Namespace) -> Dataset:
    ds = load_dataset(force_synthetic=args.synthetic)
    log.info("dataset: %s, %d sentences", "synthetic" if ds.synthetic else "NOIZEUS",
             len(ds.signals))
    return ds


def _outputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    return config.output_dirs(quick=args.quick, synthetic=args.synthetic,
                              perceptual=not args.no_perceptual)


def _grid(args: argparse.Namespace, ds: Dataset) -> tuple[range, tuple[int, ...]]:
    """(sentence indices, input SNRs) for this run mode."""
    if args.quick:
        return range(min(config.QUICK_N_SENTENCES, len(ds.signals))), config.QUICK_SNRS_DB
    return range(len(ds.signals)), config.SNRS_DB


def _write_csv(path: Path, rows: list[dict]) -> Path:
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %s (%d rows)", path, len(rows))
    return path


def _write_or_remove(path: Path, rows: list[dict]) -> None:
    """Write ``rows`` to ``path``, or delete a stale ``path`` when there are none."""
    if rows:
        _write_csv(path, rows)
    else:
        path.unlink(missing_ok=True)


def write_timing_summary(results: Path) -> None:
    """Mean and max denoiser runtime per configuration, from the timing_<sweep>.csv files.

    Wall-clock numbers: not reproducible, so the name starts with 'timing_' and
    csv_hashes() skips it.
    """
    out = []
    for name in SWEEPS:
        groups: dict[tuple, list[float]] = {}
        for r in _read_rows(results / f"timing_{name}.csv"):
            groups.setdefault(tuple(r[k] for k in RUN_KEY[3:]), []).append(float(r["runtime_s"]))
        for key, times in groups.items():
            out.append({**dict(zip(RUN_KEY[3:], key, strict=True)), "n_runs": len(times),
                        "mean_runtime_ms": 1000 * float(np.mean(times)),
                        "max_runtime_ms": 1000 * float(np.max(times))})
    out.sort(key=lambda r: _sort_key({"sentence": "", "noise": "", "snr_in": "", **r}))
    _write_or_remove(results / "timing_summary.csv", out)


def write_dataset_summary(ds: Dataset, results_dir: Path) -> None:
    """Log a per-sentence table and write it to <results>/dataset_summary.csv."""
    rows = []
    for i, (name, x) in enumerate(zip(ds.names, ds.signals, strict=True)):
        spk, gender = ds.speakers[i] if ds.speakers else ("-", "-")
        rows.append({
            "sentence": name, "speaker": spk, "gender": gender, "fs": config.FS,
            "n_samples": x.size, "duration_s": round(x.size / config.FS, 4),
            "rms": round(float(np.sqrt(np.mean(x**2))), 6),
            "peak": round(float(np.max(np.abs(x))), 6),
        })
    _write_csv(results_dir / "dataset_summary.csv", rows)
    header = ("sentence", "spk", "g", "n", "dur_s", "rms", "peak")
    lines = ["{:9s} {:3s} {:1s} {:>6s} {:>6s} {:>7s} {:>6s}".format(*header)]
    lines += [
        f"{r['sentence']:9s} {r['speaker']:3s} {r['gender']:1s} {r['n_samples']:6d} "
        f"{r['duration_s']:6.3f} {r['rms']:7.4f} {r['peak']:6.3f}" for r in rows
    ]
    durs = [r["duration_s"] for r in rows]
    lines.append(f"count={len(rows)}  fs={config.FS}  duration {min(durs):.3f}-{max(durs):.3f} s")
    log.info("dataset summary:\n%s", "\n".join(lines))


def run_step1(args: argparse.Namespace) -> None:
    """Step 1 (O1): figures/cola.png and results/stft_checks.csv.

    The CSV records the perfect-reconstruction error of STFT -> iSTFT with no
    modification, for seeded white noise of several lengths and for every
    sentence in the dataset, plus the COLA envelope deviation. (The SciPy
    cross-check lives in tests/test_stft.py: the SciPy STFT is used in tests only.)
    """
    results, figures, _ = _outputs(args)
    devs = plots.plot_cola(figures / "cola.png")
    rows = [
        {"check": "cola_interior_max_dev", "signal": "periodic_hann", "length": "",
         "value": devs["max_dev_periodic"]},
        {"check": "cola_interior_max_dev", "signal": "symmetric_hann_np.hanning", "length": "",
         "value": devs["max_dev_symmetric"]},
    ]
    signals = [
        (f"white_L{n}", np.random.default_rng([config.SEED, n]).standard_normal(n))
        for n in config.STFT_CHECK_LENGTHS
    ]
    ds = _dataset(args)
    signals += list(zip(ds.names, ds.signals, strict=True))
    for name, x in signals:
        err = float(np.max(np.abs(istft(stft(x), length=x.size) - x)))
        rows.append({"check": "roundtrip_max_abs_err", "signal": name, "length": x.size,
                     "value": err})
    _write_csv(results / "stft_checks.csv", rows)
    worst = max(r["value"] for r in rows if r["check"] == "roundtrip_max_abs_err")
    log.info("round trip: worst max|x_hat - x| = %.2e over %d signals; COLA dev %.2e",
             worst, len(signals), devs["max_dev_periodic"])


def run_step2(args: argparse.Namespace) -> None:
    """Step 2 (O3): dataset summary and figures/mixing_example.png."""
    results, figures, _ = _outputs(args)
    ds = _dataset(args)
    write_dataset_summary(ds, results)
    sentences, snrs = _grid(args, ds)
    rows = []
    for si in sentences:
        for noise_type in config.NOISE_TYPES:
            for snr in snrs:
                m = make_mixture(ds.signals, si, noise_type, snr, ds.speakers)
                got = measured_snr_db(m.clean, m.noise_scaled, m.n_lead)
                rows.append({"sentence": ds.names[si], "noise": noise_type, "snr_target": snr,
                             "snr_measured": got, "abs_err_db": abs(got - snr)})
    _write_csv(results / "mixing_check.csv", rows)
    log.info("mixing: worst |measured - target| = %.2e dB over %d mixtures",
             max(r["abs_err_db"] for r in rows), len(rows))
    i = EXAMPLE_SENTENCE
    panels = []
    for noise_type in config.NOISE_TYPES:
        m = make_mixture(ds.signals, i, noise_type, EXAMPLE_SNR_DB, ds.speakers)
        log.info("%s %s %d dB: measured SNR %.6f dB", ds.names[i], noise_type, EXAMPLE_SNR_DB,
                 measured_snr_db(m.clean, m.noise_scaled, m.n_lead))
        if not panels:
            panels.append((f"Clean ({ds.names[i]})", m.clean_padded))
        panels.append((f"{noise_type.capitalize()} noise, {EXAMPLE_SNR_DB} dB", m.noisy))
    plots.plot_mixing_example(
        panels, config.N_LEAD, figures / "mixing_example.png",
        f"Mixing example: {ds.names[i]} with a {config.LEAD_IN_S} s noise-only lead-in "
        f"(shaded; dashed line = sentence start)",
    )


def run_step3(args: argparse.Namespace) -> None:
    """Step 3 (O4): noisy-input baseline metrics -> results/noisy_baseline.csv.

    For each case, the unprocessed noisy input (lead-in trimmed) is scored
    against the clean sentence. These are the 'noisy' baselines that every
    improvement (Delta) in later steps is measured from.
    """
    results, _, _ = _outputs(args)
    ds = _dataset(args)
    sentences, snrs = _grid(args, ds)
    rows = []
    for si in sentences:
        for noise_type in config.NOISE_TYPES:
            for snr in snrs:
                m = make_mixture(ds.signals, si, noise_type, snr, ds.speakers)
                tag = f"{ds.names[si]}/{noise_type}/{snr}dB/noisy"
                met = evaluate(m.clean, m.noisy[m.n_lead :], perceptual=not args.no_perceptual,
                               tag=tag)
                rows.append({"sentence": ds.names[si], "noise": noise_type, "snr_in": snr,
                             **met})
    _write_csv(results / "noisy_baseline.csv", rows)
    lines = [f"{'noise':7s} {'snr_in':>6s} {'segSNR':>7s} {'global':>7s} {'PESQ':>6s} {'STOI':>6s}"]
    for noise_type in config.NOISE_TYPES:
        for snr in snrs:
            sel = [r for r in rows if r["noise"] == noise_type and r["snr_in"] == snr]
            mean = {k: float(np.mean([r[k] for r in sel]))
                    for k in ("segsnr", "global_snr", "pesq", "stoi")}
            lines.append(f"{noise_type:7s} {snr:6d} {mean['segsnr']:7.2f} "
                         f"{mean['global_snr']:7.2f} {mean['pesq']:6.3f} {mean['stoi']:6.3f}")
    n_nan = sum(np.isnan(r["pesq"]) for r in rows) if not args.no_perceptual else 0
    log.info("noisy-input baseline (mean over %d sentences):\n%s", len(sentences),
             "\n".join(lines))
    log.info("PESQ NaN count: %d of %d", n_nan, len(rows))


# ---------------------------------------------------------------------------
# WAV output (for listening only; metrics always use the float arrays)
# ---------------------------------------------------------------------------
WAV_CLIP_LEVEL = 1.0 - 2.0**-15  # largest PCM_16 value; reaching it counts as clipping


def write_wav_set(folder: Path, signals: dict[str, np.ndarray]) -> float:
    """Write <folder>/<name>.wav (8 kHz PCM_16) for each signal, all scaled by one
    common factor so relative levels are kept and the loudest peak is WAV_PEAK."""
    folder.mkdir(parents=True, exist_ok=True)
    peak = max(float(np.max(np.abs(x))) for x in signals.values())
    scale = config.WAV_PEAK / peak
    for name, x in signals.items():
        sf.write(folder / f"{name}.wav", scale * x, config.FS, subtype="PCM_16")
    return scale


def check_wav(path: Path, expected_len: int) -> dict:
    """Assert sample rate, mono, length, peak <= 1 and no clipped samples."""
    x, fs = sf.read(path, dtype="float64")
    shown = path.relative_to(config.ROOT) if path.is_relative_to(config.ROOT) else path
    info = {"file": str(shown), "fs": fs, "n_samples": x.size,
            "peak": float(np.max(np.abs(x))),
            "n_clipped": int(np.sum(np.abs(x) >= WAV_CLIP_LEVEL))}
    assert fs == config.FS, info
    assert x.ndim == 1 and x.size == expected_len, info
    assert info["peak"] <= 1.0 and info["n_clipped"] == 0, info
    return info


# ---------------------------------------------------------------------------
# Speech-free regions and musical-noise statistics
# ---------------------------------------------------------------------------
def speech_free_frames(clean_padded: np.ndarray, n_lead: int) -> np.ndarray:
    """Boolean mask over STFT frames: frames lying entirely after the lead-in
    whose clean energy is below the segSNR gate (config.SEGSNR_GATE_DB
    relative to the loudest frame). The lead-in itself is excluded because the
    noise estimate was taken there."""
    energy = np.sum(np.abs(stft(clean_padded)) ** 2, axis=1)
    m = np.arange(energy.size)
    after_lead = m * config.HOP - config.FRAME_LEN // 2 >= n_lead
    quiet = energy < np.max(energy) * 10.0 ** (config.SEGSNR_GATE_DB / 10.0)
    return after_lead & quiet


def longest_run(mask: np.ndarray) -> tuple[int, int]:
    """(start, stop) of the longest run of True in ``mask`` (stop exclusive)."""
    best, start = (0, 0), None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if best == (0, 0):
        raise ValueError("no speech-free frames found")
    return best


def musical_noise_stats(gain: np.ndarray, mag: np.ndarray, quiet: np.ndarray,
                        beta: float) -> dict[str, float]:
    """Residual level and surviving-bin fraction over speech-free frames.

    residual_db: 10 log10( sum |S_hat|^2 / sum |X|^2 ) over those frames.
    surviving_frac: fraction of their bins not pushed down to the floor
    (G > beta); with beta = 0 these are exactly the isolated peaks that
    make up musical noise.
    """
    g, x = gain[quiet], mag[quiet]
    return {"residual_db": float(10 * np.log10(np.sum((g * x) ** 2) / np.sum(x**2))),
            "surviving_frac": float(np.mean(g > beta + 1e-12))}


def musical_noise_demo(ds: Dataset, results: Path, figures: Path, audio: Path,
                       perceptual: bool) -> None:
    """Step 4 demo: one sentence, white noise at 5 dB, (alpha, beta) = (1, 0) vs defaults."""
    i = EXAMPLE_SENTENCE
    m = make_mixture(ds.signals, i, "white", EXAMPLE_SNR_DB, ds.speakers)
    quiet = speech_free_frames(m.clean_padded, m.n_lead)
    mag = np.abs(stft(m.noisy))
    configs = [("ss_a1_b0", 1.0, 0.0), ("ss_a2_b0.02", config.SS_ALPHA, config.SS_BETA)]

    base = evaluate(m.clean, m.noisy[m.n_lead :], perceptual=perceptual, tag="musical/noisy")
    rows = [{"signal": "noisy", "alpha": "", "beta": "", "residual_db": 0.0,
             "surviving_frac": 1.0, **{f"{k}": v for k, v in base.items()}}]
    outputs = {}
    for name, alpha, beta in configs:
        out, gain = spectral_sub.denoise(m.noisy, config.FS, m.n_lead, alpha, beta,
                                         return_gain=True)
        outputs[name] = out
        res = evaluate(m.clean, out[m.n_lead :], perceptual=perceptual, tag=f"musical/{name}")
        rows.append({"signal": name, "alpha": alpha, "beta": beta,
                     **musical_noise_stats(gain, mag, quiet, beta), **res})
    for r in rows:
        for k in ("segsnr", "pesq", "stoi"):
            r[f"d_{k}"] = r[k] - base[k]
    _write_csv(results / "musical_noise.csv", rows)
    for r in rows:
        log.info("musical noise %-12s residual %6.1f dB, surviving %.3f, dsegSNR %+.2f, "
                 "dPESQ %+.3f, dSTOI %+.3f", r["signal"], r["residual_db"],
                 r["surviving_frac"], r["d_segsnr"], r["d_pesq"], r["d_stoi"])

    start, stop = longest_run(quiet)
    t_region = ((start * config.HOP - config.HOP / 2 - m.n_lead) / config.FS,
                (stop * config.HOP - config.HOP / 2 - m.n_lead) / config.FS)
    panels = [("Clean", m.clean_padded),
              (f"Noisy (white, {EXAMPLE_SNR_DB} dB)", m.noisy),
              ("SS  α = 1, β = 0", outputs["ss_a1_b0"]),
              (f"SS  α = {config.SS_ALPHA:g}, β = {config.SS_BETA:g} (defaults)",
               outputs["ss_a2_b0.02"])]
    stats = {r["signal"]: r for r in rows}
    notes = [""] + [
        f"surviving bins {100 * stats[k]['surviving_frac']:.1f} %\n"
        f"residual {stats[k]['residual_db']:+.1f} dB re noisy"
        for k in ("noisy", "ss_a1_b0", "ss_a2_b0.02")
    ]
    plots.plot_musical_noise(
        panels, m.n_lead, t_region, figures / "musical_noise.png",
        f"Musical noise in spectral subtraction: {ds.names[i]}, white noise at "
        f"{EXAMPLE_SNR_DB} dB (one shared dB scale; zoom statistics over all speech-free frames)",
        zoom_notes=notes,
    )
    wav_dir = audio / "musical_noise"
    wavs = {"clean": m.clean_padded, "noisy": m.noisy, **outputs}
    scale = write_wav_set(wav_dir, wavs)
    for name in wavs:
        info = check_wav(wav_dir / f"{name}.wav", m.noisy.size)
        log.info("wav %s: fs %d, %d samples, peak %.3f, clipped %d (common scale %.3f)",
                 info["file"], info["fs"], info["n_samples"], info["peak"],
                 info["n_clipped"], scale)


def noise_estimate_check(ds: Dataset, sentences: Sequence[int], results: Path) -> None:
    """How well the lead-in estimate D_hat(k) represents the noise (CLAUDE.md correction 7).

    Per sentence and noise type, on the scaled noise alone:
      bias_db:   mean over bins of 20 log10( D_hat(k) / mean_m |D(m, k)| ),
                 lead-in estimate against the true mean over the whole signal;
      spread_db: std over bins of that same ratio;
      frame_energy_std_db: std over frames of the noise frame energy in dB,
                 i.e. how non-stationary the noise is.
    DC and Nyquist bins and the two edge frames are excluded.
    """
    rows = []
    for si in sentences:
        for noise_type in config.NOISE_TYPES:
            m = make_mixture(ds.signals, si, noise_type, EXAMPLE_SNR_DB, ds.speakers)
            mag = np.abs(stft(m.noise_scaled))
            d_hat = spectral_sub.estimate_noise_magnitude(mag, m.n_lead)[1:-1]
            d_true = mag[1:-1].mean(axis=0)[1:-1]
            ratio_db = 20.0 * np.log10(d_hat / d_true)
            frame_db = 10.0 * np.log10(np.sum(mag[1:-1] ** 2, axis=1))
            rows.append({"sentence": ds.names[si], "noise": noise_type,
                         "bias_db": float(ratio_db.mean()), "spread_db": float(ratio_db.std()),
                         "frame_energy_std_db": float(frame_db.std())})
    _write_csv(results / "noise_estimate_check.csv", rows)
    for noise_type in config.NOISE_TYPES:
        sel = [r for r in rows if r["noise"] == noise_type]
        b = [r["bias_db"] for r in sel]
        log.info("lead-in noise estimate, %s: bias %+.2f dB (range %+.2f..%+.2f), per-bin "
                 "spread %.2f dB, frame-energy std %.2f dB", noise_type, np.mean(b), min(b),
                 max(b), np.mean([r["spread_db"] for r in sel]),
                 np.mean([r["frame_energy_std_db"] for r in sel]))


# ---------------------------------------------------------------------------
# Sweep summaries
# ---------------------------------------------------------------------------
def summarise(rows: Sequence[dict], group: Sequence[str]) -> list[dict]:
    """Mean, std and n of d_segsnr, d_pesq, d_stoi (and noisy/out means) per group."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[g] for g in group), []).append(r)
    out = []
    for key, sel in groups.items():
        rec = dict(zip(group, key, strict=True))
        rec["n"] = len(sel)
        for col in ("d_segsnr", "d_pesq", "d_stoi", "segsnr_out", "pesq_out", "stoi_out",
                    "segsnr_noisy", "pesq_noisy", "stoi_noisy"):
            vals = np.array([float(r[col]) for r in sel])
            rec[f"{col}_mean"] = float(np.mean(vals))
            if col.startswith("d_"):
                rec[f"{col}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else math.nan
        out.append(rec)
    return sorted(out, key=lambda r: tuple(
        float(r[g]) if str(r[g]).lstrip("-").replace(".", "", 1).isdigit() else r[g]
        for g in group))


def run_step4(args: argparse.Namespace) -> None:
    """Step 4 (O2, O5): musical-noise demo, WAVs, alpha x beta sweep and heatmap."""
    results, figures, audio = _outputs(args)
    ds = _dataset(args)
    musical_noise_demo(ds, results, figures, audio, perceptual=not args.no_perceptual)
    sentences, snrs = _grid(args, ds)
    noise_estimate_check(ds, sentences, results)
    rows = run_sweep("spectral_sub", ss_grid(), args, ds, sentences, snrs, results)
    by_snr = summarise(rows, ("noise", "snr_in", "alpha", "beta"))
    _write_csv(results / "ss_sweep_summary.csv", by_snr)
    overall = summarise(rows, ("noise", "alpha", "beta"))
    _write_csv(results / "ss_sweep_summary_all_snr.csv", overall)
    lines = [f"{'noise':7s} {'alpha':>5s} {'beta':>6s} {'dSegSNR':>8s} {'dPESQ':>7s} "
             f"{'dSTOI':>7s}"]
    lines += [f"{r['noise']:7s} {float(r['alpha']):5g} {float(r['beta']):6g} "
              f"{r['d_segsnr_mean']:+8.2f} {r['d_pesq_mean']:+7.3f} {r['d_stoi_mean']:+7.3f}"
              for r in overall]
    log.info("alpha x beta sweep, mean over %d sentences x SNRs %s:\n%s", len(sentences),
             list(snrs), "\n".join(lines))
    if EXAMPLE_SNR_DB in snrs:
        plots.plot_alpha_beta_heatmap(
            [r for r in by_snr if r["noise"] == "white" and int(r["snr_in"]) == EXAMPLE_SNR_DB],
            figures / "alpha_beta_heatmap.png",
            f"Spectral subtraction α × β sweep: white noise at {EXAMPLE_SNR_DB} dB "
            f"(mean over {len(sentences)} sentences)",
        )


def run_step5(args: argparse.Namespace) -> None:
    """Step 5 (O2): Wiener eta sweep and figures/eta_sweep.png."""
    results, figures, _ = _outputs(args)
    ds = _dataset(args)
    sentences, snrs = _grid(args, ds)
    rows = run_sweep("wiener", wiener_grid(), args, ds, sentences, snrs, results)
    by_snr = summarise(rows, ("noise", "snr_in", "eta"))
    _write_csv(results / "wiener_sweep_summary.csv", by_snr)
    overall = summarise(rows, ("noise", "eta"))
    _write_csv(results / "wiener_sweep_summary_all_snr.csv", overall)
    lines = [f"{'noise':7s} {'eta':>5s} {'dSegSNR':>8s} {'dPESQ':>7s} {'dSTOI':>7s}"]
    lines += [f"{r['noise']:7s} {float(r['eta']):5g} {r['d_segsnr_mean']:+8.2f} "
              f"{r['d_pesq_mean']:+7.3f} {r['d_stoi_mean']:+7.3f}" for r in overall]
    log.info("Wiener eta sweep, mean over %d sentences x SNRs %s:\n%s", len(sentences),
             list(snrs), "\n".join(lines))
    if EXAMPLE_SNR_DB in snrs:
        plots.plot_eta_sweep(
            [r for r in by_snr if int(r["snr_in"]) == EXAMPLE_SNR_DB],
            figures / "eta_sweep.png",
            f"Wiener filter η sweep at {EXAMPLE_SNR_DB} dB input SNR "
            f"(mean ± 1 std over {len(sentences)} sentences)",
        )


# ---------------------------------------------------------------------------
# Step 6: transient preservation (O5)
# ---------------------------------------------------------------------------
def _s2n(seconds: float) -> int:
    return int(round(seconds * config.FS))


def click_train() -> tuple[np.ndarray, np.ndarray]:
    """2 ms exponentially decaying broadband bursts every 100 ms (seeded).

    Every click uses the same burst waveform, so clicks are directly comparable.
    Returns (signal, onset sample indices).
    """
    rng = np.random.default_rng([config.SEED, config.TRANSIENT_STREAM])
    n_len = _s2n(config.CLICK_LEN_S)
    burst = rng.standard_normal(n_len) * np.exp(-np.arange(n_len) / (config.CLICK_DECAY_S
                                                                      * config.FS))
    x = np.zeros(_s2n(config.CLICK_TRAIN_S))
    onsets = np.arange(_s2n(config.CLICK_FIRST_S), x.size - n_len, _s2n(config.CLICK_PERIOD_S))
    for o in onsets:
        x[o : o + n_len] = burst
    return x, onsets


def find_plosive_onset(signals: Sequence[np.ndarray]) -> tuple[int, int, float]:
    """Largest short-term energy jump that follows a quiet gap, over all sentences.

    Energy is taken in non-overlapping ONSET_FRAME_S frames; a frame is quiet
    when it is more than |SEGSNR_GATE_DB| below the loudest frame of its
    sentence. Returns (sentence index, onset sample, jump in dB).
    """
    win = _s2n(config.ONSET_FRAME_S)
    need = _s2n(config.ONSET_QUIET_GAP_S) // win
    best = None
    for si, x in enumerate(signals):
        n = x.size // win
        if n <= need:  # too short to hold a quiet gap followed by an onset
            continue
        with np.errstate(divide="ignore"):
            e_db = 10.0 * np.log10(np.sum(x[: n * win].reshape(n, win) ** 2, axis=1))
        quiet = e_db < np.max(e_db) + config.SEGSNR_GATE_DB
        for i in range(need, n):
            if quiet[i - need : i].all() and not quiet[i]:
                jump = float(e_db[i] - e_db[i - 1])
                if best is None or jump > best[2]:
                    best = (si, i * win, jump)
    if best is None:
        raise ValueError("no onset after a quiet gap found")
    return best


def transient_measures(out: np.ndarray, clean: np.ndarray, onsets: Sequence[int]) -> dict:
    """Per-event concentration and pre-echo, averaged over events.

    concentration = E_out(|n - o| < core) / E_out(|n - o| < wide)
    pre_echo_db   = 10 log10( E_out[o - pre, o) / E_clean(|n - o| < core) )
    core_db       = 10 log10( E_out(|n - o| < core) / E_clean(|n - o| < core) )
    Energies of the clean event are the reference, so methods that attenuate
    the event are not rewarded with a better pre-echo ratio.
    """
    core, wide, pre = (_s2n(config.TRANSIENT_CORE_S), _s2n(config.TRANSIENT_WIDE_S),
                       _s2n(config.TRANSIENT_PRE_S))

    def energy(x, a, b):
        return float(np.sum(x[max(a, 0) : b] ** 2))

    conc, pre_db, core_db = [], [], []
    with np.errstate(divide="ignore"):
        for o in onsets:
            e_ref = energy(clean, o - core, o + core)
            e_core = energy(out, o - core, o + core)
            conc.append(e_core / energy(out, o - wide, o + wide))
            pre_db.append(10.0 * np.log10(energy(out, o - pre, o) / e_ref))
            core_db.append(10.0 * np.log10(e_core / e_ref))
    def sd(v):
        # The clean reference has no energy before its onsets (pre-echo = -inf):
        # its spread is undefined, not a number to compute.
        return float(np.std(v, ddof=1)) if len(v) > 1 and np.all(np.isfinite(v)) else math.nan

    return {"n_events": len(onsets),
            "concentration_mean": float(np.mean(conc)), "concentration_std": sd(conc),
            "pre_echo_db_mean": float(np.mean(pre_db)), "pre_echo_db_std": sd(pre_db),
            "core_energy_db_mean": float(np.mean(core_db))}


def click_envelope_db(sig: np.ndarray, onsets: Sequence[int], ref_energy: float
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Click-synchronous energy envelope in dB re the clean click's peak.

    Averages sig^2 over all clicks aligned at their onsets, smooths it with an
    ENVELOPE_SMOOTH_S moving average and returns (time in ms re onset, dB).
    Pre-echo shows up as energy before 0 ms; smearing as a wider peak.
    """
    half = _s2n(config.ENVELOPE_HALF_S)
    seg = np.mean([sig[o - half : o + half] ** 2 for o in onsets], axis=0)
    k = _s2n(config.ENVELOPE_SMOOTH_S)
    smooth = np.convolve(seg, np.ones(k) / k, mode="same")
    with np.errstate(divide="ignore"):
        env_db = 10.0 * np.log10(smooth / ref_energy)
    return (np.arange(-half, half) * 1000.0 / config.FS), env_db


def run_transient_experiment(ds: Dataset, results: Path, figures: Path) -> None:
    """Click train in white noise and a real plosive: concentration and pre-echo."""
    configs = default_configs()
    rows, traces = [], {}

    clicks, onsets = click_train()
    noise = np.random.default_rng([config.SEED, config.TRANSIENT_STREAM, 1]).standard_normal(
        clicks.size + config.N_LEAD)
    noisy_c, _, _ = mix_at_snr(clicks, noise, config.CLICK_SNR_DB, config.N_LEAD)

    si, p_onset, jump = find_plosive_onset(ds.signals)
    m = make_mixture(ds.signals, si, "white", config.PLOSIVE_SNR_DB, ds.speakers)
    log.info("plosive onset: %s at %.3f s (energy jump %.1f dB after a >= %g ms quiet gap)",
             ds.names[si], p_onset / config.FS, jump, 1000 * config.ONSET_QUIET_GAP_S)

    cases = {
        "click_train": (clicks, noisy_c, list(onsets),
                        f"click train, white {config.CLICK_SNR_DB:g} dB"),
        "plosive": (m.clean, m.noisy, [p_onset],
                    f"{ds.names[si]} plosive at {p_onset / config.FS:.3f} s, "
                    f"white {config.PLOSIVE_SNR_DB} dB"),
    }
    for case, (clean, noisy, evts, desc) in cases.items():
        signals = {"clean": clean, "noisy": noisy[config.N_LEAD :]}
        for cfg in configs:
            out = DENOISERS[cfg.method](noisy, config.FS, config.N_LEAD, **cfg.kwargs())
            signals[cfg.plot_key()] = out[config.N_LEAD :]
        for key, sig in signals.items():
            rows.append({"case": case, "description": desc, "signal": key,
                         **transient_measures(sig, clean, evts)})
        traces[case] = (signals, evts, desc)
    _write_csv(results / "transient_metrics.csv", rows)
    for r in rows:
        log.info("transient %-11s %-12s concentration %.3f ± %.3f, pre-echo %7.1f dB, "
                 "core energy %+6.1f dB", r["case"], r["signal"], r["concentration_mean"],
                 r["concentration_std"] if not math.isnan(r["concentration_std"]) else 0.0,
                 r["pre_echo_db_mean"], r["core_energy_db_mean"])
    signals, evts, _ = traces["click_train"]
    half, k = _s2n(config.ENVELOPE_HALF_S), _s2n(config.ENVELOPE_SMOOTH_S)
    clean_peak = float(np.max(np.convolve(signals["clean"][evts[0] - half : evts[0] + half] ** 2,
                                          np.ones(k) / k, mode="same")))
    envelopes = {key: click_envelope_db(sig, evts, clean_peak) for key, sig in signals.items()}
    t_ms = next(iter(envelopes.values()))[0]
    _write_csv(results / "transient_envelope.csv",
               [{"t_ms": t, **{k: float(env[i]) for k, (_, env) in envelopes.items()}}
                for i, t in enumerate(t_ms)])
    plots.plot_transient_closeup(
        traces, config.CLICK_PLOT_PAIR, figures / "transient_closeup.png", envelopes,
        "Transient preservation: time-domain close-ups at the onsets (dashed); "
        "shaded = 10 ms pre-echo window",
    )


def wavelet_band_check(ds: Dataset, results: Path) -> None:
    """Oracle breakdown of what the default wavelet thresholds keep, per detail level.

    Uses the known clean and noise components of two cases: the example
    sentence in white noise at EXAMPLE_SNR_DB, and the plosive case. For each
    variant and level: clean and noise energy in that band, the threshold, and
    the percentage of noisy coefficients above it (i.e. not zeroed), overall
    and within the TRANSIENT_WIDE_S window around the plosive onset.
    """
    si_p, onset, _ = find_plosive_onset(ds.signals)
    cases = [("example", EXAMPLE_SENTENCE, EXAMPLE_SNR_DB, None),
             ("plosive", si_p, config.PLOSIVE_SNR_DB, onset)]
    wide = _s2n(config.TRANSIENT_WIDE_S)
    rows = []
    for case, si, snr, ev in cases:
        m = make_mixture(ds.signals, si, "white", snr, ds.speakers)
        cs = wavelet.analysis(m.clean_padded, config.WAVELET, config.WAVELET_LEVEL)
        cn = wavelet.analysis(m.noise_scaled, config.WAVELET, config.WAVELET_LEVEL)
        for variant in config.WAVELET_VARIANTS:
            _, info = wavelet.denoise(m.noisy, config.FS, m.n_lead, variant=variant,
                                      return_details=True)
            lev_top = info["level"]
            rows.append({"case": case, "sentence": ds.names[si], "snr_in": snr,
                         "variant": variant, "band": f"cA{lev_top}", "lambda": "",
                         "clean_energy_db": 10 * np.log10(np.sum(cs[0] ** 2)),
                         "noise_energy_db": 10 * np.log10(np.sum(cn[0] ** 2)),
                         "pct_kept": 100.0, "pct_kept_near_onset": ""})
            for j, (d, lam) in enumerate(zip(info["coeffs"][1:], info["lambdas"], strict=True)):
                lev = lev_top - j
                kept = np.abs(d) > lam
                near = ""
                if ev is not None:
                    c = m.n_lead + ev
                    a, b = (c - wide) // 2**lev, (c + wide) // 2**lev + 1
                    near = 100.0 * float(np.mean(kept[a:b]))
                rows.append({"case": case, "sentence": ds.names[si], "snr_in": snr,
                             "variant": variant, "band": f"cD{lev}", "lambda": lam,
                             "clean_energy_db": 10 * np.log10(np.sum(cs[j + 1] ** 2)),
                             "noise_energy_db": 10 * np.log10(np.sum(cn[j + 1] ** 2)),
                             "pct_kept": 100.0 * float(np.mean(kept)),
                             "pct_kept_near_onset": near})
    _write_csv(results / "wavelet_band_check.csv", rows)


def run_step6(args: argparse.Namespace) -> None:
    """Step 6 (O2, O5): transient experiment, wavelet family x level x variant sweep."""
    results, figures, _ = _outputs(args)
    ds = _dataset(args)
    run_transient_experiment(ds, results, figures)
    wavelet_band_check(ds, results)
    sentences, snrs = _grid(args, ds)
    rows = run_sweep("wavelet", wavelet_grid(), args, ds, sentences, snrs, results)
    by_snr = summarise(rows, ("noise", "snr_in", "variant", "wavelet", "level"))
    _write_csv(results / "wavelet_sweep_summary.csv", by_snr)
    overall = summarise(rows, ("noise", "variant", "wavelet", "level"))
    _write_csv(results / "wavelet_sweep_summary_all_snr.csv", overall)
    lines = [f"{'noise':7s} {'variant':16s} {'wav':4s} {'L':>2s} {'dSegSNR':>8s} {'dPESQ':>7s} "
             f"{'dSTOI':>7s}"]
    lines += [f"{r['noise']:7s} {r['variant']:16s} {r['wavelet']:4s} {int(r['level']):2d} "
              f"{r['d_segsnr_mean']:+8.2f} {r['d_pesq_mean']:+7.3f} {r['d_stoi_mean']:+7.3f}"
              for r in overall]
    log.info("wavelet sweep, mean over %d sentences x SNRs %s:\n%s", len(sentences),
             list(snrs), "\n".join(lines))
    if EXAMPLE_SNR_DB in snrs:
        plots.plot_wavelet_sweep(
            [r for r in by_snr if int(r["snr_in"]) == EXAMPLE_SNR_DB],
            figures / "wavelet_sweep.png",
            f"Wavelet thresholding sweep at {EXAMPLE_SNR_DB} dB input SNR: mean ΔPESQ "
            f"over {len(sentences)} sentences (one shared colour scale)",
        )


# ---------------------------------------------------------------------------
# Step 7: aggregation, summary tables, figures, audio, sanity checks
# ---------------------------------------------------------------------------
SWEEPS: dict[str, Callable[[], list[MethodConfig]]] = {
    "spectral_sub": ss_grid, "wiener": wiener_grid, "wavelet": wavelet_grid,
}
AUDIO_CASES = (("white", 0), ("white", 10), ("babble", 0), ("babble", 10))
METRICS = ("segsnr", "pesq", "stoi")


def _cfg_matches(row: dict, cfg: MethodConfig) -> bool:
    return all(str(row[k]) == str(v) for k, v in cfg.fields().items())


def _expected_keys(configs, ds, sentences, snrs) -> set:
    return {_key_str({"sentence": ds.names[si], "noise": n, "snr_in": snr, **c.fields()})
            for si in sentences for n in config.NOISE_TYPES for snr in snrs for c in configs}


def ensure_runs(args, ds, sentences, snrs, results) -> list[dict]:
    """Rows of every sweep for this run mode; runs (or resumes) any sweep whose
    CSV is missing or does not hold exactly the expected rows."""
    rows = []
    for name, grid in SWEEPS.items():
        existing = _read_rows(results / f"runs_{name}.csv")
        if {_key_str(r) for r in existing} == _expected_keys(grid(), ds, sentences, snrs) \
                and len(existing) == len(_expected_keys(grid(), ds, sentences, snrs)):
            rows += existing
        else:
            log.info("runs_%s.csv missing or incomplete: running that sweep", name)
            rows += run_sweep(name, grid(), argparse.Namespace(**{**vars(args), "resume": True}),
                              ds, sentences, snrs, results)
    return sorted(rows, key=_sort_key)


def default_summary(rows: Sequence[dict]) -> list[dict]:
    """Mean, std and n of every metric per default method x noise x SNR."""
    out = []
    for cfg in default_configs():
        sel = [r for r in rows if _cfg_matches(r, cfg)]
        for noise_type in config.NOISE_TYPES:
            for snr in sorted({int(r["snr_in"]) for r in sel}):
                grp = [r for r in sel if r["noise"] == noise_type and int(r["snr_in"]) == snr]
                rec = {"method": cfg.plot_key(), "config": cfg.label(), "noise": noise_type,
                       "snr_in": snr, "n": len(grp)}
                for m in METRICS:
                    for col in (f"{m}_noisy", f"{m}_out", f"d_{m}"):
                        v = np.array([float(r[col]) for r in grp])
                        rec[f"{col}_mean"] = float(np.mean(v))
                        rec[f"{col}_std"] = float(np.std(v, ddof=1))
                out.append(rec)
    return out


def write_summary_md(summary: Sequence[dict], path: Path, n_sentences: int) -> None:
    lines = ["# Results at the default parameters",
             "",
             f"Mean ± 1 std over {n_sentences} sentences per cell (source: `summary.csv`, "
             "built from `all_runs.csv`). Metrics are computed on the sentence only "
             "(lead-in trimmed). PESQ is narrowband MOS-LQO.",
             "",
             "Defaults: " + "; ".join(f"{c.plot_key()} = {c.label()}" for c in default_configs()),
             ""]
    for noise_type in config.NOISE_TYPES:
        lines += [f"## {noise_type.capitalize()} noise", "",
                  "| SNR in (dB) | Method | segSNR (dB) | ΔsegSNR (dB) | PESQ | ΔPESQ "
                  "| STOI | ΔSTOI |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|"]
        sel = [r for r in summary if r["noise"] == noise_type]
        for snr in sorted({r["snr_in"] for r in sel}):
            grp = [r for r in sel if r["snr_in"] == snr]
            b = grp[0]
            lines.append(f"| {snr} | noisy input | {b['segsnr_noisy_mean']:.2f} ± "
                         f"{b['segsnr_noisy_std']:.2f} | | {b['pesq_noisy_mean']:.2f} ± "
                         f"{b['pesq_noisy_std']:.2f} | | {b['stoi_noisy_mean']:.3f} ± "
                         f"{b['stoi_noisy_std']:.3f} | |")
            for r in grp:
                lines.append(
                    f"| {snr} | {plots.METHOD_LABELS[r['method']]} | "
                    f"{r['segsnr_out_mean']:.2f} ± {r['segsnr_out_std']:.2f} | "
                    f"{r['d_segsnr_mean']:+.2f} ± {r['d_segsnr_std']:.2f} | "
                    f"{r['pesq_out_mean']:.2f} ± {r['pesq_out_std']:.2f} | "
                    f"{r['d_pesq_mean']:+.2f} ± {r['d_pesq_std']:.2f} | "
                    f"{r['stoi_out_mean']:.3f} ± {r['stoi_out_std']:.3f} | "
                    f"{r['d_stoi_mean']:+.3f} ± {r['d_stoi_std']:.3f} |")
        lines.append("")
    path.write_text("\n".join(lines))


def write_config_used(path: Path, args, ds: Dataset, sentences, snrs) -> None:
    params = {}
    for name in sorted(dir(config)):
        v = getattr(config, name)
        if name.isupper() and isinstance(v, (int, float, str, tuple, Path)):
            params[name] = str(v.relative_to(config.ROOT)) if isinstance(v, Path) else v
    run = {"quick": args.quick, "synthetic": args.synthetic,
           "perceptual": not args.no_perceptual,
           "dataset": "synthetic" if ds.synthetic else "NOIZEUS",
           "sentences": [ds.names[i] for i in sentences], "snrs_db": list(snrs),
           "default_configs": [c.label() for c in default_configs()],
           "sweep_sizes": {k: len(g()) for k, g in SWEEPS.items()}}
    path.write_text(json.dumps({"config": params, "run": run}, indent=2, default=list) + "\n")


def write_environment(path: Path) -> None:
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                            text=True, check=True).stdout
    path.write_text(f"# Python {platform.python_version()}, {platform.system()} "
                    f"{platform.machine()}\n{freeze}")


def metric_disagreement(rows: Sequence[dict], results: Path) -> dict:
    """O4: cases where the fidelity metric improves but the perceptual one gets worse.

    Writes metric_disagreement.csv (per configuration and noise: how many of its
    cases have dsegSNR > 0 and dPESQ < 0, and dsegSNR > 0 and dSTOI < 0) and
    metric_disagreement_examples.csv (for each method, the five disagreeing
    cases with the largest dsegSNR). Returns the overall counts.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in RUN_KEY[3:]) + (r["noise"],), []).append(r)
    out = []
    for key, sel in groups.items():
        up = [float(r["d_segsnr"]) > 0 for r in sel]
        pesq_down = [float(r["d_pesq"]) < 0 for r in sel]
        stoi_down = [float(r["d_stoi"]) < 0 for r in sel]
        n_p = sum(u and p for u, p in zip(up, pesq_down, strict=True))
        n_s = sum(u and s for u, s in zip(up, stoi_down, strict=True))
        out.append({**dict(zip(RUN_KEY[3:] + ("noise",), key, strict=True)), "n_cases": len(sel),
                    "n_segsnr_up_pesq_down": n_p, "frac_segsnr_up_pesq_down": n_p / len(sel),
                    "n_segsnr_up_stoi_down": n_s, "frac_segsnr_up_stoi_down": n_s / len(sel)})
    out.sort(key=lambda r: _sort_key({"sentence": "", "noise": r["noise"], "snr_in": "", **r}))
    total = {"n_points": len(rows),
             "n_segsnr_up_pesq_down": sum(r["n_segsnr_up_pesq_down"] for r in out),
             "n_segsnr_up_stoi_down": sum(r["n_segsnr_up_stoi_down"] for r in out)}
    _write_csv(results / "metric_disagreement.csv", out)

    examples = []
    for method in SWEEPS:
        dis = [r for r in rows if r["method"] == method and float(r["d_segsnr"]) > 0
               and float(r["d_pesq"]) < 0]
        dis.sort(key=lambda r: (-float(r["d_segsnr"]), _sort_key(r)))
        examples += [{k: r[k] for k in RUN_COLUMNS} for r in dis[:5]]
    _write_or_remove(results / "metric_disagreement_examples.csv", examples)
    return total


def write_example_audio(ds: Dataset, audio: Path) -> list[dict]:
    """clean, noisy and every default method for AUDIO_CASES (own scale per case)."""
    infos = []
    i = EXAMPLE_SENTENCE
    for noise_type, snr in AUDIO_CASES:
        m = make_mixture(ds.signals, i, noise_type, snr, ds.speakers)
        sigs = {"clean": m.clean_padded, "noisy": m.noisy}
        for cfg in default_configs():
            sigs[cfg.plot_key()] = DENOISERS[cfg.method](m.noisy, config.FS, m.n_lead,
                                                         **cfg.kwargs())
        folder = audio / "examples" / f"{ds.names[i]}_{noise_type}_{snr}dB"
        scale = write_wav_set(folder, sigs)
        for name in sigs:
            infos.append({**check_wav(folder / f"{name}.wav", m.noisy.size), "scale": scale})
    return infos


def denoise_file(path: Path, audio_root: Path | None = None) -> Path:
    """Denoise one recording with every default method (``python main.py --input FILE``).

    The file is converted to mono at config.FS (data.load_audio_file).
    Spectral subtraction and the Wiener filter estimate the noise from the
    first config.LEAD_IN_S seconds, so the recording must start with noise
    only (no speech). There is no clean reference, so no metrics are
    computed. Writes <audio_root>/input/<file stem>/input.wav (the converted
    input) and one WAV per method, all scaled by one common factor.
    Returns that folder.
    """
    path = Path(path)
    x = load_audio_file(path)
    min_samples = config.N_LEAD + config.FRAME_LEN
    if x.size < min_samples:
        raise ValueError(
            f"{path.name} is {x.size / config.FS:.2f} s long; need at least "
            f"{min_samples / config.FS:.2f} s (a {config.LEAD_IN_S} s noise-only start plus speech)"
        )
    log.info("%s: %.2f s, mono, %d Hz; noise is estimated from the first %.2f s, which must "
             "contain no speech", path.name, x.size / config.FS, config.FS, config.LEAD_IN_S)
    sigs = {"input": x}
    for cfg in default_configs():
        sigs[cfg.plot_key()] = DENOISERS[cfg.method](x, config.FS, config.N_LEAD, **cfg.kwargs())
    folder = (audio_root or config.AUDIO_DIR) / "input" / path.stem
    write_wav_set(folder, sigs)
    for name in sigs:
        check_wav(folder / f"{name}.wav", x.size)
    return folder


def csv_hashes(results: Path) -> dict[str, str]:
    """sha256 of every results CSV except the (non-reproducible) timing files."""
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(results.glob("*.csv")) if not p.name.startswith("timing_")}


def _cfg_from_row(r: dict) -> MethodConfig:
    def num(v, cast):
        return None if v == "" else cast(v)

    return MethodConfig(r["method"], r["variant"], num(r["alpha"], float), num(r["beta"], float),
                        num(r["eta"], float), r["wavelet"], num(r["level"], int))


def pesq_stability_check(rows: Sequence[dict], ds: Dataset, results: Path) -> list[dict]:
    """Re-score rows where PESQ rises strongly while STOI falls strongly.

    Each suspect output is rebuilt and PESQ is recomputed with the output
    delayed by one sample (a zero prepended, last sample dropped). A valid
    score changes little; a jump above PESQ_UNSTABLE_JUMP marks a PESQ
    alignment failure (seen on outputs with almost no in-band speech).
    The rows are not altered or dropped; this only annotates them.
    """
    suspects = [r for r in rows if float(r["d_pesq"]) > config.PESQ_SUSPECT_D_PESQ
                and float(r["d_stoi"]) < config.PESQ_SUSPECT_D_STOI]
    out_rows = []
    for r in suspects:
        cfg = _cfg_from_row(r)
        si = ds.names.index(r["sentence"])
        m = make_mixture(ds.signals, si, r["noise"], int(r["snr_in"]), ds.speakers)
        out = DENOISERS[cfg.method](m.noisy, config.FS, m.n_lead, **cfg.kwargs())[m.n_lead :]
        shifted = np.concatenate([[0.0], out[:-1]])
        p0 = pesq_nb(m.clean, out, tag=f"{r['sentence']}/{cfg.label()}")
        p1 = pesq_nb(m.clean, shifted, tag=f"{r['sentence']}/{cfg.label()}/shift1")
        assert repr(p0) == r["pesq_out"], (p0, r["pesq_out"])  # rebuilt run == stored run
        out_rows.append({**{k: r[k] for k in RUN_KEY}, "pesq_noisy": r["pesq_noisy"],
                         "pesq_out": p0, "pesq_out_shift1": p1, "stoi_noisy": r["stoi_noisy"],
                         "stoi_out": r["stoi_out"], "segsnr_out": r["segsnr_out"],
                         "unstable": abs(p0 - p1) > config.PESQ_UNSTABLE_JUMP})
    _write_or_remove(results / "pesq_stability_check.csv", out_rows)
    return out_rows


def sanity_checks(rows, summary, ds, sentences, snrs, results, perceptual,
                  audio_infos, hashes_now, hashes_prev,
                  pesq_suspects=()) -> list[tuple[str, str, str]]:
    """BUILD_PROMPT Step 7 sanity checks: (check, PASS/FLAG, evidence)."""
    checks = []

    def add(name, ok, evidence, flag_note=""):
        checks.append((name, "PASS" if ok else "FLAG", evidence + (f" {flag_note}" if not ok
                                                                   and flag_note else "")))

    n_exp = sum(len(_expected_keys(g(), ds, sentences, snrs)) for g in SWEEPS.values())
    add("Row count", len(rows) == n_exp, f"all_runs.csv has {len(rows)} rows, expected {n_exp}.")
    keys = [_key_str(r) for r in rows]
    add("No duplicate rows", len(set(keys)) == len(keys), f"{len(set(keys))} unique keys.")
    if perceptual:
        n_nan = sum(math.isnan(float(r[c])) for r in rows for c in ("pesq_noisy", "pesq_out"))
        add("PESQ NaN count", n_nan == 0, f"{n_nan} NaN PESQ values.")

    cases = {(r["sentence"], r["noise"], r["snr_in"]): r for r in rows}.values()
    for m in (METRICS if perceptual else ("segsnr",)):
        for noise_type in config.NOISE_TYPES:
            means = [float(np.mean([float(c[f"{m}_noisy"]) for c in cases
                                    if c["noise"] == noise_type and int(c["snr_in"]) == s]))
                     for s in snrs]
            ok = all(b > a for a, b in zip(means, means[1:], strict=False))
            add(f"Noisy-input {m} rises with input SNR ({noise_type})", ok,
                "means " + ", ".join(f"{s} dB: {v:.3f}" for s, v in zip(snrs, means,
                                                                         strict=True)) + ".",
                "Would indicate a mixing bug.")

    seg = [float(r[c]) for r in rows for c in ("segsnr_noisy", "segsnr_out")]
    add("segSNR in [-10, 35]",
        min(seg) >= config.SEGSNR_MIN_DB and max(seg) <= config.SEGSNR_MAX_DB,
        f"range [{min(seg):.2f}, {max(seg):.2f}] dB.")
    if perceptual:
        st = [float(r[c]) for r in rows for c in ("stoi_noisy", "stoi_out")]
        neg = [r for r in rows if min(float(r["stoi_noisy"]), float(r["stoi_out"])) < 0]
        neg_cfgs = sorted({f"{r['sentence']} {r['noise']} {r['snr_in']} dB "
                           f"{MethodConfig(r['method'], r['variant'], wavelet=r['wavelet']).method}"
                           f" {r['variant']} {r['wavelet']} L{r['level']} "
                           f"(STOI {float(r['stoi_out']):.3f}, PESQ {float(r['pesq_out']):.2f})"
                           for r in neg})
        add("STOI in [0, 1]", min(st) >= 0 and max(st) <= 1,
            f"range [{min(st):.3f}, {max(st):.3f}]; {len(neg)} of {len(rows)} rows below 0.",
            "STOI is a mean of short-time correlation coefficients (Taal et al. 2011), so its "
            "mathematical range is [-1, 1]; values just below 0 mean the output keeps no "
            "intelligible envelope. Rows: " + "; ".join(neg_cfgs) + ". Not a pipeline bug.")
        pq = [float(r[c]) for r in rows for c in ("pesq_noisy", "pesq_out")]
        add("PESQ in library range [1.0, 4.56]", min(pq) >= 1.0 and max(pq) <= 4.56,
            f"range [{min(pq):.3f}, {max(pq):.3f}].")
        unstable = [r for r in pesq_suspects if r["unstable"]]
        detail = "; ".join(
            f"{r['sentence']} {r['noise']} {r['snr_in']} dB {_cfg_from_row(r).label()}: "
            f"PESQ {r['pesq_out']:.2f} -> {r['pesq_out_shift1']:.2f} when delayed 1 sample "
            f"(STOI {float(r['stoi_out']):.2f})" for r in unstable)
        add("PESQ plausibility (large PESQ gain with large STOI loss)", not unstable,
            f"{len(pesq_suspects)} of {len(rows)} rows have ΔPESQ > "
            f"{config.PESQ_SUSPECT_D_PESQ:g} and ΔSTOI < {config.PESQ_SUSPECT_D_STOI:g}; "
            f"{len(unstable)} of them change PESQ by more than {config.PESQ_UNSTABLE_JUMP:g} "
            "when the output is delayed by one sample (pesq_stability_check.csv).",
            "Those scores are PESQ alignment failures on outputs with almost no in-band speech "
            "(mostly the untouched < 125 Hz wavelet approximation band), not valid quality "
            "estimates. Rows are kept unaltered in all_runs.csv; they inflate the mean and std "
            "of those cells and are themselves an O4 finding. Unstable rows: " + detail + ".")

    for method in ("spectral_sub", "wiener"):
        for snr in [s for s in snrs if s <= 5]:
            r = next(x for x in summary if x["method"] == method and x["noise"] == "white"
                     and x["snr_in"] == snr)
            add(f"{method} default: mean ΔsegSNR > 0, white {snr} dB", r["d_segsnr_mean"] > 0,
                f"mean ΔsegSNR {r['d_segsnr_mean']:+.2f} dB (n = {r['n']}).",
                "Suspect an alignment, gain or phase bug.")

    lags = []
    for si in list(sentences)[:3]:
        for noise_type in config.NOISE_TYPES:
            m = make_mixture(ds.signals, si, noise_type, EXAMPLE_SNR_DB, ds.speakers)
            for cfg in default_configs():
                out = DENOISERS[cfg.method](m.noisy, config.FS, m.n_lead, **cfg.kwargs())
                lags.append(find_lag(m.clean, out[m.n_lead :]))
    add("Output alignment (check_alignment sample)", all(v == 0 for v in lags),
        f"{len(lags)} outputs (3 sentences x 2 noises x {len(default_configs())} methods at "
        f"{EXAMPLE_SNR_DB} dB): lags {sorted(set(lags))}.")

    for method in [c.plot_key() for c in default_configs()]:
        for m in (METRICS if perceptual else ("segsnr",)):
            w = np.mean([r[f"d_{m}_mean"] for r in summary if r["method"] == method
                         and r["noise"] == "white"])
            b = np.mean([r[f"d_{m}_mean"] for r in summary if r["method"] == method
                         and r["noise"] == "babble"])
            note = "Reported as a finding, not fixed."
            if method == "wavelet":
                note += (" The universal lambda is set from cD1 (2-4 kHz), where babble has "
                         "little energy, so it barely changes babble mixtures; on white noise "
                         "the same rule removes most speech detail (wavelet_band_check.csv). "
                         "Both Δ are negative, so 'smaller improvement' is not meaningful here.")
            add(f"Babble Δ{m} smaller than white ({method})", b < w,
                f"mean over SNRs: white {w:+.3f}, babble {b:+.3f}.", note)

    if min(snrs) < 15 <= max(snrs):
        lo = min(snrs)
        floor_note = {}
        for noise_type in config.NOISE_TYPES:
            clamp, pq = {}, {}
            for snr in (lo, 15):
                fr = []
                for si in sentences:
                    m = make_mixture(ds.signals, si, noise_type, snr, ds.speakers)
                    fr.append(np.mean(seg_snr_frames(m.clean, m.noisy[m.n_lead :])
                                      == config.SEGSNR_MIN_DB))
                clamp[snr] = 100.0 * float(np.mean(fr))
                pq[snr] = float(np.mean([float(c["pesq_noisy"]) for c in cases
                                         if c["noise"] == noise_type and int(c["snr_in"]) == snr]))
            floor_note[noise_type] = (
                f"Metric floor effect: at {lo} dB, {clamp[lo]:.1f} % of noisy-input frames sit "
                f"at the {config.SEGSNR_MIN_DB:g} dB segSNR clamp (vs {clamp[15]:.1f} % at "
                f"15 dB) and mean noisy PESQ is {pq[lo]:.2f} (MOS-LQO floor ~1.0) vs "
                f"{pq[15]:.2f} at 15 dB, so both metrics have little room to move at low SNR "
                "and Δ is compressed there. Reported as a finding, not a bug.")
        ld_note = {}
        for noise_type in config.NOISE_TYPES:
            stats = {}
            for snr in (lo, 15):
                ratios = []
                for si in sentences:
                    m = make_mixture(ds.signals, si, noise_type, snr, ds.speakers)
                    _, info = wavelet.denoise(m.noisy, config.FS, m.n_lead,
                                              variant="level_dependent", return_details=True)
                    true = wavelet.thresholds(
                        wavelet.analysis(m.noise_scaled, config.WAVELET, info["level"])[1:],
                        m.noisy.size, "level_dependent")
                    ratios.append(np.array(info["lambdas"]) / np.array(true))
                r = np.array(ratios)
                stats[snr] = (r.mean(axis=0).min(), r.mean(axis=0).max(),
                              (r.std(axis=0) / r.mean(axis=0)).max())
            ld_note[noise_type] = (
                f" Level-dependent thresholds: lambda_j / lambda_j(true noise) spans "
                f"{stats[lo][0]:.2f}-{stats[lo][1]:.2f} across levels at {lo} dB (max "
                f"across-sentence CV {100 * stats[lo][2]:.0f} %) but {stats[15][0]:.2f}-"
                f"{stats[15][1]:.2f} at 15 dB (max CV {100 * stats[15][2]:.0f} %): at high SNR "
                "the MAD is inflated by speech by a sentence-dependent amount, so how much "
                "speech is removed varies more between sentences.")
        for method in [c.plot_key() for c in default_configs()]:
            for noise_type in config.NOISE_TYPES:
                for m in (METRICS if perceptual else ("segsnr",)):
                    def sd(s, method=method, noise_type=noise_type, m=m):
                        return next(r[f"d_{m}_std"] for r in summary if r["method"] == method
                                    and r["noise"] == noise_type and r["snr_in"] == s)
                    add(f"Δ{m} spread shrinks at 15 dB ({method}, {noise_type})",
                        sd(15) < sd(lo), f"std at {lo} dB {sd(lo):.3f}, at 15 dB {sd(15):.3f}.",
                        (floor_note[noise_type] if m != "stoi" else "Reported as a finding.")
                        + (ld_note[noise_type] if method == "wavelet_ld" else ""))
    else:
        checks.append(("Δ spread shrinks at 15 dB", "N/A", "15 dB not in this run's SNR set."))

    ok_audio = all(a["fs"] == config.FS and a["n_clipped"] == 0 and a["peak"] <= 1.0
                   for a in audio_infos)
    add("Example WAVs (8 kHz, peak <= 1, no clipping, expected length)", ok_audio,
        f"{len(audio_infos)} files checked; peaks {min(a['peak'] for a in audio_infos):.3f}-"
        f"{max(a['peak'] for a in audio_infos):.3f}.")

    if hashes_prev:
        same = hashes_now == hashes_prev
        diff = sorted(k for k in set(hashes_now) | set(hashes_prev)
                      if hashes_now.get(k) != hashes_prev.get(k))
        add("Reproducibility: CSV sha256 identical to the previous run of this mode", same,
            f"{len(hashes_now)} CSVs compared." if same else f"differ: {diff}.",
            "Expected only if the code changed between the two runs.")
    else:
        checks.append(("Reproducibility vs previous run", "N/A",
                       "no previous run of this mode (run it again to compare)."))
    quick_dir = config.output_dirs(quick=True, synthetic=ds.synthetic, create=False)[0]
    if quick_dir != results and (quick_dir / "csv_sha256.prev.txt").exists():
        a = (quick_dir / "csv_sha256.txt").read_text()
        b = (quick_dir / "csv_sha256.prev.txt").read_text()
        add("Reproducibility: last two `python main.py --quick` runs give identical CSV sha256",
            a == b, f"{len(a.splitlines())} CSVs compared ({quick_dir.relative_to(config.ROOT)}"
            "/csv_sha256*.txt).", "Expected only if the code changed between the two runs.")
    return checks


def run_step7(args: argparse.Namespace) -> None:
    """Step 7 (O4, O5, O6): all_runs.csv, summaries, cross-method figures, audio, checks."""
    results, figures, audio = _outputs(args)
    ds = _dataset(args)
    sentences, snrs = _grid(args, ds)
    perceptual = not args.no_perceptual
    rows = ensure_runs(args, ds, sentences, snrs, results)
    _write_csv(results / "all_runs.csv", [{k: r[k] for k in RUN_COLUMNS} for r in rows])

    summary = default_summary(rows)
    _write_csv(results / "summary.csv", summary)
    write_summary_md(summary, results / "summary.md", len(sentences))
    write_config_used(results / "config_used.json", args, ds, sentences, snrs)
    write_environment(results / "environment.txt")
    write_timing_summary(results)
    total = metric_disagreement(rows, results)
    log.info("metric disagreement: %d of %d points have dsegSNR > 0 and dPESQ < 0 "
             "(%d have dsegSNR > 0 and dSTOI < 0)", total["n_segsnr_up_pesq_down"],
             total["n_points"], total["n_segsnr_up_stoi_down"])

    for noise_type in config.NOISE_TYPES:
        plots.plot_metric_vs_snr(
            [r for r in summary if r["noise"] == noise_type],
            figures / f"metric_vs_snr_{noise_type}.png",
            f"{noise_type.capitalize()} noise: metrics against input SNR at the default "
            f"parameters (mean ± 1 std over {len(sentences)} sentences)",
            perceptual=perceptual,
        )
    grid_panels = {}
    for noise_type in config.NOISE_TYPES:
        m = make_mixture(ds.signals, EXAMPLE_SENTENCE, noise_type, EXAMPLE_SNR_DB, ds.speakers)
        panels = [("Clean", m.clean_padded), (f"Noisy ({EXAMPLE_SNR_DB} dB)", m.noisy)]
        for cfg in default_configs()[:3]:
            out = DENOISERS[cfg.method](m.noisy, config.FS, m.n_lead, **cfg.kwargs())
            panels.append((plots.METHOD_LABELS[cfg.plot_key()], out))
        grid_panels[noise_type] = panels
    plots.plot_spectrogram_grid(
        grid_panels, config.N_LEAD, figures / "spectrogram_grid.png",
        f"{ds.names[EXAMPLE_SENTENCE]} at {EXAMPLE_SNR_DB} dB input SNR, default parameters "
        "(one shared dB scale; lead-in at t < 0)",
    )
    if perceptual:
        plots.plot_fidelity_vs_perceptual(
            rows, figures / "fidelity_vs_perceptual.png",
            f"Fidelity vs perceptual metric over every sweep configuration and case "
            f"({len(rows)} points): shaded = ΔsegSNR > 0 but ΔPESQ < 0",
        )
    pesq_suspects = pesq_stability_check(rows, ds, results) if perceptual else []
    audio_infos = write_example_audio(ds, audio)
    log.info("example audio: %d WAVs checked in %s", len(audio_infos), audio / "examples")

    hashes_now = csv_hashes(results)
    hash_file = results / "csv_sha256.txt"
    prev = {}
    if hash_file.exists():
        prev = dict(line.split("  ")[::-1] for line in hash_file.read_text().splitlines())
        hash_file.replace(results / "csv_sha256.prev.txt")
    hash_file.write_text("".join(f"{h}  {n}\n" for n, h in hashes_now.items()))

    checks = sanity_checks(rows, summary, ds, sentences, snrs, results, perceptual,
                           audio_infos, hashes_now, prev, pesq_suspects)
    md = ["# Sanity checks (Step 7)", "",
          f"Run mode: {'synthetic' if ds.synthetic else 'NOIZEUS'}, "
          f"{len(sentences)} sentences, SNRs {list(snrs)} dB, "
          f"perceptual metrics {'on' if perceptual else 'off'}. Generated by "
          "`src/experiments.py: sanity_checks`.", "",
          "| Check | Result | Evidence |", "|---|---|---|"]
    md += [f"| {c} | **{s}** | {e} |" for c, s, e in checks]
    (results / "sanity_checks.md").write_text("\n".join(md) + "\n")
    n_flag = sum(s == "FLAG" for _, s, _ in checks)
    log.info("sanity checks: %d PASS, %d FLAG, %d N/A -> %s",
             sum(s == "PASS" for _, s, _ in checks), n_flag,
             sum(s == "N/A" for _, s, _ in checks), results / "sanity_checks.md")
