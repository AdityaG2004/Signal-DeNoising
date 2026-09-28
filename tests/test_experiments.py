"""Sweep engine and experiment helpers (used by Steps 4-7)."""

import argparse
import hashlib

import numpy as np
import pytest
import soundfile as sf

import config
from src import experiments as ex
from src.data import Dataset, synthetic_pool


def _args(**kw):
    base = {"quick": True, "synthetic": True, "no_perceptual": True, "workers": 1,
            "resume": False}
    return argparse.Namespace(**{**base, **kw})


@pytest.fixture(scope="module")
def tiny_ds():
    pool = synthetic_pool(8)
    return Dataset([n for n, _ in pool], [s for _, s in pool], None, True)


CONFIGS = [ex.MethodConfig("spectral_sub", alpha=1.0, beta=0.0), ex.ss_default()]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sweep_rows_and_deltas(tiny_ds, tmp_path):
    rows = ex.run_sweep("t", CONFIGS, _args(), tiny_ds, [0, 1], [0, 5], tmp_path)
    assert len(rows) == 2 * len(config.NOISE_TYPES) * 2 * len(CONFIGS)
    assert list(rows[0]) == list(ex.RUN_COLUMNS)
    for r in rows:
        d = float(r["segsnr_out"]) - float(r["segsnr_noisy"])
        assert float(r["d_segsnr"]) == pytest.approx(d, abs=1e-12)
    assert not (tmp_path / "runs_t.partial.csv").exists()
    assert (tmp_path / "timing_t.csv").exists()


def test_sweep_is_byte_identical_serial_vs_parallel(tiny_ds, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    ex.run_sweep("t", CONFIGS, _args(workers=1), tiny_ds, [0, 1], [0, 5], a)
    ex.run_sweep("t", CONFIGS, _args(workers=2), tiny_ds, [0, 1], [0, 5], b)
    assert _sha(a / "runs_t.csv") == _sha(b / "runs_t.csv")


def test_sweep_resume_fills_missing_rows(tiny_ds, tmp_path):
    ex.run_sweep("t", CONFIGS, _args(), tiny_ds, [0, 1], [0, 5], tmp_path)
    final = tmp_path / "runs_t.csv"
    full = final.read_text()
    lines = full.splitlines(keepends=True)
    final.write_text("".join(lines[:-3]))  # lose the last three rows
    rows = ex.run_sweep("t", CONFIGS, _args(resume=True), tiny_ds, [0, 1], [0, 5], tmp_path)
    assert final.read_text() == full
    assert len(rows) == len(lines) - 1
    # Regression: recomputed rows are re-timed, not duplicated, in the timing CSV.
    timing = ex._read_rows(tmp_path / "timing_t.csv")
    assert len(timing) == len(rows)
    assert len({ex._key_str(r) for r in timing}) == len(rows)


@pytest.mark.parametrize("method", ["spectral_sub", "wiener"])
def test_sweep_metrics_match_direct_evaluation(tiny_ds, tmp_path, method):
    from src import spectral_sub, wiener
    from src.metrics import seg_snr
    from src.mixing import make_mixture

    cfg, denoise = {"spectral_sub": (ex.ss_default(), spectral_sub.denoise),
                    "wiener": (ex.wiener_default(), wiener.denoise)}[method]
    rows = ex.run_sweep("t", [cfg], _args(), tiny_ds, [1], [5], tmp_path)
    r = next(r for r in rows if r["noise"] == "white")
    m = make_mixture(tiny_ds.signals, 1, "white", 5)
    out = denoise(m.noisy, config.FS, m.n_lead)  # module defaults == config defaults
    assert float(r["segsnr_out"]) == seg_snr(m.clean, out[m.n_lead :])
    assert r["method"] == method


def test_longest_run():
    assert ex.longest_run(np.array([0, 1, 1, 0, 1, 1, 1, 0], bool)) == (4, 7)
    assert ex.longest_run(np.array([1, 1, 0], bool)) == (0, 2)
    assert ex.longest_run(np.array([0, 1, 1, 1], bool)) == (1, 4)
    with pytest.raises(ValueError):
        ex.longest_run(np.zeros(5, bool))


def test_speech_free_frames_exclude_lead_in():
    clean = np.concatenate([np.zeros(config.N_LEAD), np.ones(4000), np.zeros(3000)])
    mask = ex.speech_free_frames(clean, config.N_LEAD)
    m = np.arange(mask.size)
    assert not mask[m * config.HOP - config.FRAME_LEN // 2 < config.N_LEAD].any()
    assert mask[-5:].all()


def test_wav_set_common_scale_and_checks(tmp_path):
    rng = np.random.default_rng(0)
    sigs = {"a": rng.standard_normal(1000), "b": 0.1 * rng.standard_normal(1000)}
    scale = ex.write_wav_set(tmp_path, sigs)
    xa, fs = sf.read(tmp_path / "a.wav", dtype="float64")
    xb, _ = sf.read(tmp_path / "b.wav", dtype="float64")
    assert fs == config.FS
    assert max(np.max(np.abs(xa)), np.max(np.abs(xb))) == pytest.approx(config.WAV_PEAK, abs=1e-4)
    np.testing.assert_allclose(xb, scale * sigs["b"], atol=2**-15)  # one common factor
    info = ex.check_wav(tmp_path / "a.wav", 1000)
    assert info["n_clipped"] == 0
    with pytest.raises(AssertionError):
        ex.check_wav(tmp_path / "a.wav", 999)


def test_summarise_means():
    rows = [{"noise": "white", "alpha": "1.0", **{c: str(v) for c in ex.RUN_METRICS}}
            for v in (1.0, 3.0)]
    (rec,) = ex.summarise(rows, ("noise", "alpha"))
    assert rec["n"] == 2 and rec["d_pesq_mean"] == 2.0
    assert rec["d_pesq_std"] == pytest.approx(np.std([1.0, 3.0], ddof=1))


# --- Step 6 transient helpers -------------------------------------------------
def test_click_train_layout():
    x, onsets = ex.click_train()
    assert x.size == int(config.CLICK_TRAIN_S * config.FS)
    assert onsets[0] == int(config.CLICK_FIRST_S * config.FS)
    assert np.all(np.diff(onsets) == int(config.CLICK_PERIOD_S * config.FS))
    n_len = int(config.CLICK_LEN_S * config.FS)
    for o in onsets:  # silent between clicks, identical bursts
        np.testing.assert_array_equal(x[o : o + n_len], x[onsets[0] : onsets[0] + n_len])
        assert np.all(x[o - 10 : o] == 0.0)


def test_transient_measures_on_clean_and_with_pre_echo():
    x, onsets = ex.click_train()
    ref = ex.transient_measures(x, x, onsets)
    assert ref["concentration_mean"] == 1.0
    assert ref["pre_echo_db_mean"] == -np.inf and np.isnan(ref["pre_echo_db_std"])
    assert ref["core_energy_db_mean"] == 0.0
    # Put exactly 1 % of each click's energy into its pre-echo window -> -20 dB.
    y = x.copy()
    pre = int(config.TRANSIENT_PRE_S * config.FS)
    n_len = int(config.CLICK_LEN_S * config.FS)
    e_click = np.sum(x[onsets[0] : onsets[0] + n_len] ** 2)
    for o in onsets:
        y[o - pre : o] = np.sqrt(0.01 * e_click / pre)
    got = ex.transient_measures(y, x, onsets)
    assert got["pre_echo_db_mean"] == pytest.approx(-20.0, abs=1e-9)
    assert got["concentration_mean"] < 1.0


def test_find_plosive_onset_on_synthetic_burst():
    rng = np.random.default_rng(0)
    quiet = np.concatenate([0.3 * rng.standard_normal(800), np.zeros(800)])  # speech, gap
    burst = rng.standard_normal(400)
    sig = np.concatenate([quiet, burst, 0.2 * rng.standard_normal(800)])
    si, onset, jump = ex.find_plosive_onset([np.zeros(10) + 1e-3, sig])
    assert si == 1 and onset == 1600 and jump > 40.0


def test_click_envelope_peaks_at_onset():
    x, onsets = ex.click_train()
    t_ms, env = ex.click_envelope_db(x, onsets, 1.0)
    assert t_ms[0] == pytest.approx(-1000 * config.ENVELOPE_HALF_S)
    assert 0.0 <= t_ms[np.argmax(env)] <= 1000 * config.CLICK_LEN_S


# --- python main.py --input FILE -----------------------------------------------
def _recording(tmp_path, fs_out=16000, seconds=2.0):
    """A 'phone recording': 0.25 s of noise only, then synthetic speech in the same
    noise, written at 16 kHz so the loader has to resample it."""
    import scipy.signal

    from src.data import synthetic_speech

    rng = np.random.default_rng(7)
    speech = synthetic_speech(config.FS, seconds, rng)
    x = np.concatenate([np.zeros(config.N_LEAD), speech])
    x = x + 0.05 * rng.standard_normal(x.size)
    path = tmp_path / "phone_clip.wav"
    sf.write(path, scipy.signal.resample_poly(x, fs_out // config.FS, 1), fs_out,
             subtype="FLOAT")
    return path, x.size


def test_denoise_file_writes_every_method(tmp_path):
    path, n = _recording(tmp_path)
    folder = ex.denoise_file(path, tmp_path / "out")
    assert folder == tmp_path / "out" / "input" / "phone_clip"
    names = sorted(p.stem for p in folder.glob("*.wav"))
    assert names == sorted(["input", "spectral_sub", "wiener", "wavelet", "wavelet_ld"])
    outs = {}
    for name in names:
        y, fs = sf.read(folder / f"{name}.wav", dtype="float64")
        assert fs == config.FS and y.size == n
        outs[name] = y
    # The noise-only start is strongly suppressed by the Fourier methods (one
    # common scale factor, so the files can be compared directly).
    lead = slice(0, config.N_LEAD)
    for name in ("spectral_sub", "wiener"):
        reduction = 10 * np.log10(np.sum(outs["input"][lead] ** 2) / np.sum(outs[name][lead] ** 2))
        assert reduction > 10.0, (name, reduction)


def test_denoise_file_rejects_too_short(tmp_path):
    sf.write(tmp_path / "short.wav", np.zeros(int(0.2 * config.FS)), config.FS)
    with pytest.raises(ValueError, match="need at least"):
        ex.denoise_file(tmp_path / "short.wav", tmp_path / "out")


def test_cli_input_mode(tmp_path, monkeypatch, capsys):
    import main

    path, _ = _recording(tmp_path)
    monkeypatch.setattr(config, "AUDIO_DIR", tmp_path / "audio_out")
    assert main.main(["--input", str(path)]) == 0
    assert (tmp_path / "audio_out" / "input" / "phone_clip" / "wiener.wav").exists()
    assert "written to" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main.main(["--input", str(tmp_path / "missing.wav")])
