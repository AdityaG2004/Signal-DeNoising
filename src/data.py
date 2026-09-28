r"""Clean speech, synthetic fallback and noise sources (objective O3).

Report §3.5 with CLAUDE.md "Data" and correction 1.

NOIZEUS (Hu & Loizou 2007; Loizou 2013, Speech Enhancement, 2nd ed.): 30 IEEE
sentences, 6 speakers (3 male, 3 female), mono, 8 kHz. Files are read as
float64 in [-1, 1) without amplitude normalisation (SNR is relative, so the
absolute level does not matter). Any file not at config.FS is resampled with
scipy.signal.resample_poly by the rational factor FS/fs.

Synthetic speech-like fallback, fully seeded, peak-normalised to 0.5:
  * Voiced segment ("vowel"), 150-300 ms. Source
        e[n] = sum_{k=1}^{K} (1/k) sin(k phi[n]),  phi[n] = 2 pi sum_{i<=n} f0[i] / fs,
    with f0 gliding smoothly (raised-cosine interpolation) between two values
    drawn from [100, 220] Hz, and K = floor(3500 / max f0) so every harmonic
    stays below 3.5 kHz. The source is filtered by 2-3 cascaded two-pole
    resonators (formants F1, F2, F3 from [300, 800], [900, 2500], [2500, 3500] Hz,
    bandwidth B from [80, 150] Hz):
        r = exp(-pi B / fs),   a = [1, -2 r cos(2 pi F / fs), r^2],   y = lfilter([1], a, x),
    then shaped by 20 ms raised-cosine attack and decay ramps.
  * Fricative, 30-80 ms: white noise high-passed above 2 kHz (Butterworth).
  * Plosive: 30 ms closure silence, a 5 ms broadband burst, then a vowel.
  * Pauses of 50-200 ms between syllables, and 0.3 s of leading silence.

Noise:
  * White: d[n] = rng.standard_normal(L)  (Gaussian, unit variance).
  * Babble for target sentence i: n_talkers = 6 other sentences (never i). With
    speaker metadata, talkers exclude i's speaker and mix genders (3 M + 3 F).
    Each talker is scaled to unit RMS and tiled to the required length from a
    random circular offset (so talkers are not time-aligned); the sum is scaled
    to unit RMS. The synthetic fallback feeds its own independently seeded
    synthetic sentences through the same routine.
"""

import csv
import logging
import subprocess
import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio files
# ---------------------------------------------------------------------------
def _to_project_rate(x: np.ndarray, fs: int, name: str) -> np.ndarray:
    """Resample ``x`` from ``fs`` to config.FS with resample_poly (rational factor FS/fs)."""
    if fs != config.FS:
        g = gcd(int(fs), config.FS)
        x = scipy.signal.resample_poly(x, config.FS // g, int(fs) // g)
        log.info("%s: resampled %d Hz -> %d Hz", name, fs, config.FS)
    return np.ascontiguousarray(x, dtype=np.float64)


def load_noizeus(folder: Path = config.CLEAN_DIR) -> list[tuple[str, np.ndarray]]:
    """Sorted list of (name, float64 signal at config.FS) for every WAV in ``folder``."""
    out = []
    for path in sorted(Path(folder).glob("*.wav")):
        x, fs = sf.read(path, dtype="float64", always_2d=True)
        if x.shape[1] != 1:
            raise ValueError(f"{path.name}: expected mono, got {x.shape[1]} channels")
        out.append((path.stem, _to_project_rate(x[:, 0], fs, path.name)))
    return out


def load_audio_file(path: Path) -> np.ndarray:
    """Any recording soundfile can read, as a mono float64 signal at config.FS.

    Multi-channel audio is mixed down to mono (mean of the channels); any
    other sample rate is resampled with resample_poly. Used by
    ``python main.py --input FILE``.
    """
    path = Path(path)
    x, fs = sf.read(path, dtype="float64", always_2d=True)
    if x.shape[0] == 0:
        raise ValueError(f"{path.name} contains no samples")
    if x.shape[1] > 1:
        log.info("%s: mixed %d channels down to mono", path.name, x.shape[1])
    return _to_project_rate(x.mean(axis=1), fs, path.name)


def extract_clean_zip(zip_file, folder: Path = config.CLEAN_DIR) -> list[Path]:
    """Write every .wav member of ``zip_file`` into ``folder`` under its base name.

    Only the base name is used, so member paths cannot write outside ``folder``.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    with zipfile.ZipFile(zip_file) as z:
        for member in z.namelist():
            name = Path(member).name
            if name.lower().endswith(".wav"):
                (folder / name).write_bytes(z.read(member))
                written.append(folder / name)
    return sorted(written)


def download_noizeus(folder: Path = config.CLEAN_DIR, url: str = config.NOIZEUS_URL) -> int:
    """Fetch the NOIZEUS clean sentences into ``folder`` unless all are already there.

    Uses the system ``curl`` (macOS, Linux, Windows 10+) rather than urllib:
    the server sends an incomplete certificate chain that Python's OpenSSL
    rejects, while curl verifies it against the system trust store.
    Certificate checking is never disabled. Verifies the result with
    load_noizeus (mono, 8 kHz after any resampling) and the expected count.
    Returns the number of sentences found.
    """
    folder = Path(folder)
    if len(list(folder.glob("*.wav"))) >= config.N_NOIZEUS_SENTENCES:
        log.info("NOIZEUS already present in %s", folder)
    else:
        log.info("downloading %s", url)
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "clean.zip"
            subprocess.run(["curl", "-fsSL", "--max-time", str(config.DOWNLOAD_TIMEOUT_S),
                            "-o", str(zip_path), url], check=True)
            extract_clean_zip(zip_path, folder)
    n = len(load_noizeus(folder))
    if n != config.N_NOIZEUS_SENTENCES:
        raise RuntimeError(f"expected {config.N_NOIZEUS_SENTENCES} sentences in {folder}, got {n}")
    return n


def load_speakers(path: Path = config.SPEAKERS_CSV) -> dict[str, tuple[str, str]]:
    """Map sentence name (e.g. 'sp01') -> (speaker, gender) from data/speakers.csv."""
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {Path(r["file"]).stem: (r["speaker"], r["gender"]) for r in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# Synthetic speech-like fallback
# ---------------------------------------------------------------------------
def _ramp(n_ramp: int) -> np.ndarray:
    """Raised-cosine rise from 0 towards 1 over n_ramp samples."""
    return 0.5 - 0.5 * np.cos(np.pi * np.arange(n_ramp) / n_ramp)


def _unit_rms(x: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(x**2))
    if rms <= 0:
        raise ValueError("cannot scale an all-zero signal to unit RMS")
    return x / rms


def _vowel(fs: int, rng: np.random.Generator) -> np.ndarray:
    n = int(round(rng.uniform(*config.SYNTH_VOICED_DUR_S) * fs))
    f0_a, f0_b = rng.uniform(*config.SYNTH_F0_RANGE_HZ, size=2)
    glide = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0.0, 1.0, n))
    f0 = f0_a + (f0_b - f0_a) * glide
    phi = 2.0 * np.pi * np.cumsum(f0) / fs + rng.uniform(0.0, 2.0 * np.pi)
    n_harm = int(config.SYNTH_HARMONIC_MAX_HZ // f0.max())
    src = sum(np.sin(k * phi) / k for k in range(1, n_harm + 1))

    n_form = rng.integers(config.SYNTH_N_FORMANTS[0], config.SYNTH_N_FORMANTS[1] + 1)
    for f_lo, f_hi in config.SYNTH_FORMANT_RANGES_HZ[:n_form]:
        f_c = rng.uniform(f_lo, f_hi)
        bw = rng.uniform(*config.SYNTH_FORMANT_BW_HZ)
        r = np.exp(-np.pi * bw / fs)
        a = [1.0, -2.0 * r * np.cos(2.0 * np.pi * f_c / fs), r * r]
        src = scipy.signal.lfilter([1.0], a, src)

    n_ramp = int(round(config.SYNTH_RAMP_S * fs))
    env = np.ones(n)
    env[:n_ramp] = _ramp(n_ramp)
    env[n - n_ramp :] = _ramp(n_ramp)[::-1]
    return _unit_rms(src) * env


def _fricative(fs: int, rng: np.random.Generator) -> np.ndarray:
    n = int(round(rng.uniform(*config.SYNTH_FRIC_DUR_S) * fs))
    sos = scipy.signal.butter(
        config.SYNTH_FRIC_ORDER, config.SYNTH_FRIC_CUTOFF_HZ, "highpass", fs=fs, output="sos"
    )
    return config.SYNTH_FRIC_LEVEL * _unit_rms(scipy.signal.sosfilt(sos, rng.standard_normal(n)))


def _plosive_onset(fs: int, rng: np.random.Generator) -> np.ndarray:
    gap = np.zeros(int(round(config.SYNTH_PLOSIVE_GAP_S * fs)))
    burst = config.SYNTH_BURST_LEVEL * _unit_rms(
        rng.standard_normal(int(round(config.SYNTH_BURST_S * fs)))
    )
    return np.concatenate([gap, burst])


def synthetic_speech(
    fs: int = config.FS,
    duration_s: float = config.SYNTH_DURATION_S,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Seeded speech-like test signal of exactly round(duration_s * fs) samples,
    peak-normalised to config.SYNTH_PEAK. See the module docstring for the model."""
    if rng is None:
        rng = np.random.default_rng(config.SEED)
    n_total = int(round(duration_s * fs))
    parts = [np.zeros(int(round(config.SYNTH_LEAD_SILENCE_S * fs)))]
    used = parts[0].size
    while True:
        level = 10.0 ** (rng.uniform(*config.SYNTH_LEVEL_DB) / 20.0)
        onset = rng.choice(3, p=config.SYNTH_ONSET_PROBS)
        syllable = []
        if onset == 1:
            syllable.append(_fricative(fs, rng))
        elif onset == 2:
            syllable.append(_plosive_onset(fs, rng))
        syllable.append(_vowel(fs, rng))
        syllable.append(np.zeros(int(round(rng.uniform(*config.SYNTH_PAUSE_S) * fs))))
        syl = level * np.concatenate(syllable)
        if used + syl.size > n_total:
            break
        parts.append(syl)
        used += syl.size
    if len(parts) == 1:
        raise ValueError(f"duration {duration_s} s is too short for one syllable")
    x = np.concatenate(parts + [np.zeros(n_total - used)])
    x *= config.SYNTH_PEAK / np.max(np.abs(x))
    assert x.shape == (n_total,) and np.all(np.isfinite(x))
    return x


def synthetic_pool(
    n_sentences: int = config.SYNTH_N_SENTENCES, fs: int = config.FS
) -> list[tuple[str, np.ndarray]]:
    """Independently seeded synthetic 'sentences' syn01, syn02, ..."""
    return [
        (
            f"syn{i + 1:02d}",
            synthetic_speech(fs, config.SYNTH_DURATION_S,
                             np.random.default_rng([config.SEED, config.SYNTH_STREAM, i])),
        )
        for i in range(n_sentences)
    ]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    names: list[str]
    signals: list[np.ndarray]
    speakers: list[tuple[str, str]] | None  # aligned with names; None if unknown
    synthetic: bool


def load_dataset(force_synthetic: bool = False, folder: Path = config.CLEAN_DIR) -> Dataset:
    """NOIZEUS if present (and not forced off), otherwise the synthetic pool."""
    pool = [] if force_synthetic else load_noizeus(folder)
    if not pool:
        if not force_synthetic:
            log.warning("no WAVs in %s: using the synthetic speech fallback", folder)
        pool = synthetic_pool()
        return Dataset([n for n, _ in pool], [s for _, s in pool], None, True)
    names = [n for n, _ in pool]
    spk = load_speakers()
    speakers = [spk[n] for n in names] if spk and all(n in spk for n in names) else None
    if speakers is None:
        log.warning("speaker metadata unavailable: babble talkers chosen without it")
    return Dataset(names, [s for _, s in pool], speakers, False)


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------
def white_noise(length: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance white Gaussian noise."""
    return rng.standard_normal(length)


def choose_talkers(
    n_sentences: int,
    exclude_idx: int,
    rng: np.random.Generator,
    n_talkers: int = config.N_BABBLE_TALKERS,
    speakers: Sequence[tuple[str, str]] | None = None,
) -> np.ndarray:
    """Indices of the babble talkers for target sentence ``exclude_idx``.

    Never includes ``exclude_idx``. With speaker metadata, prefers sentences
    from other speakers with a gender mix of n_talkers//2 male and the rest
    female; falls back to any other speaker, then to any other sentence.
    """
    others = [i for i in range(n_sentences) if i != exclude_idx]
    if len(others) < n_talkers:
        raise ValueError(f"need {n_talkers} other sentences, have {len(others)}")
    if speakers is not None:
        target_speaker = speakers[exclude_idx][0]
        pool = [i for i in others if speakers[i][0] != target_speaker]
        male = [i for i in pool if speakers[i][1] == "M"]
        female = [i for i in pool if speakers[i][1] == "F"]
        n_male = n_talkers // 2
        if len(male) >= n_male and len(female) >= n_talkers - n_male:
            chosen = np.concatenate([
                rng.choice(male, n_male, replace=False),
                rng.choice(female, n_talkers - n_male, replace=False),
            ])
            return np.sort(chosen)
        if len(pool) >= n_talkers:
            return np.sort(rng.choice(pool, n_talkers, replace=False))
    return np.sort(rng.choice(others, n_talkers, replace=False))


def make_babble(
    sentences: Sequence[np.ndarray],
    exclude_idx: int,
    length: int,
    rng: np.random.Generator,
    n_talkers: int = config.N_BABBLE_TALKERS,
    speakers: Sequence[tuple[str, str]] | None = None,
) -> np.ndarray:
    """Unit-RMS babble of ``length`` samples from n_talkers sentences other than
    ``exclude_idx``, each unit-RMS and tiled from a random circular offset."""
    talkers = choose_talkers(len(sentences), exclude_idx, rng, n_talkers, speakers)
    babble = np.zeros(length)
    for t in talkers:
        s = _unit_rms(np.asarray(sentences[t], dtype=np.float64))
        offset = int(rng.integers(s.size))
        reps = -(-(offset + length) // s.size)  # ceil
        babble += np.tile(s, reps)[offset : offset + length]
    return _unit_rms(babble)
