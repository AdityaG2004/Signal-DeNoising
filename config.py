"""Project-wide parameters for the EE 317 speech-denoising study.

Every numeric constant used by ``src/`` lives here (CLAUDE.md: "no magic
numbers in modules"). Values follow the "Fixed parameters" table in CLAUDE.md,
which overrides the mid-semester report where they differ.
"""

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"
SPEAKERS_CSV = DATA_DIR / "speakers.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
AUDIO_DIR = ROOT / "audio_out"

# ---------------------------------------------------------------------------
# Sampling (correction 1: the report never states fs; NOIZEUS is 8 kHz)
# ---------------------------------------------------------------------------
FS = 8000  # Hz

# ---------------------------------------------------------------------------
# STFT framework (report §3.1, correction 2)
# ---------------------------------------------------------------------------
FRAME_LEN = 256  # samples (32 ms at 8 kHz); the STFT frame length N
NFFT = 256  # FFT length (= FRAME_LEN, no zero-padding)
HOP = 128  # samples (50 % overlap); the STFT hop H
WINDOW = "hann"  # periodic Hann, i.e. scipy.signal.get_window('hann', N)
COLA_TOL = 1e-12  # max allowed |window-sum envelope - 1| over the kept region
STFT_CHECK_LENGTHS = (12345, 256, 257, 128, 100)  # white-noise round-trip lengths (Step 1)

# ---------------------------------------------------------------------------
# Dataset and mixing (report §3.5)
# ---------------------------------------------------------------------------
LEAD_IN_S = 0.25  # s of noise-only lead-in prepended by the mixer
N_LEAD = int(round(LEAD_IN_S * FS))  # 2000 samples
SNRS_DB = (-5, 0, 5, 10, 15)  # input SNRs (dB)
NOISE_TYPES = ("white", "babble")
SEED = 0
N_BABBLE_TALKERS = 6  # sentences summed to make babble
N_NOIZEUS_SENTENCES = 30
NOIZEUS_URL = "https://ecs.utdallas.edu/loizou/speech/noizeus/clean.zip"
DOWNLOAD_TIMEOUT_S = 120

# ---------------------------------------------------------------------------
# Synthetic speech fallback (used when data/clean/ is empty or --synthetic)
# ---------------------------------------------------------------------------
SYNTH_N_SENTENCES = 30  # same pool size as NOIZEUS, so babble works identically
SYNTH_DURATION_S = 3.0
SYNTH_STREAM = 1_000_000  # rng tag: sentence i uses default_rng([SEED, SYNTH_STREAM, i])
SYNTH_PEAK = 0.5  # peak normalisation
SYNTH_LEAD_SILENCE_S = 0.3
SYNTH_F0_RANGE_HZ = (100.0, 220.0)
SYNTH_HARMONIC_MAX_HZ = 3500.0  # harmonics k*f0 above this are omitted
SYNTH_FORMANT_RANGES_HZ = ((300.0, 800.0), (900.0, 2500.0), (2500.0, 3500.0))
SYNTH_FORMANT_BW_HZ = (80.0, 150.0)
SYNTH_N_FORMANTS = (2, 3)  # inclusive range of resonators per vowel
SYNTH_VOICED_DUR_S = (0.15, 0.30)
SYNTH_RAMP_S = 0.02  # attack/decay of voiced segments
SYNTH_FRIC_DUR_S = (0.03, 0.08)
SYNTH_FRIC_CUTOFF_HZ = 2000.0  # fricative noise is high-passed above this
SYNTH_FRIC_ORDER = 4  # Butterworth order of that high-pass
SYNTH_FRIC_LEVEL = 0.3  # fricative RMS relative to the following vowel
SYNTH_PLOSIVE_GAP_S = 0.03  # closure silence before the burst
SYNTH_BURST_S = 0.005  # broadband burst length
SYNTH_BURST_LEVEL = 1.0  # burst RMS relative to the following vowel
SYNTH_PAUSE_S = (0.05, 0.20)
SYNTH_ONSET_PROBS = (0.4, 0.3, 0.3)  # syllable onset: none, fricative, plosive
SYNTH_LEVEL_DB = (-10.0, 0.0)  # per-syllable level variation

# ---------------------------------------------------------------------------
# Method 1: spectral subtraction (report §3.2; Berouti et al. 1979)
# ---------------------------------------------------------------------------
SS_ALPHA = 2.0  # over-subtraction factor (default)
SS_BETA = 0.02  # spectral floor (default)
SS_ALPHA_GRID = (1.0, 2.0, 3.0, 4.0, 5.0)
SS_BETA_GRID = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1)

# ---------------------------------------------------------------------------
# Method 2: Wiener filter, decision-directed (report §3.3; Ephraim & Malah
# 1984; Scalart & Vieira Filho 1996)
# ---------------------------------------------------------------------------
WIENER_ETA = 0.98  # decision-directed smoothing (default)
WIENER_XI_MIN_DB = -25.0  # a-priori SNR floor (dB)
WIENER_ETA_GRID = (0.5, 0.9, 0.95, 0.98, 0.99)
WIENER_PSD_FLOOR = 1e-20  # floor on the noise PSD estimate lambda_D (avoids 0/0)

# ---------------------------------------------------------------------------
# Method 3: wavelet soft thresholding (report §3.4; Donoho 1995;
# Johnstone & Silverman 1997)
# ---------------------------------------------------------------------------
WAVELET = "db8"  # default family
WAVELET_LEVEL = 5  # default decomposition depth
WAVELET_MODE = "periodization"  # boundary extension for wavedec/waverec
WAVELET_VARIANT = "universal"  # default threshold variant
WAVELET_FAMILIES = ("db4", "db8", "sym8")
WAVELET_LEVELS = (3, 4, 5, 6)
WAVELET_VARIANTS = ("universal", "level_dependent")
MAD_TO_SIGMA = 0.6745  # median(|w|)/0.6745 estimates sigma for Gaussian w

# ---------------------------------------------------------------------------
# Transient experiment (O5, Step 6)
# ---------------------------------------------------------------------------
TRANSIENT_STREAM = 2_000_000  # rng tag for the click train and its noise
CLICK_TRAIN_S = 2.0  # length of the click train
CLICK_PERIOD_S = 0.1  # one click every 100 ms
CLICK_FIRST_S = 0.05  # onset of the first click
CLICK_LEN_S = 0.002  # 2 ms broadband burst
CLICK_DECAY_S = 0.0005  # exponential decay time constant of each burst
CLICK_SNR_DB = 10.0  # white noise level for the click train
CLICK_PLOT_PAIR = 9  # the figure shows clicks 9 and 10 (0-based), mid-train
PLOSIVE_SNR_DB = 5  # white noise level for the real-speech plosive
TRANSIENT_CORE_S = 0.0025  # concentration: energy within +-2.5 ms of the onset ...
TRANSIENT_WIDE_S = 0.025  # ... over the energy within +-25 ms
TRANSIENT_PRE_S = 0.010  # pre-echo: energy in the 10 ms before the onset
ONSET_FRAME_S = 0.0025  # plosive detector: short-term energy frame
ONSET_QUIET_GAP_S = 0.020  # plosive detector: minimum quiet gap before the onset
ENVELOPE_HALF_S = 0.030  # click-synchronous energy envelope spans +-30 ms
ENVELOPE_SMOOTH_S = 0.001  # moving-average length of that envelope

# ---------------------------------------------------------------------------
# Metrics (report §3.6, correction 8)
# ---------------------------------------------------------------------------
SEGSNR_FRAME = 256  # samples, non-overlapping
SEGSNR_GATE_DB = -40.0  # drop frames with clean energy this far below the loudest
SEGSNR_MIN_DB = -10.0  # per-frame clamp (lower)
SEGSNR_MAX_DB = 35.0  # per-frame clamp (upper)
SEGSNR_EPS = 1e-20  # guards log of zero error energy
PESQ_MODE = "nb"  # narrowband: wideband needs 16 kHz
STOI_EXTENDED = False
# PESQ plausibility screen (Step 7): rows whose PESQ rises a lot while STOI falls a lot
# are re-scored with the output delayed by one sample; a valid score barely moves.
PESQ_SUSPECT_D_PESQ = 0.5
PESQ_SUSPECT_D_STOI = -0.1
PESQ_UNSTABLE_JUMP = 0.5  # |PESQ(out) - PESQ(out delayed 1 sample)| above this = unstable

# ---------------------------------------------------------------------------
# Spectrogram display and figures
# ---------------------------------------------------------------------------
SPEC_FLOOR_DB = -100.0  # floor for 20*log10|X|
SPEC_DYN_RANGE_DB = 80.0  # shared colour scale spans [vmax - this, vmax]
FIG_DPI = 300
COLA_FIG_WINDOWS = 8  # shifted windows drawn in figures/cola.png
WAV_PEAK = 0.99  # common-factor peak for WAVs written for listening

# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------
QUICK_N_SENTENCES = 3
QUICK_SNRS_DB = (0, 5, 10)


def output_dirs(
    quick: bool = False, synthetic: bool = False, perceptual: bool = True, create: bool = True
) -> tuple[Path, Path, Path]:
    """(results, figures, audio) folders for a run mode.

    The full NOIZEUS run with PESQ and STOI writes to the top-level folders.
    --synthetic, --quick and --no-perceptual runs write to a sub-folder named
    after the flags (e.g. 'quick', 'synthetic_quick', 'no_perceptual'), so a
    debug run never overwrites the report's outputs.
    """
    flags = (("synthetic", synthetic), ("quick", quick), ("no_perceptual", not perceptual))
    tag = "_".join(name for name, on in flags if on)
    dirs = tuple((root / tag) if tag else root for root in (RESULTS_DIR, FIGURES_DIR, AUDIO_DIR))
    if create:
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    return dirs


def case_rng(sentence_idx: int, noise_idx: int, snr_idx: int) -> np.random.Generator:
    """Return the reproducible random stream for one noisy mixture.

    Every (sentence, noise type, SNR) case gets its own independent stream,
    derived from the global SEED, so cases can run in any order or in parallel
    and still produce bit-identical noise.
    """
    return np.random.default_rng([SEED, sentence_idx, noise_idx, snr_idx])
