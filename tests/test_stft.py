"""Step 1 gate: STFT analysis-synthesis framework (objective O1)."""

import numpy as np
import pytest
import scipy.signal
import soundfile as sf

import config
from src.data import synthetic_speech
from src.stft import (
    bin_freqs,
    frame_times,
    istft,
    n_frames_for,
    noise_frame_indices,
    periodic_hann,
    spectrogram_db,
    stft,
    window_sum,
)

N = config.FRAME_LEN
H = config.HOP
FS = config.FS
PR_TOL = 1e-12  # perfect-reconstruction tolerance (CLAUDE.md Step 1 gate)
NOISE_LENGTHS = (12345, 256, 257, 128, 100)
SPEECH_FILE = config.CLEAN_DIR / "sp01.wav"


def _white(length: int) -> np.ndarray:
    return np.random.default_rng([config.SEED, length]).standard_normal(length)


def _chirp() -> np.ndarray:
    t = np.arange(FS) / FS
    return scipy.signal.chirp(t, f0=50.0, t1=1.0, f1=3900.0, method="linear")


def test_periodic_hann_matches_scipy():
    for n_len in (8, 255, 256, 512):
        np.testing.assert_allclose(
            periodic_hann(n_len), scipy.signal.get_window(config.WINDOW, n_len), rtol=0, atol=1e-15
        )


def test_periodic_hann_is_not_symmetric_hann():
    # np.hanning uses N-1 in the denominator; it must not be what we use.
    assert np.max(np.abs(periodic_hann(N) - np.hanning(N))) > 1e-3


def test_window_sum_is_one_in_interior():
    env = window_sum(8, N, H)
    interior = env[N // 2 : -(N // 2)]
    assert np.max(np.abs(interior - 1.0)) < 1e-15


def test_symmetric_hann_would_fail_cola():
    w_sym = np.hanning(N)
    env = np.zeros(7 * H + N)
    for m in range(8):
        env[m * H : m * H + N] += w_sym
    assert np.max(np.abs(env[N // 2 : -(N // 2)] - 1.0)) > 1e-3


@pytest.mark.parametrize("length", NOISE_LENGTHS)
def test_round_trip_white_noise(length):
    x = _white(length)
    X = stft(x)
    assert X.shape == (n_frames_for(length), N // 2 + 1)
    y = istft(X, N, H, length)
    assert y.shape == x.shape and y.dtype == np.float64
    assert np.max(np.abs(y - x)) < PR_TOL


def test_round_trip_chirp():
    x = _chirp()
    assert np.max(np.abs(istft(stft(x), N, H, x.size) - x)) < PR_TOL


def test_round_trip_synthetic_speech():
    x = synthetic_speech(FS, 3.0, np.random.default_rng(config.SEED))
    assert np.max(np.abs(istft(stft(x), N, H, x.size) - x)) < PR_TOL


@pytest.mark.skipif(not SPEECH_FILE.exists(), reason="NOIZEUS not downloaded")
def test_round_trip_noizeus_speech():
    x, fs = sf.read(SPEECH_FILE, dtype="float64")
    assert fs == FS
    assert np.max(np.abs(istft(stft(x), N, H, x.size) - x)) < PR_TOL


# Only lengths >= N: for shorter inputs scipy.signal.stft silently shrinks
# nperseg to len(x) (it warns "nperseg = 256 is greater than input length"),
# so it is no longer the same transform and cannot serve as a reference.
# Our own round-trip tests above still cover lengths 100 and 128.
@pytest.mark.parametrize("length", [n for n in NOISE_LENGTHS if n >= N] + [FS])
def test_matches_scipy_stft(length):
    x = _white(length)
    _, _, Zxx = scipy.signal.stft(
        x, fs=FS, window="hann", nperseg=N, noverlap=N - H, boundary="zeros", padded=True
    )
    ours = stft(x) / periodic_hann(N).sum()
    assert ours.shape == Zxx.T.shape
    np.testing.assert_allclose(ours, Zxx.T, rtol=0, atol=1e-12)


def test_magnitude_phase_round_trip():
    x = _white(4000)
    X = stft(x)
    gain = np.ones(X.shape)
    Y = gain * np.abs(X) * np.exp(1j * np.angle(X))
    assert np.max(np.abs(istft(Y, N, H, x.size) - x)) < PR_TOL


def test_frame_m_is_centred_on_sample_mH():
    # An impulse at x[mH] sits at the window peak w[N/2] = 1 in frame m,
    # so |X(m, k)| = 1 for every bin k.
    m = 5
    x = np.zeros(3000)
    x[m * H] = 1.0
    np.testing.assert_allclose(np.abs(stft(x)[m]), 1.0, rtol=0, atol=1e-12)


def test_noise_frame_indices_lead_in_2000():
    idx = noise_frame_indices(2000, N, H)
    np.testing.assert_array_equal(idx, np.arange(1, 15))
    assert idx.size == 14
    # Each selected frame lies entirely inside [0, 2000).
    assert np.all(idx * H - N // 2 >= 0) and np.all(idx * H + N // 2 <= 2000)
    # The next frame would cross the lead-in boundary.
    assert (idx[-1] + 1) * H + N // 2 > 2000


def test_noise_frame_indices_too_short():
    assert noise_frame_indices(N - 1, N, H).size == 0
    np.testing.assert_array_equal(noise_frame_indices(N, N, H), [1])


def test_cola_check_fails_loudly_for_non_cola_hop():
    x = _white(3000)
    hop = 64  # Hann at 75 % overlap sums to 2, not 1
    X = stft(x, N, hop)
    with pytest.raises(RuntimeError, match="COLA"):
        istft(X, N, hop, x.size)
    # The envelope division still reconstructs exactly when the check is waived.
    y = istft(X, N, hop, x.size, require_cola=False)
    assert np.max(np.abs(y - x)) < PR_TOL


def test_istft_rejects_inconsistent_length():
    X = stft(_white(1000))
    with pytest.raises(ValueError):
        istft(X, N, H, 5000)


def test_spectrogram_db_floor_and_axes():
    x = np.zeros(1000)
    x[500] = 1.0
    S = spectrogram_db(x)
    assert S.shape == (n_frames_for(1000), N // 2 + 1)
    assert S.min() == pytest.approx(config.SPEC_FLOOR_DB)
    assert np.all(np.isfinite(S))
    assert frame_times(S.shape[0])[1] == pytest.approx(H / FS)
    f = bin_freqs()
    assert f[0] == 0.0 and f[-1] == pytest.approx(FS / 2)
