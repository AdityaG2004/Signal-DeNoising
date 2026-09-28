"""Step 4 gate: spectral subtraction (Boll 1979; Berouti et al. 1979)."""

import numpy as np
import pytest

import config
from src import spectral_sub
from src.data import load_dataset
from src.metrics import check_alignment
from src.mixing import make_mixture
from src.stft import noise_frame_indices, stft

FS = config.FS
TOL = 1e-12


@pytest.fixture(scope="module")
def mixture():
    ds = load_dataset()
    return make_mixture(ds.signals, 0, "white", 5, ds.speakers)


def test_alpha_zero_returns_input(mixture):
    out = spectral_sub.denoise(mixture.noisy, FS, config.N_LEAD, alpha=0.0, beta=0.02)
    assert np.max(np.abs(out - mixture.noisy)) < TOL


def test_beta_one_returns_input(mixture):
    # max(|X| - alpha D, 1 * |X|) always picks |X|.
    out = spectral_sub.denoise(mixture.noisy, FS, config.N_LEAD, alpha=3.0, beta=1.0)
    assert np.max(np.abs(out - mixture.noisy)) < TOL


def test_noise_only_suppressed_by_10_db():
    d = np.random.default_rng(config.SEED).standard_normal(3 * FS)
    out = spectral_sub.denoise(d, FS, d.size, config.SS_ALPHA, config.SS_BETA)
    reduction_db = 10 * np.log10(np.mean(d**2) / np.mean(out**2))
    assert reduction_db >= 10.0, reduction_db


def test_max_rule_hand_example():
    mag = np.array([[1.0, 0.5, 0.1, 2.0, 0.0]])
    noise = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    # |X| - 2D = [0.6, 0.1, -0.3, 1.6, -0.4];  0.1|X| = [0.1, 0.05, 0.01, 0.2, 0.0]
    expected = np.array([[0.6, 0.1, 0.01, 1.6, 0.0]])
    got = spectral_sub.subtract_magnitude(mag, noise, alpha=2.0, beta=0.1)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-15)


def test_noise_estimate_uses_only_lead_in_frames():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(6000)
    mag = np.abs(stft(x))
    d_hat = spectral_sub.estimate_noise_magnitude(mag, config.N_LEAD)
    np.testing.assert_allclose(d_hat, mag[1:15].mean(axis=0), rtol=0, atol=1e-15)
    # Frame 14, the last one used, ends at sample 14*128 + 128 = 1920; changing
    # anything from there on must not change the estimate.
    last = noise_frame_indices(config.N_LEAD)[-1]
    y = x.copy()
    y[last * config.HOP + config.FRAME_LEN // 2 :] = 100.0
    d_hat_y = spectral_sub.estimate_noise_magnitude(np.abs(stft(y)), config.N_LEAD)
    np.testing.assert_array_equal(d_hat, d_hat_y)


def test_beta_zero_zeros_about_half_the_noise_bins():
    # Musical-noise mechanism: for white noise |X| is ~Rayleigh per bin, so with
    # alpha = 1 the fraction of bins with |X| < mean|X| (set to 0) is
    # P(R < E[R]) = 1 - exp(-pi/4) = 0.544.
    d = np.random.default_rng(2).standard_normal(8 * FS)
    _, gain = spectral_sub.denoise(d, FS, d.size, alpha=1.0, beta=0.0, return_gain=True)
    zero_frac = np.mean(gain[1:-1, 1:-1] == 0.0)  # skip edge frames and DC/Nyquist
    assert zero_frac == pytest.approx(1 - np.exp(-np.pi / 4), abs=0.02)


def test_gain_between_beta_and_one(mixture):
    _, gain = spectral_sub.denoise(mixture.noisy, FS, config.N_LEAD, 2.0, 0.02, return_gain=True)
    mag = np.abs(stft(mixture.noisy))
    g = gain[mag > 0]
    assert g.min() >= 0.02 - 1e-15 and g.max() <= 1.0 + 1e-15


def test_output_aligned_with_clean(mixture):
    out = spectral_sub.denoise(mixture.noisy, FS, config.N_LEAD)
    check_alignment(mixture.clean, out[config.N_LEAD :])


def test_shape_dtype_odd_length():
    x = np.random.default_rng(3).standard_normal(4001)
    out = spectral_sub.denoise(x, FS, config.N_LEAD)
    assert out.shape == x.shape and out.dtype == np.float64 and np.all(np.isfinite(out))


def test_rejects_bad_arguments(mixture):
    with pytest.raises(ValueError, match="no complete"):
        spectral_sub.denoise(mixture.noisy, FS, 200)
    with pytest.raises(ValueError, match="fs"):
        spectral_sub.denoise(mixture.noisy, 16000, config.N_LEAD)
    with pytest.raises(ValueError, match="beta"):
        spectral_sub.denoise(mixture.noisy, FS, config.N_LEAD, beta=1.5)
