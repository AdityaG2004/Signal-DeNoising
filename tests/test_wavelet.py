"""Step 6 gate: wavelet soft thresholding (Donoho 1995; Johnstone & Silverman 1997)."""

import logging

import numpy as np
import pytest

import config
from src import wavelet
from src.data import load_dataset, make_babble
from src.metrics import check_alignment
from src.mixing import make_mixture

FS = config.FS


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


@pytest.mark.parametrize("family", config.WAVELET_FAMILIES)
@pytest.mark.parametrize("length", [24000, 24001, 22529, 4097])
def test_analysis_synthesis_perfect_reconstruction(family, length):
    x = np.random.default_rng([config.SEED, length]).standard_normal(length)
    for level in config.WAVELET_LEVELS:
        coeffs = wavelet.analysis(x, family, level)
        assert len(coeffs) == level + 1
        y = wavelet.synthesis(coeffs, family, length)
        assert y.shape == x.shape
        assert np.max(np.abs(y - x)) < 1e-10


def test_soft_threshold_hand_example():
    w = np.array([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.25, 3.0])
    expected = np.array([-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 2.0])  # lambda = 1
    got = wavelet.soft_threshold(w, 1.0)
    np.testing.assert_array_equal(got, expected)  # exactly +-lambda maps to 0


def test_mad_sigma_on_white_noise():
    d = 0.1 * np.random.default_rng(config.SEED).standard_normal(24000)
    cd1 = wavelet.analysis(d, config.WAVELET, config.WAVELET_LEVEL)[-1]
    assert wavelet.mad_sigma(cd1) == pytest.approx(0.1, rel=0.05)


def test_level_dependent_sigmas_flat_for_white_noise():
    # An orthonormal DWT maps white noise to white coefficients of the same
    # sigma at every level. The MAD estimate from n_j coefficients has a
    # relative std of about 1.17/sqrt(n_j) (4.3 % for the 750 at level 5),
    # so 15 % is a ~3.5-sigma bound.
    d = 0.1 * np.random.default_rng(1).standard_normal(24000)
    coeffs = wavelet.analysis(d, config.WAVELET, config.WAVELET_LEVEL)
    sigmas = [wavelet.mad_sigma(c) for c in coeffs[1:]]
    for s in sigmas:
        assert s == pytest.approx(0.1, rel=0.15)


def test_level_dependent_sigmas_differ_for_babble(dataset):
    b = make_babble(dataset.signals, 0, 24000, np.random.default_rng(2))
    coeffs = wavelet.analysis(b, config.WAVELET, config.WAVELET_LEVEL)
    sigmas = np.array([wavelet.mad_sigma(c) for c in coeffs[1:]])
    # Speech-shaped noise: most energy below 1 kHz, little in 2-4 kHz (cD_1).
    assert sigmas.max() / sigmas.min() > 3.0, sigmas
    lams = wavelet.thresholds(coeffs[1:], b.size, "level_dependent")
    assert len(set(lams)) == len(lams)


def test_universal_uses_one_lambda_from_finest_level():
    d = np.random.default_rng(3).standard_normal(8000)
    coeffs = wavelet.analysis(d, "db4", 4)
    lams = wavelet.thresholds(coeffs[1:], d.size, "universal")
    expected = wavelet.mad_sigma(coeffs[-1]) * np.sqrt(2 * np.log(d.size))
    assert lams == [expected] * 4


def test_approximation_untouched_details_shrunk(dataset):
    m = make_mixture(dataset.signals, 0, "white", 5, dataset.speakers)
    _, info = wavelet.denoise(m.noisy, FS, m.n_lead, return_details=True)
    np.testing.assert_array_equal(info["shrunk"][0], info["coeffs"][0])
    for before, after, lam in zip(info["coeffs"][1:], info["shrunk"][1:], info["lambdas"],
                                  strict=True):
        np.testing.assert_array_equal(after, wavelet.soft_threshold(before, lam))
        assert np.all(np.abs(after) <= np.abs(before))


@pytest.mark.parametrize("variant", config.WAVELET_VARIANTS)
def test_heavisine_textbook_denoising(variant):
    # Donoho's HeaviSine-type test signal (smooth plus two jumps) in unit white
    # noise: a correct soft-threshold denoiser raises the SNR by well over 10 dB.
    n = 16384
    t = np.arange(n) / n
    f = 4 * np.sin(4 * np.pi * t) - np.sign(t - 0.3) - np.sign(0.72 - t)
    noisy = f + np.random.default_rng(config.SEED).standard_normal(n)
    out = wavelet.denoise(noisy, FS, 0, "db8", 5, variant)

    def snr(est):
        return 10 * np.log10(np.sum(f**2) / np.sum((f - est) ** 2))

    assert snr(out) - snr(noisy) > 10.0


def test_n_noise_samples_is_unused(dataset):
    m = make_mixture(dataset.signals, 1, "white", 5, dataset.speakers)
    a = wavelet.denoise(m.noisy, FS, m.n_lead)
    b = wavelet.denoise(m.noisy, FS, 123)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("variant", config.WAVELET_VARIANTS)
def test_output_aligned_with_clean(dataset, variant):
    m = make_mixture(dataset.signals, 0, "white", 5, dataset.speakers)
    out = wavelet.denoise(m.noisy, FS, m.n_lead, variant=variant)
    check_alignment(m.clean, out[m.n_lead :])


@pytest.mark.parametrize("variant", config.WAVELET_VARIANTS)
def test_both_variants_reduce_white_noise(variant):
    d = np.random.default_rng(4).standard_normal(3 * FS)
    out = wavelet.denoise(d, FS, d.size, variant=variant)
    assert 10 * np.log10(np.mean(d**2) / np.mean(out**2)) > 10.0


def test_level_capped_with_warning(caplog):
    x = np.random.default_rng(5).standard_normal(100)
    with caplog.at_level(logging.WARNING):
        out, info = wavelet.denoise(x, FS, 0, "db8", 6, return_details=True)
    assert info["level"] < 6 and "level 6 > max" in caplog.text
    assert out.shape == x.shape


def test_shape_dtype_odd_length_and_bad_variant():
    x = np.random.default_rng(6).standard_normal(4001)
    out = wavelet.denoise(x, FS, config.N_LEAD, variant="level_dependent")
    assert out.shape == x.shape and out.dtype == np.float64 and np.all(np.isfinite(out))
    with pytest.raises(ValueError, match="variant"):
        wavelet.denoise(x, FS, config.N_LEAD, variant="minimax")
