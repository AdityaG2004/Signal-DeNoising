"""Step 5 gate: decision-directed Wiener filter.

Ephraim & Malah (1984); Scalart & Vieira Filho (1996).
"""

import numpy as np
import pytest

import config
from src import wiener
from src.data import load_dataset
from src.metrics import check_alignment
from src.mixing import make_mixture, mix_at_snr
from src.stft import stft

FS = config.FS


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


def test_gain_in_unit_interval(dataset):
    for noise_type in config.NOISE_TYPES:
        for snr in (-5, 5, 15):
            m = make_mixture(dataset.signals, 2, noise_type, snr, dataset.speakers)
            _, gain = wiener.denoise(m.noisy, FS, m.n_lead, return_gain=True)
            assert gain.min() >= 0.0 and gain.max() <= 1.0


def test_hand_computed_two_frames():
    # eta = 0.5, xi_min = 0.01, lambda_D = [1, 1, 2].
    # Frame 0: P = [4, 1, 9] -> gamma = [4, 1, 4.5]; xi = 0.5*[3, 0, 3.5] = [1.5, 0, 1.75],
    #          floored to [1.5, 0.01, 1.75] -> G = [3/5, 1/101, 7/11];
    #          |S|^2 = G^2 P = [1.44, 1/10201, 441/121].
    # Frame 1: P = [16, 2, 1] -> gamma = [16, 2, 0.5];
    #          xi = 0.5*|S|^2/lambda + 0.5*max(gamma-1, 0)
    #             = [0.72 + 7.5, 0.5/10201 + 0.5, (441/121)/4] = [8.22, 1/2 + 1/20402, 441/484]
    #          -> G = [411/461, (1/2 + 1/20402)/(3/2 + 1/20402), 441/925].
    power = np.array([[4.0, 1.0, 9.0], [16.0, 2.0, 1.0]])
    noise_psd = np.array([1.0, 1.0, 2.0])
    expected = np.array([
        [3 / 5, 1 / 101, 7 / 11],
        [411 / 461, (0.5 + 1 / 20402) / (1.5 + 1 / 20402), 441 / 925],
    ])
    got = wiener.decision_directed_gain(power, noise_psd, eta=0.5, xi_min=0.01)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_eta_zero_no_floor_is_ml_power_subtraction_gain(dataset):
    m = make_mixture(dataset.signals, 0, "white", 5, dataset.speakers)
    _, gain = wiener.denoise(m.noisy, FS, m.n_lead, eta=0.0, xi_min_db=-np.inf,
                             return_gain=True)
    power = np.abs(stft(m.noisy)) ** 2
    lam = wiener.estimate_noise_psd(power, m.n_lead)
    ml = np.maximum(power / lam - 1.0, 0.0)
    np.testing.assert_array_equal(gain, ml / (1.0 + ml))


def test_high_snr_sinusoid_passes_through():
    t = np.arange(2 * FS) / FS
    s = np.sin(2 * np.pi * 1000.0 * t)
    noise = np.random.default_rng(config.SEED).standard_normal(s.size + config.N_LEAD)
    noisy, _, _ = mix_at_snr(s, noise, 40.0, config.N_LEAD)
    out = wiener.denoise(noisy, FS, config.N_LEAD)[config.N_LEAD :]
    rel_err = np.linalg.norm(out - s) / np.linalg.norm(s)
    assert rel_err < 0.05, rel_err


def test_noise_only_suppressed_by_10_db():
    d = np.random.default_rng(config.SEED).standard_normal(3 * FS)
    out = wiener.denoise(d, FS, d.size)
    reduction_db = 10 * np.log10(np.mean(d**2) / np.mean(out**2))
    assert reduction_db >= 10.0, reduction_db


def test_noise_psd_uses_only_lead_in_frames():
    x = np.random.default_rng(1).standard_normal(6000)
    power = np.abs(stft(x)) ** 2
    lam = wiener.estimate_noise_psd(power, config.N_LEAD)
    np.testing.assert_allclose(lam, power[1:15].mean(axis=0), rtol=0, atol=1e-15)


def test_output_aligned_with_clean(dataset):
    m = make_mixture(dataset.signals, 0, "white", 5, dataset.speakers)
    out = wiener.denoise(m.noisy, FS, m.n_lead)
    check_alignment(m.clean, out[m.n_lead :])


def test_shape_dtype_odd_length():
    x = np.random.default_rng(3).standard_normal(4001)
    out = wiener.denoise(x, FS, config.N_LEAD)
    assert out.shape == x.shape and out.dtype == np.float64 and np.all(np.isfinite(out))


def test_rejects_bad_arguments():
    x = np.random.default_rng(4).standard_normal(4000)
    with pytest.raises(ValueError, match="no complete"):
        wiener.denoise(x, FS, 200)
    with pytest.raises(ValueError, match="fs"):
        wiener.denoise(x, 16000, config.N_LEAD)
    with pytest.raises(ValueError, match="eta"):
        wiener.denoise(x, FS, config.N_LEAD, eta=1.5)
