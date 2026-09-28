"""Step 2 gate: exact-SNR mixing with a noise-only lead-in."""

import numpy as np
import pytest

import config
from src.data import load_dataset
from src.mixing import make_mixture, measured_snr_db, mix_at_snr

SNR_TOL_DB = 0.01  # CLAUDE.md Step 2 gate


@pytest.fixture(scope="module")
def dataset():
    # NOIZEUS when downloaded, otherwise the synthetic fallback: always runs.
    return load_dataset()


@pytest.mark.parametrize("noise_type", config.NOISE_TYPES)
@pytest.mark.parametrize("snr_db", config.SNRS_DB)
def test_measured_snr_matches_target(dataset, noise_type, snr_db):
    for i in (0, 7, 29):
        m = make_mixture(dataset.signals, i, noise_type, snr_db, dataset.speakers)
        err = abs(measured_snr_db(m.clean, m.noise_scaled, m.n_lead) - snr_db)
        assert err < SNR_TOL_DB
        assert err < 1e-9  # in practice it is exact to rounding


@pytest.mark.parametrize("noise_type", config.NOISE_TYPES)
def test_lead_in_is_noise_only(dataset, noise_type):
    m = make_mixture(dataset.signals, 3, noise_type, 5, dataset.speakers)
    n = config.N_LEAD
    assert m.noisy.size == m.clean.size + n
    np.testing.assert_array_equal(m.clean_padded[:n], 0.0)
    np.testing.assert_array_equal(m.noisy[:n], m.noise_scaled[:n])
    assert np.all(m.noisy[:n] != 0.0)
    np.testing.assert_array_equal(m.clean_padded[n:], m.clean)
    np.testing.assert_array_equal(m.noisy, m.clean_padded + m.noise_scaled)


@pytest.mark.parametrize("noise_type", config.NOISE_TYPES)
def test_same_case_is_bit_identical(dataset, noise_type):
    a = make_mixture(dataset.signals, 4, noise_type, 0, dataset.speakers)
    b = make_mixture(dataset.signals, 4, noise_type, 0, dataset.speakers)
    np.testing.assert_array_equal(a.noisy, b.noisy)


@pytest.mark.parametrize("noise_type", config.NOISE_TYPES)
def test_different_indices_give_different_noise(dataset, noise_type):
    base = make_mixture(dataset.signals, 4, noise_type, 0, dataset.speakers)
    other_snr = make_mixture(dataset.signals, 4, noise_type, 5, dataset.speakers)
    other_sentence = make_mixture(dataset.signals, 5, noise_type, 0, dataset.speakers)

    def unit(d):
        return d / np.sqrt(np.mean(d**2))

    assert not np.allclose(unit(base.noise_scaled), unit(other_snr.noise_scaled))
    n = min(base.noise_scaled.size, other_sentence.noise_scaled.size)
    assert not np.allclose(unit(base.noise_scaled)[:n], unit(other_sentence.noise_scaled)[:n])


def test_mix_scale_formula():
    rng = np.random.default_rng(0)
    clean = rng.standard_normal(1000)
    noise = 3.0 * rng.standard_normal(1000 + 200)
    noisy, clean_p, noise_s = mix_at_snr(clean, noise, 10.0, 200)
    c = np.sqrt(np.mean(clean**2) / (np.mean(noise[200:] ** 2) * 10.0))
    np.testing.assert_allclose(noise_s, c * noise, rtol=1e-15)
    assert noisy.dtype == np.float64 and noisy.shape == (1200,)


def test_mix_rejects_bad_inputs():
    with pytest.raises(ValueError, match="length"):
        mix_at_snr(np.ones(100), np.ones(100), 0.0, 10)
    with pytest.raises(ValueError, match="power"):
        mix_at_snr(np.zeros(100), np.ones(110), 0.0, 10)
    with pytest.raises(ValueError, match="unknown noise"):
        make_mixture([np.ones(100)], 0, "pink", 0)
