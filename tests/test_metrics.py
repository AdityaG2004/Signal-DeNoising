"""Step 3 gate: segSNR, global SNR, PESQ-NB, STOI and the alignment check."""

import logging
import math

import numpy as np
import pytest

import config
from src.data import load_dataset
from src.metrics import (
    check_alignment,
    evaluate,
    find_lag,
    global_snr,
    pesq_nb,
    seg_snr,
    seg_snr_frames,
    stoi_score,
)
from src.mixing import make_mixture, mix_at_snr

FS = config.FS
F = config.SEGSNR_FRAME


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()  # NOIZEUS if present, else the synthetic fallback


def _noisy_inputs(dataset, noise_type, sentence=0):
    """(clean, noisy-with-lead-in-trimmed) for each input SNR, ascending."""
    out = []
    for snr in config.SNRS_DB:
        m = make_mixture(dataset.signals, sentence, noise_type, snr, dataset.speakers)
        out.append((snr, m.clean, m.noisy[m.n_lead :]))
    return out


# --- segSNR ---------------------------------------------------------------
def test_segsnr_identity_is_upper_clamp(dataset):
    s = dataset.signals[0]
    assert seg_snr(s, s) == 35.0


def test_segsnr_stationary_sine_matches_global_snr():
    # A constant-amplitude sine has equal energy in every frame, so segSNR
    # should match the global SNR of the additive noise.
    t = np.arange(2 * FS) / FS
    s = np.sin(2 * np.pi * 440.0 * t)
    rng = np.random.default_rng(config.SEED)
    noise = rng.standard_normal(s.size + config.N_LEAD)
    noisy, _, _ = mix_at_snr(s, noise, 10.0, config.N_LEAD)
    est = noisy[config.N_LEAD :]
    assert global_snr(s, est) == pytest.approx(10.0, abs=1e-9)
    assert seg_snr(s, est) == pytest.approx(10.0, abs=0.5)


@pytest.mark.parametrize("noise_type", config.NOISE_TYPES)
def test_segsnr_of_noisy_input_rises_with_input_snr(dataset, noise_type):
    # For speech, the noisy-input segSNR sits below the global input SNR:
    # quiet speech frames get low (or clamped -10 dB) per-frame SNRs and count
    # as much as loud ones. That offset is expected, not a mixing bug.
    vals = [seg_snr(clean, noisy) for _, clean, noisy in _noisy_inputs(dataset, noise_type)]
    assert all(b > a for a, b in zip(vals, vals[1:], strict=False)), vals


def test_segsnr_excludes_frames_below_gate():
    # Two loud frames at exactly 20 dB SNR and two frames 60 dB quieter whose
    # per-frame SNR would be -10 dB (clamped). The quiet frames are below the
    # -40 dB gate, so segSNR must be 20, not (20+20-10-10)/4 = 5.
    rng = np.random.default_rng(1)
    loud = rng.standard_normal(2 * F)
    quiet = 1e-3 * rng.standard_normal(2 * F)
    s = np.concatenate([loud, quiet])
    err = np.concatenate([rng.standard_normal(2 * F), 10.0 * quiet])
    for m in range(2):  # scale each loud frame's error to exactly 20 dB
        sl = slice(m * F, (m + 1) * F)
        err[sl] *= np.sqrt(np.sum(s[sl] ** 2) / (100.0 * np.sum(err[sl] ** 2)))
    assert seg_snr(s, s + err) == pytest.approx(20.0, abs=1e-9)
    # The quiet frames really are below the gate (-60 dB) and would score -20 dB,
    # i.e. -10 after clamping, if they were kept.
    e_quiet = np.sum(quiet[:F] ** 2) / np.sum(loud[:F] ** 2)
    assert 10 * np.log10(e_quiet) < config.SEGSNR_GATE_DB


def test_segsnr_frame_just_above_gate_is_kept():
    rng = np.random.default_rng(2)
    loud = rng.standard_normal(F)
    loud *= 1.0 / np.sqrt(np.sum(loud**2))  # E_s = 1
    quiet = rng.standard_normal(F)
    quiet *= np.sqrt(1.01e-4 / np.sum(quiet**2))  # E_s just above 1e-4 (the -40 dB gate)
    s = np.concatenate([loud, quiet])
    est = np.concatenate([loud, np.zeros(F)])  # quiet frame: SNR 0 dB, loud: clamp 35
    assert seg_snr(s, est) == pytest.approx((35.0 + 0.0) / 2, abs=1e-9)


def test_segsnr_drops_trailing_partial_frame():
    rng = np.random.default_rng(3)
    s = rng.standard_normal(3 * F + 100)
    est = s.copy()
    est[3 * F :] = 0.0  # error only in the partial frame
    assert seg_snr(s, est) == 35.0


def test_segsnr_lower_clamp():
    s = np.random.default_rng(4).standard_normal(4 * F)
    assert seg_snr(s, -10.0 * s) == -10.0


def test_segsnr_rejects_bad_input():
    with pytest.raises(ValueError):
        seg_snr(np.ones(1000), np.ones(999))
    with pytest.raises(ValueError):
        seg_snr(np.zeros(1000), np.zeros(1000))


# --- PESQ -----------------------------------------------------------------
def test_pesq_clean_near_top(dataset):
    s = dataset.signals[0]
    assert pesq_nb(s, s) >= 4.0


def test_pesq_noisy_much_lower(dataset):
    s = dataset.signals[0]
    (_, clean, noisy), = [c for c in _noisy_inputs(dataset, "white") if c[0] == 0]
    assert pesq_nb(clean, noisy) < pesq_nb(s, s) - 1.5


def test_pesq_all_zero_returns_nan_and_logs(dataset, caplog):
    s = dataset.signals[0]
    with caplog.at_level(logging.WARNING):
        score = pesq_nb(s, np.zeros_like(s), tag="all-zero test")
    assert math.isnan(score)
    assert "PESQ failed for all-zero test" in caplog.text


def test_pesq_rejects_length_mismatch(dataset):
    s = dataset.signals[0]
    with pytest.raises(ValueError):
        pesq_nb(s, s[:-1])


# --- STOI -----------------------------------------------------------------
def test_stoi_clean_is_one(dataset):
    s = dataset.signals[0]
    assert stoi_score(s, s) >= 0.99


@pytest.mark.parametrize("noise_type", config.NOISE_TYPES)
def test_stoi_falls_with_snr(dataset, noise_type):
    vals = [stoi_score(clean, noisy) for _, clean, noisy in _noisy_inputs(dataset, noise_type)]
    assert all(b > a for a, b in zip(vals, vals[1:], strict=False)), vals


# --- evaluate / alignment -------------------------------------------------
def test_evaluate_keys_and_fast_mode(dataset):
    s = dataset.signals[1]
    full = evaluate(s, s)
    assert set(full) == {"segsnr", "global_snr", "pesq", "stoi"}
    assert full["segsnr"] == 35.0 and full["global_snr"] == math.inf
    fast = evaluate(s, s, perceptual=False)
    assert math.isnan(fast["pesq"]) and math.isnan(fast["stoi"])


def test_alignment_zero_for_aligned(dataset):
    s = dataset.signals[0]
    check_alignment(s, s)
    (_, clean, noisy), = [c for c in _noisy_inputs(dataset, "white") if c[0] == 5]
    check_alignment(clean, noisy)


@pytest.mark.parametrize("shift", [1, 128, -128, 256])
def test_alignment_detects_shift(dataset, shift):
    s = dataset.signals[0]
    shifted = np.roll(s, shift)
    assert find_lag(s, shifted) == shift
    with pytest.raises(AssertionError, match="misaligned"):
        check_alignment(s, shifted)


def test_segsnr_is_mean_of_frames(dataset):
    (_, clean, noisy), = [c for c in _noisy_inputs(dataset, "white") if c[0] == -5]
    frames = seg_snr_frames(clean, noisy)
    assert seg_snr(clean, noisy) == float(np.mean(frames))
    assert frames.min() >= config.SEGSNR_MIN_DB and frames.max() <= config.SEGSNR_MAX_DB
    assert np.mean(frames == config.SEGSNR_MIN_DB) > 0.2  # many clamped frames at -5 dB
