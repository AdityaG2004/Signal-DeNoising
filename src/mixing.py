r"""Noisy-mixture construction at an exact input SNR (objective O3).

Report §3.5 with CLAUDE.md "Data", correction 8 and BUILD_PROMPT Step 2.

Given clean speech s (length L) and a noise signal d of length L + n_lead:
    P_s = (1/L) sum_n s[n]^2
    P_d = (1/L) sum_{n >= n_lead} d[n]^2      (noise power over the span where speech exists)
    c   = sqrt( P_s / (P_d 10^(SNR_dB/10)) )
    d_s = c d
    s_p = [0]*n_lead ++ s                        (clean, delayed by the noise-only lead-in)
    x   = s_p + d_s
so that 10 log10( P_s / mean(d_s[n_lead:]^2) ) = SNR_dB exactly (to rounding).
The first n_lead samples of x are noise only; the Fourier methods estimate the
noise spectrum from them. Metrics compare s with x_hat[n_lead:] (correction 8).

SNR definition. This is plain power over the whole sentence, including its
short internal pauses. It is *not* the ITU-T P.56 active-speech level that
NOIZEUS used for its own pre-mixed noisy files, which measures speech power
only while speech is active. For the same nominal SNR, P.56 mixing puts more
noise in, so our conditions are somewhat easier than NOIZEUS's published ones
at the same label, and absolute numbers are not directly comparable to
results reported on the NOIZEUS noisy set.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

import config
from src.data import make_babble, white_noise


def mix_at_snr(
    clean: np.ndarray, noise: np.ndarray, snr_db: float, n_lead: int = config.N_LEAD
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (noisy, clean_padded, noise_scaled), each of length len(clean) + n_lead."""
    clean = np.asarray(clean, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    if clean.ndim != 1 or noise.ndim != 1:
        raise ValueError("clean and noise must be 1-D")
    if noise.size != clean.size + n_lead:
        raise ValueError(
            f"noise length {noise.size} != len(clean) + n_lead = {clean.size + n_lead}"
        )
    p_s = np.mean(clean**2)
    p_d = np.mean(noise[n_lead:] ** 2)
    if p_s <= 0 or p_d <= 0:
        raise ValueError("clean and noise must both have non-zero power")
    c = np.sqrt(p_s / (p_d * 10.0 ** (snr_db / 10.0)))
    noise_s = c * noise
    clean_padded = np.concatenate([np.zeros(n_lead), clean])
    noisy = clean_padded + noise_s
    assert np.all(np.isfinite(noisy)) and noisy.shape == noise.shape
    assert noisy.dtype == np.float64
    return noisy, clean_padded, noise_s


def measured_snr_db(
    clean: np.ndarray, noise_scaled: np.ndarray, n_lead: int = config.N_LEAD
) -> float:
    """10 log10( mean(s^2) / mean(d_s[n_lead:]^2) ): the SNR over the sentence span."""
    return float(10.0 * np.log10(np.mean(clean**2) / np.mean(noise_scaled[n_lead:] ** 2)))


def make_noise(
    noise_type: str,
    length: int,
    rng: np.random.Generator,
    sentences: Sequence[np.ndarray] | None = None,
    exclude_idx: int | None = None,
    speakers: Sequence[tuple[str, str]] | None = None,
) -> np.ndarray:
    """Unscaled noise of the given type ('white' or 'babble')."""
    if noise_type == "white":
        return white_noise(length, rng)
    if noise_type == "babble":
        if sentences is None or exclude_idx is None:
            raise ValueError("babble needs the sentence pool and the target index")
        return make_babble(sentences, exclude_idx, length, rng, speakers=speakers)
    raise ValueError(f"unknown noise type {noise_type!r}")


@dataclass
class Mixture:
    clean: np.ndarray  # original sentence, length L
    clean_padded: np.ndarray  # lead-in zeros + clean, length L + n_lead
    noise_scaled: np.ndarray  # length L + n_lead
    noisy: np.ndarray  # length L + n_lead
    n_lead: int
    sentence_idx: int
    noise_type: str
    snr_db: float


def make_mixture(
    sentences: Sequence[np.ndarray],
    sentence_idx: int,
    noise_type: str,
    snr_db: float,
    speakers: Sequence[tuple[str, str]] | None = None,
    n_lead: int = config.N_LEAD,
) -> Mixture:
    """Build one noisy case from its own random stream, config.case_rng.

    The stream is indexed by position in config.NOISE_TYPES and config.SNRS_DB,
    so a case is bit-identical whether it runs in --quick mode or the full sweep.
    """
    if noise_type not in config.NOISE_TYPES:
        raise ValueError(f"unknown noise type {noise_type!r}")
    if snr_db not in config.SNRS_DB:
        raise ValueError(f"SNR {snr_db} dB is not in config.SNRS_DB")
    rng = config.case_rng(
        sentence_idx, config.NOISE_TYPES.index(noise_type), config.SNRS_DB.index(snr_db)
    )
    clean = np.asarray(sentences[sentence_idx], dtype=np.float64)
    noise = make_noise(noise_type, clean.size + n_lead, rng, sentences, sentence_idx, speakers)
    noisy, clean_padded, noise_s = mix_at_snr(clean, noise, snr_db, n_lead)
    return Mixture(clean, clean_padded, noise_s, noisy, n_lead, sentence_idx, noise_type, snr_db)
