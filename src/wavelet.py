r"""Method 3: wavelet soft thresholding of the whole utterance (O2, O5).

Report §3.4 with CLAUDE.md corrections 3 and 6. Donoho (1995) for soft
thresholding and the universal threshold; Johnstone & Silverman (1997) for
level-dependent thresholds under coloured noise.

This method is *not* Fourier-domain: it works on the time-domain signal as a
single block of length n_samples (not N, which is the STFT frame length).

Decomposition (PyWavelets, orthogonal wavelet, 'periodization' boundary mode):
    x  ->  [cA_L, cD_L, cD_{L-1}, ..., cD_1]      (cD_1 = finest detail, highest band).
At 8 kHz, cD_j covers roughly [fs / 2^(j+1), fs / 2^j]: cD_1 is 2-4 kHz, cD_2 is
1-2 kHz, and so on; cA_L holds everything below fs / 2^(L+1).

Noise level from the median absolute deviation (Donoho 1995):
    sigma_hat = median(|w|) / 0.6745,
which is unbiased for Gaussian w and insensitive to the few large speech
coefficients.

Thresholds:
    variant 'universal':        sigma = MAD(cD_1),  lambda = sigma sqrt(2 ln n_samples),
                                the same lambda for every detail level;
    variant 'level_dependent':  sigma_j = MAD(cD_j), lambda_j = sigma_j sqrt(2 ln n_samples),
                                one threshold per level, needed when the noise
                                is coloured (e.g. babble) so its level differs by scale.

Soft thresholding, applied to detail coefficients only (cA_L is kept):
    T_lambda(w) = sgn(w) max(|w| - lambda, 0).

Reconstruction: x_hat = waverec(...)[:n_samples].

Differences from the Fourier methods. ``n_noise_samples`` is accepted only so
all three denoisers share one signature and is *unused*: the report estimates
sigma blindly with the MAD rule, so this method never sees the noise-only
lead-in that spectral subtraction and the Wiener filter use. Expect the
universal threshold, set from the finest level, to remove weak speech along
with the noise; that is a property of the method, not a bug.
"""

import logging

import numpy as np
import pywt

import config

log = logging.getLogger(__name__)

VARIANTS = config.WAVELET_VARIANTS


def soft_threshold(w: np.ndarray, lam: float) -> np.ndarray:
    """T_lambda(w) = sgn(w) max(|w| - lambda, 0)."""
    return np.sign(w) * np.maximum(np.abs(w) - lam, 0.0)


def mad_sigma(w: np.ndarray) -> float:
    """Robust noise standard deviation median(|w|) / 0.6745."""
    return float(np.median(np.abs(w)) / config.MAD_TO_SIGMA)


def usable_level(n_samples: int, wavelet: str, level: int) -> int:
    """``level`` capped at pywt.dwt_max_level, with a warning when capped."""
    max_level = pywt.dwt_max_level(n_samples, pywt.Wavelet(wavelet).dec_len)
    if level > max_level:
        log.warning("wavelet %s: level %d > max %d for n_samples=%d; using %d",
                    wavelet, level, max_level, n_samples, max_level)
        return max_level
    return level


def analysis(x: np.ndarray, wavelet: str, level: int) -> list[np.ndarray]:
    """[cA_L, cD_L, ..., cD_1] with periodization boundary handling."""
    return pywt.wavedec(x, wavelet, level=level, mode=config.WAVELET_MODE)


def synthesis(coeffs: list[np.ndarray], wavelet: str, n_samples: int) -> np.ndarray:
    """Inverse of :func:`analysis`, trimmed to n_samples."""
    return pywt.waverec(coeffs, wavelet, mode=config.WAVELET_MODE)[:n_samples]


def thresholds(details: list[np.ndarray], n_samples: int, variant: str) -> list[float]:
    """One lambda per detail array (same order as ``details`` = [cD_L, ..., cD_1])."""
    factor = np.sqrt(2.0 * np.log(n_samples))
    if variant == "universal":
        lam = mad_sigma(details[-1]) * factor
        return [lam] * len(details)
    if variant == "level_dependent":
        return [mad_sigma(d) * factor for d in details]
    raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")


def denoise(
    noisy: np.ndarray,
    fs: int = config.FS,
    n_noise_samples: int = config.N_LEAD,
    wavelet: str = config.WAVELET,
    level: int = config.WAVELET_LEVEL,
    variant: str = config.WAVELET_VARIANT,
    return_details: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Soft-threshold the detail coefficients of the whole signal.

    ``fs`` and ``n_noise_samples`` are unused (common denoiser signature; see
    the module docstring). With return_details=True also returns a dict with
    the level used, the thresholds and the coefficients before and after.
    """
    del fs, n_noise_samples  # unused by design
    noisy = np.asarray(noisy, dtype=np.float64)
    n_samples = noisy.size
    level = usable_level(n_samples, wavelet, level)
    coeffs = analysis(noisy, wavelet, level)
    lams = thresholds(coeffs[1:], n_samples, variant)
    shrunk = [coeffs[0]] + [soft_threshold(d, lam) for d, lam in zip(coeffs[1:], lams,
                                                                    strict=True)]
    out = synthesis(shrunk, wavelet, n_samples)

    assert np.all(np.isfinite(out))
    assert out.shape == noisy.shape
    assert out.dtype == np.float64
    if not return_details:
        return out
    return out, {"level": level, "lambdas": lams, "coeffs": coeffs, "shrunk": shrunk}
