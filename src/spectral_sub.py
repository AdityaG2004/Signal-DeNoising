r"""Method 1: magnitude spectral subtraction with over-subtraction and a floor (O2, O5).

Report §3.2 with CLAUDE.md correction 4. Boll (1979) for the method; Berouti,
Schwartz & Makhoul (ICASSP 1979) for the over-subtraction factor alpha and
the spectral floor beta.

Analysis (src/stft.py): X(m, k) = STFT{x}(m, k), noisy phase angle X(m, k).

Noise magnitude estimate, from the frames I that lie entirely inside the
noise-only lead-in (the first n_noise_samples samples; see
stft.noise_frame_indices):
    D_hat(k) = (1/|I|) sum_{m in I} |X(m, k)|.

Subtraction with over-subtraction and floor:
    |S_hat(m, k)| = max( |X(m, k)| - alpha D_hat(k),  beta |X(m, k)| ).
Equivalently, a real gain on the noisy magnitude,
    G(m, k) = max( 1 - alpha D_hat(k) / |X(m, k)|,  beta ),   0 <= G <= 1 for beta <= 1.

Synthesis with the noisy phase (report §3.1):
    S_hat(m, k) = |S_hat(m, k)| exp(j angle X(m, k)),   s_hat = iSTFT{S_hat}.

Musical noise. In noise-only bins |X| fluctuates around D_hat. With beta = 0
every bin with |X| < alpha D_hat is set to zero and the random survivors
remain as isolated, short-lived spectral peaks: tonal "musical" noise. alpha > 1
removes more of those survivors (fewer peaks, at the cost of speech
distortion) and beta > 0 fills the zeroed bins with an attenuated copy of
the noisy spectrum, which masks the remaining peaks. With beta = 0.02 the
floor sits 20 log10(0.02) = -34 dB below the noisy level.
"""

import numpy as np

import config
from src.stft import istft, lead_in_frames, stft


def estimate_noise_magnitude(mag: np.ndarray, n_noise_samples: int) -> np.ndarray:
    """D_hat(k): mean of |X(m, k)| over the frames inside the lead-in."""
    return np.mean(mag[lead_in_frames(n_noise_samples, mag.shape[0])], axis=0)


def subtract_magnitude(
    mag: np.ndarray, noise_mag: np.ndarray, alpha: float, beta: float
) -> np.ndarray:
    """|S_hat| = max(|X| - alpha D_hat, beta |X|), broadcasting D_hat over frames."""
    return np.maximum(mag - alpha * noise_mag, beta * mag)


def denoise(
    noisy: np.ndarray,
    fs: int = config.FS,
    n_noise_samples: int = config.N_LEAD,
    alpha: float = config.SS_ALPHA,
    beta: float = config.SS_BETA,
    return_gain: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Spectral subtraction of ``noisy`` (lead-in included). Output has the input's length.

    With return_gain=True also returns G(m, k) = |S_hat| / |X| (0 where |X| = 0).
    """
    if fs != config.FS:
        raise ValueError(f"frame and hop are defined for fs = {config.FS} Hz, got {fs}")
    if alpha < 0 or not 0 <= beta <= 1:
        raise ValueError(f"need alpha >= 0 and 0 <= beta <= 1, got alpha={alpha}, beta={beta}")
    noisy = np.asarray(noisy, dtype=np.float64)
    X = stft(noisy)
    mag = np.abs(X)
    phase = np.angle(X)
    noise_mag = estimate_noise_magnitude(mag, n_noise_samples)
    s_mag = subtract_magnitude(mag, noise_mag, alpha, beta)
    out = istft(s_mag * np.exp(1j * phase), length=noisy.size)

    assert np.all(np.isfinite(out))
    assert out.shape == noisy.shape
    assert out.dtype == np.float64
    if not return_gain:
        return out
    gain = np.divide(s_mag, mag, out=np.zeros_like(mag), where=mag > 0)
    return out, gain
