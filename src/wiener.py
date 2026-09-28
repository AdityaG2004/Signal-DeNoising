r"""Method 2: STFT-domain Wiener filter with decision-directed a-priori SNR (O2).

Report §3.3 with CLAUDE.md correction 5. Decision-directed estimator from
Ephraim & Malah (1984); its use with the Wiener gain from Scalart & Vieira
Filho (ICASSP 1996).

Analysis (src/stft.py): X(m, k) = STFT{x}(m, k), P(m, k) = |X(m, k)|^2.

Noise PSD, from the frames I lying entirely inside the noise-only lead-in:
    lambda_D(k) = max( (1/|I|) sum_{m in I} |X(m, k)|^2,  eps ).

A-posteriori SNR:
    gamma(m, k) = |X(m, k)|^2 / lambda_D(k).

A-priori SNR, decision-directed, with |S_hat(-1, k)|^2 = 0:
    xi(m, k) = eta |S_hat(m-1, k)|^2 / lambda_D(k) + (1 - eta) max(gamma(m, k) - 1, 0),
    xi(m, k) <- max( xi(m, k), xi_min ),       xi_min = 10^(xi_min_dB / 10).

Wiener gain and clean-magnitude estimate:
    G(m, k) = xi(m, k) / (1 + xi(m, k)),        0 <= G < 1,
    |S_hat(m, k)| = G(m, k) |X(m, k)|.

Synthesis with the noisy phase (report §3.1):
    S_hat(m, k) = |S_hat(m, k)| exp(j angle X(m, k)),   s_hat = iSTFT{S_hat}.

eta trades smoothness against responsiveness. With eta near 1, xi follows the
previous frame's clean estimate, so in noise-only bins xi changes slowly and
stays near xi_min (little musical noise), but xi lags speech onsets and
offsets by a few frames. With eta = 0, xi = max(gamma - 1, 0) is the
instantaneous maximum-likelihood estimate and the gain is the plain
power-subtraction Wiener gain, which fluctuates like spectral subtraction.
"""

import numpy as np

import config
from src.stft import istft, lead_in_frames, stft


def estimate_noise_psd(power: np.ndarray, n_noise_samples: int) -> np.ndarray:
    """lambda_D(k): mean of |X(m, k)|^2 over the lead-in frames, floored at eps."""
    lead = lead_in_frames(n_noise_samples, power.shape[0])
    return np.maximum(np.mean(power[lead], axis=0), config.WIENER_PSD_FLOOR)


def decision_directed_gain(
    power: np.ndarray, noise_psd: np.ndarray, eta: float, xi_min: float
) -> np.ndarray:
    """Run the decision-directed recursion frame by frame; return G(m, k).

    ``power`` is |X|^2 with shape (n_frames, K); ``noise_psd`` is lambda_D(k).
    """
    gain = np.empty_like(power)
    prev_s2 = np.zeros(power.shape[1])  # |S_hat(-1, k)|^2 = 0
    for m in range(power.shape[0]):
        gamma = power[m] / noise_psd
        xi = eta * prev_s2 / noise_psd + (1.0 - eta) * np.maximum(gamma - 1.0, 0.0)
        xi = np.maximum(xi, xi_min)
        g = xi / (1.0 + xi)
        gain[m] = g
        prev_s2 = g * g * power[m]  # |S_hat(m, k)|^2 = (G |X|)^2
    return gain


def denoise(
    noisy: np.ndarray,
    fs: int = config.FS,
    n_noise_samples: int = config.N_LEAD,
    eta: float = config.WIENER_ETA,
    xi_min_db: float = config.WIENER_XI_MIN_DB,
    return_gain: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Decision-directed Wiener filtering of ``noisy`` (lead-in included).

    Output has the input's length. With return_gain=True also returns G(m, k).
    ``xi_min_db = -inf`` disables the a-priori SNR floor.
    """
    if fs != config.FS:
        raise ValueError(f"frame and hop are defined for fs = {config.FS} Hz, got {fs}")
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must be in [0, 1], got {eta}")
    noisy = np.asarray(noisy, dtype=np.float64)
    X = stft(noisy)
    power = np.abs(X) ** 2
    phase = np.angle(X)
    noise_psd = estimate_noise_psd(power, n_noise_samples)
    xi_min = 10.0 ** (xi_min_db / 10.0)  # 0.0 for -inf
    gain = decision_directed_gain(power, noise_psd, eta, xi_min)
    out = istft(gain * np.sqrt(power) * np.exp(1j * phase), length=noisy.size)

    assert np.all(np.isfinite(out))
    assert out.shape == noisy.shape
    assert out.dtype == np.float64
    assert np.all((gain >= 0.0) & (gain <= 1.0))
    return (out, gain) if return_gain else out
