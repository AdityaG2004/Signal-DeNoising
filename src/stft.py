r"""Short-time Fourier analysis-synthesis framework (objective O1).

Implements report §3.1 with CLAUDE.md correction 2 (periodic window, edge
padding, analysis-only windowing, plain overlap-add).

Notation: N = frame length (= FFT length), H = hop, L = len(x), K = N/2 + 1.

Window (periodic Hann)
    w[n] = 0.5 - 0.5 cos(2 pi n / N),   n = 0 .. N-1.
  For H = N/2 the shifted copies satisfy the COLA condition
    sum_m w[n - mH] = 1   for every n,
  exactly. The symmetric Hann (np.hanning, denominator N-1) does not; its sum
  ripples by O(1/N), so it would not give perfect reconstruction.

Edge padding
    x_p = [0]*(N/2)  ++  x  ++  [0]*(N/2 + ((-L) mod H)).
  The front N/2 zeros put the first frame's centre on x[0], so every original
  sample lies where two frames overlap and the envelope is 1. The back padding
  does the same for the last sample and makes (len(x_p) - N) a multiple of H.
  Number of frames: M = 1 + (L + ((-L) mod H)) / H.

Analysis (report §3.1, applied to the padded signal)
    X(m, k) = sum_{n=0}^{N-1} x_p[n + mH] w[n] exp(-j 2 pi k n / N),  k = 0 .. N/2.
  In original-signal coordinates, frame m covers samples [mH - N/2, mH + N/2)
  and is centred on sample mH.

Synthesis (overlap-add, no synthesis window)
    y_p[n] = sum_m IDFT_N{X(m, .)}[n - mH]  /  sum_m w[n - mH],
    x_hat  = y_p[N/2 : N/2 + L].
  With no modification, IDFT_N{X(m, .)}[n] = x_p[n + mH] w[n], so the numerator
  is x_p[n] * sum_m w[n - mH] and the division returns x_p[n] exactly. For the
  periodic Hann at H = N/2 the envelope is 1 over the kept region, which is
  asserted (the COLA check), so the division is a no-op; it is kept so other
  hop sizes still reconstruct correctly (with ``require_cola=False``).

References: Allen & Rabiner (1977), "A unified approach to short-time Fourier
analysis and synthesis", Proc. IEEE 65(11); Loizou (2013), Speech Enhancement,
2nd ed., ch. 2 (short-time Fourier analysis and overlap-add synthesis).
"""

import numpy as np

import config

# Envelope values below this cannot be divided out safely (region not covered).
_MIN_ENVELOPE = 1e-8


def periodic_hann(n_len: int) -> np.ndarray:
    """Periodic Hann window w[n] = 0.5 - 0.5 cos(2 pi n / N), n = 0..N-1."""
    n = np.arange(n_len, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / n_len)


def _check_params(n_len: int, hop: int) -> None:
    if n_len % 2 != 0:
        raise ValueError(f"frame length must be even, got {n_len}")
    if not 0 < hop <= n_len:
        raise ValueError(f"hop must satisfy 0 < H <= N, got H={hop}, N={n_len}")


def n_frames_for(length: int, hop: int = config.HOP) -> int:
    """Number of STFT frames for ``length`` samples: 1 + (L + (-L mod H)) / H (independent of N)."""
    return 1 + (length + (-length) % hop) // hop


def window_sum(
    n_frames: int, n_len: int = config.FRAME_LEN, hop: int = config.HOP
) -> np.ndarray:
    """Overlap-added window envelope sum_m w[n - mH] over the padded time axis."""
    w = periodic_hann(n_len)
    env = np.zeros((n_frames - 1) * hop + n_len)
    for m in range(n_frames):
        env[m * hop : m * hop + n_len] += w
    return env


def stft(x: np.ndarray, n_len: int = config.FRAME_LEN, hop: int = config.HOP) -> np.ndarray:
    """STFT with half-frame edge padding and a periodic Hann analysis window.

    Returns a complex array of shape (n_frames, n_len // 2 + 1).
    """
    _check_params(n_len, hop)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("stft expects a non-empty 1-D signal")
    length = x.size
    half = n_len // 2
    x_p = np.concatenate([np.zeros(half), x, np.zeros(half + (-length) % hop)])
    frames = np.lib.stride_tricks.sliding_window_view(x_p, n_len)[::hop].copy()
    assert frames.shape[0] == n_frames_for(length, hop)
    frames *= periodic_hann(n_len)
    return np.fft.rfft(frames, n=n_len, axis=1)


def istft(
    X: np.ndarray,
    n_len: int = config.FRAME_LEN,
    hop: int = config.HOP,
    length: int | None = None,
    require_cola: bool = True,
) -> np.ndarray:
    """Inverse of :func:`stft` by overlap-add, trimmed to ``length`` samples.

    ``require_cola`` asserts the window-sum envelope is 1 (to config.COLA_TOL)
    over the kept region, which holds for the periodic Hann at H = N/2.
    """
    _check_params(n_len, hop)
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != n_len // 2 + 1:
        raise ValueError(f"expected shape (n_frames, {n_len // 2 + 1}), got {X.shape}")
    n_frames = X.shape[0]
    half = n_len // 2
    if length is None:
        length = (n_frames - 1) * hop
    if n_frames_for(length, hop) != n_frames:
        raise ValueError(f"{n_frames} frames is inconsistent with length {length}")

    frames = np.fft.irfft(X, n=n_len, axis=1)
    y = np.zeros((n_frames - 1) * hop + n_len)
    for m in range(n_frames):
        y[m * hop : m * hop + n_len] += frames[m]
    env = window_sum(n_frames, n_len, hop)

    keep = slice(half, half + length)
    env_kept = env[keep]
    if require_cola:
        cola_err = np.max(np.abs(env_kept - 1.0))
        if cola_err > config.COLA_TOL:
            raise RuntimeError(
                f"COLA check failed: max |sum_m w[n-mH] - 1| = {cola_err:.3e} "
                f"> {config.COLA_TOL:g} (N={n_len}, H={hop})"
            )
    if np.min(env_kept) < _MIN_ENVELOPE:
        raise RuntimeError("window-sum envelope is ~0 inside the kept region")
    out = y[keep] / env_kept
    assert out.shape == (length,)
    return out


def noise_frame_indices(
    n_lead: int, n_len: int = config.FRAME_LEN, hop: int = config.HOP
) -> np.ndarray:
    """Frames lying entirely inside the first ``n_lead`` samples of the signal.

    Frame m covers original samples [mH - N/2, mH + N/2), so the condition is
    mH - N/2 >= 0 and mH + N/2 <= n_lead.
    """
    half = n_len // 2
    m_first = -(-half // hop)  # ceil(half / hop)
    m_last = (n_lead - half) // hop
    return np.arange(m_first, m_last + 1) if m_last >= m_first else np.arange(0)


def lead_in_frames(n_noise_samples: int, n_frames: int) -> np.ndarray:
    """noise_frame_indices for the configured N and H, checked to be non-empty
    and to lie inside a signal of ``n_frames`` frames."""
    idx = noise_frame_indices(n_noise_samples)
    if idx.size == 0:
        raise ValueError(
            f"n_noise_samples={n_noise_samples} holds no complete {config.FRAME_LEN}-sample frame"
        )
    if idx[-1] >= n_frames:
        raise ValueError("noise-only region extends past the end of the signal")
    return idx


def frame_times(n_frames: int, fs: int = config.FS, hop: int = config.HOP) -> np.ndarray:
    """Centre time (s) of each frame in original-signal coordinates: t_m = mH / fs."""
    return np.arange(n_frames) * hop / fs


def bin_freqs(fs: int = config.FS, n_len: int = config.NFFT) -> np.ndarray:
    """Frequency (Hz) of each rfft bin: f_k = k fs / N."""
    return np.fft.rfftfreq(n_len, d=1.0 / fs)


def spectrogram_db(
    x: np.ndarray, n_len: int = config.FRAME_LEN, hop: int = config.HOP
) -> np.ndarray:
    """20 log10 |STFT(x)|, floored at config.SPEC_FLOOR_DB. Shape (n_frames, K)."""
    mag = np.abs(stft(x, n_len, hop))
    floor = 10.0 ** (config.SPEC_FLOOR_DB / 20.0)
    return 20.0 * np.log10(np.maximum(mag, floor))
