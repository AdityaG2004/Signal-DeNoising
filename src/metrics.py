r"""Evaluation metrics (objective O4).

Report §3.6 with CLAUDE.md "Fixed parameters" and correction 8. Every metric
compares the original sentence s with the estimate over the same span: the
caller passes ``clean`` and ``est[n_lead:]`` (the lead-in is trimmed first).

Segmental SNR (report §3.6; conventions from Loizou 2013, Speech Enhancement,
2nd ed., ch. 11, objective quality measures). With non-overlapping frames F_m
of 256 samples (a trailing
partial frame is dropped):
    E_s(m) = sum_{n in F_m} s[n]^2,     E_e(m) = sum_{n in F_m} (s[n] - s_hat[n])^2
    keep frame m  iff  E_s(m) >= max_m' E_s(m') * 10^(-40/10)
    SNR_m  = clamp( 10 log10( E_s(m) / max(E_e(m), eps) ), -10, 35 ),  eps = 1e-20
    segSNR = mean of SNR_m over the kept frames.
  The log is inside the average, so quiet frames weigh as much as loud ones;
  for speech this puts segSNR below the global SNR. The 40 dB gate drops
  silent frames (which would otherwise sit at the -10 dB clamp) and the upper
  clamp stops near-perfect frames from dominating.

Global SNR (for reference):
    SNR = 10 log10( sum s^2 / sum (s - s_hat)^2 ).

PESQ (ITU-T P.862), narrowband because fs = 8 kHz (wideband P.862.2 needs
16 kHz): pesq(8000, ref, deg, 'nb'). The `pesq` package returns the P.862.1
MOS-LQO mapping of the raw score, so values lie in about [1.02, 4.55]
(clean vs clean gives 4.549). A PESQ failure on one file is the single
exception this project catches: it is logged and returned as NaN.

STOI (Taal, Hendriks, Heusdens & Jensen 2011): pystoi.stoi(ref, deg, 8000,
extended=False). pystoi resamples to 10 kHz and removes silent frames itself.
Range [0, 1] in practice (correlation-based).
"""

import logging
import math

import numpy as np
from pesq import pesq
from pystoi import stoi

import config

log = logging.getLogger(__name__)

PESQ_NB_RANGE = (1.0, 4.56)  # P.862.1 MOS-LQO bounds, with rounding slack


def _check_pair(clean: np.ndarray, est: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clean = np.asarray(clean, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)
    if clean.ndim != 1 or clean.shape != est.shape:
        raise ValueError(f"clean {clean.shape} and est {est.shape} must be equal-length 1-D")
    assert np.all(np.isfinite(clean)) and np.all(np.isfinite(est))
    return clean, est


def seg_snr(clean: np.ndarray, est: np.ndarray, frame: int = config.SEGSNR_FRAME) -> float:
    """Segmental SNR in dB (see module docstring for the exact definition)."""
    out = float(np.mean(seg_snr_frames(clean, est, frame)))
    assert math.isfinite(out)
    return out


def seg_snr_frames(
    clean: np.ndarray, est: np.ndarray, frame: int = config.SEGSNR_FRAME
) -> np.ndarray:
    """Clamped per-frame SNRs (dB) of the frames that pass the energy gate.

    seg_snr is their mean; the fraction sitting at SEGSNR_MIN_DB shows how
    much the lower clamp compresses the metric at low input SNR.
    """
    clean, est = _check_pair(clean, est)
    n_frames = clean.size // frame
    if n_frames == 0:
        raise ValueError(f"signal shorter than one {frame}-sample frame")
    s = clean[: n_frames * frame].reshape(n_frames, frame)
    e = s - est[: n_frames * frame].reshape(n_frames, frame)
    e_s = np.sum(s**2, axis=1)
    e_e = np.sum(e**2, axis=1)
    if np.max(e_s) <= 0:
        raise ValueError("clean signal is all zeros")
    keep = e_s >= np.max(e_s) * 10.0 ** (config.SEGSNR_GATE_DB / 10.0)  # kept frames: E_s > 0
    snr = 10.0 * np.log10(e_s[keep] / np.maximum(e_e[keep], config.SEGSNR_EPS))
    return np.clip(snr, config.SEGSNR_MIN_DB, config.SEGSNR_MAX_DB)


def global_snr(clean: np.ndarray, est: np.ndarray) -> float:
    """10 log10( sum s^2 / sum (s - s_hat)^2 ) in dB (inf if est == clean)."""
    clean, est = _check_pair(clean, est)
    err = np.sum((clean - est) ** 2)
    return math.inf if err == 0 else float(10.0 * np.log10(np.sum(clean**2) / err))


def pesq_nb(clean: np.ndarray, est: np.ndarray, fs: int = config.FS, tag: str = "") -> float:
    """Narrowband PESQ (MOS-LQO). Returns NaN, with a logged warning, on failure."""
    clean, est = _check_pair(clean, est)  # equal lengths, finite
    try:
        score = float(pesq(fs, clean, est, config.PESQ_MODE))
    # Deliberately broad: the `pesq` C code raises builtins such as ValueError
    # (e.g. on an all-zero input), not only PesqError. This is the one
    # catch-and-continue the project allows (BUILD_PROMPT rule 6).
    except Exception as exc:
        log.warning("PESQ failed%s: %s: %s", f" for {tag}" if tag else "",
                    type(exc).__name__, exc)
        return math.nan
    assert PESQ_NB_RANGE[0] <= score <= PESQ_NB_RANGE[1], score
    return score


def stoi_score(clean: np.ndarray, est: np.ndarray, fs: int = config.FS) -> float:
    """Short-time objective intelligibility (Taal et al. 2011)."""
    clean, est = _check_pair(clean, est)
    score = float(stoi(clean, est, fs, extended=config.STOI_EXTENDED))
    assert math.isfinite(score)
    return score


def evaluate(
    clean: np.ndarray, est: np.ndarray, fs: int = config.FS, perceptual: bool = True,
    tag: str = "",
) -> dict[str, float]:
    """segSNR, global SNR, PESQ and STOI of ``est`` against ``clean``.

    With perceptual=False, PESQ and STOI are NaN (fast debugging).
    """
    return {
        "segsnr": seg_snr(clean, est),
        "global_snr": global_snr(clean, est),
        "pesq": pesq_nb(clean, est, fs, tag) if perceptual else math.nan,
        "stoi": stoi_score(clean, est, fs) if perceptual else math.nan,
    }


def find_lag(clean: np.ndarray, est: np.ndarray) -> int:
    """Lag (samples) maximising the cross-correlation sum_n s[n] s_hat[n + lag].

    Positive means ``est`` is delayed relative to ``clean``.
    """
    clean, est = _check_pair(clean, est)
    n = 2 * clean.size
    xcorr = np.fft.irfft(np.fft.rfft(est, n) * np.conj(np.fft.rfft(clean, n)), n)
    lag = int(np.argmax(xcorr))
    return lag - n if lag >= clean.size else lag


def check_alignment(clean: np.ndarray, est: np.ndarray) -> None:
    """Assert ``est`` is time-aligned with ``clean`` (peak cross-correlation at lag 0).

    Catches off-by-N/2 framing bugs in the denoisers.
    """
    lag = find_lag(clean, est)
    assert lag == 0, f"estimate is misaligned by {lag} samples"
