"""Figures for the final report (all saved at config.FIG_DPI).

Style rules (BUILD_PROMPT Step 7): one colour per method across every figure,
taken from a colour-blind-safe palette defined once here; axis labels carry
units; compared spectrograms share one dB colour scale.

Palette: slots 1-3 and 7 of the dataviz reference categorical palette (blue,
orange, aqua, violet). Validated all-pairs on a white surface with the
Machado et al. (2009) CVD simulation: worst CVD Delta E 9.2 (>= 8 target),
worst normal-vision Delta E 16.3 (>= 15 floor). Aqua is 2.8:1 against white,
so every chart also carries a legend and a distinct marker per method.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402

import config  # noqa: E402
from src.stft import bin_freqs, frame_times, periodic_hann, spectrogram_db, window_sum  # noqa: E402

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"

METHOD_COLORS = {
    "clean": INK,
    "noisy": INK_SECONDARY,
    "spectral_sub": "#2a78d6",
    "wiener": "#eb6834",
    "wavelet": "#1baf7a",
    "wavelet_ld": "#4a3aa7",
}
METHOD_MARKERS = {
    "noisy": "x",
    "spectral_sub": "o",
    "wiener": "s",
    "wavelet": "^",
    "wavelet_ld": "D",
}
METHOD_LABELS = {
    "clean": "Clean",
    "noisy": "Noisy input",
    "spectral_sub": "Spectral subtraction",
    "wiener": "Wiener (decision-directed)",
    "wavelet": "Wavelet (universal)",
    "wavelet_ld": "Wavelet (level-dependent)",
}
# Perceptually uniform, monotone in lightness, legible in greyscale print.
SPEC_CMAP = "magma"


def method_key(method: str, variant: str = "") -> str:
    """Key into METHOD_COLORS / MARKERS / LABELS for a (method, variant) pair."""
    return "wavelet_ld" if method == "wavelet" and variant == "level_dependent" else method


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": config.FIG_DPI,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": INK_MUTED,
            "axes.labelcolor": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "legend.frameon": False,
            "lines.linewidth": 1.5,
        }
    )


def save_fig(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Spectrograms on a shared dB scale
# ---------------------------------------------------------------------------
def shared_db_limits(
    specs_db: Sequence[np.ndarray], dyn_range_db: float = config.SPEC_DYN_RANGE_DB
) -> tuple[float, float]:
    """(vmin, vmax) common to all panels: vmax is the loudest bin over every
    panel (rounded up to 5 dB) and vmin sits dyn_range_db below it."""
    vmax = float(np.ceil(max(float(np.max(s)) for s in specs_db) / 5.0) * 5.0)
    return vmax - dyn_range_db, vmax


def draw_spectrogram(
    ax: plt.Axes,
    spec_db: np.ndarray,
    vmin: float,
    vmax: float,
    fs: int = config.FS,
    t0: float = 0.0,
):
    """Draw one (n_frames, K) dB spectrogram; frame m is centred at t0 + mH/fs."""
    t = t0 + frame_times(spec_db.shape[0], fs)
    f = bin_freqs(fs)
    mesh = ax.pcolormesh(
        t, f, spec_db.T, cmap=SPEC_CMAP, vmin=vmin, vmax=vmax, shading="nearest", rasterized=True
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.grid(False)
    return mesh


def plot_spectrograms(
    signals: Sequence[np.ndarray],
    titles: Sequence[str],
    path: Path,
    suptitle: str,
    fs: int = config.FS,
    ncols: int | None = None,
    t0: float = 0.0,
) -> tuple[float, float]:
    """One spectrogram panel per signal, all on one shared dB colour scale.

    Returns the (vmin, vmax) used so callers can reuse it for insets.
    """
    if len(signals) != len(titles):
        raise ValueError("signals and titles must have the same length")
    specs = [spectrogram_db(s) for s in signals]
    vmin, vmax = shared_db_limits(specs)
    ncols = ncols or len(signals)
    nrows = int(np.ceil(len(signals) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), squeeze=False, sharex=True,
        sharey=True, layout="constrained",
    )
    mesh = None
    for ax, spec, title in zip(axes.flat, specs, titles, strict=False):
        mesh = draw_spectrogram(ax, spec, vmin, vmax, fs, t0)
        ax.set_title(title)
    for ax in list(axes.flat)[len(signals) :]:
        ax.set_visible(False)
    cbar = fig.colorbar(mesh, ax=axes, shrink=0.9)
    cbar.set_label("Magnitude (dB)")
    fig.suptitle(suptitle)
    save_fig(fig, path)
    return vmin, vmax


# ---------------------------------------------------------------------------
# Step 2: mixing example
# ---------------------------------------------------------------------------
def plot_mixing_example(
    panels: Sequence[tuple[str, np.ndarray]],
    n_lead: int,
    path: Path,
    suptitle: str,
    fs: int = config.FS,
) -> tuple[float, float]:
    """Waveform (top row) and spectrogram (bottom row) per signal.

    Every signal includes the noise-only lead-in; time is measured from the
    start of the sentence, so the lead-in occupies [-n_lead/fs, 0). Waveforms
    share one amplitude axis and spectrograms share one dB colour scale.
    """
    apply_style()
    t0 = -n_lead / fs
    specs = [spectrogram_db(x) for _, x in panels]
    vmin, vmax = shared_db_limits(specs)
    amp = max(float(np.max(np.abs(x))) for _, x in panels) * 1.05
    ncols = len(panels)
    fig, axes = plt.subplots(
        2, ncols, figsize=(3.3 * ncols, 5.0), sharex=True, squeeze=False, layout="constrained",
        gridspec_kw={"height_ratios": [1, 1.4]},
    )
    mesh = None
    for col, ((title, x), spec) in enumerate(zip(panels, specs, strict=True)):
        ax_w, ax_s = axes[0, col], axes[1, col]
        t = t0 + np.arange(x.size) / fs
        ax_w.axvspan(t0, 0.0, color=GRID, alpha=0.8, lw=0)
        ax_w.plot(t, x, color=INK, lw=0.4)
        ax_w.set_ylim(-amp, amp)
        ax_w.set_title(title)
        ax_w.set_ylabel("Amplitude (full scale)")
        mesh = draw_spectrogram(ax_s, spec, vmin, vmax, fs, t0)
        ax_s.axvline(0.0, color="#ffffff", ls="--", lw=0.8, alpha=0.8)
        if col > 0:
            ax_w.set_ylabel("")
            ax_s.set_ylabel("")
    axes[0, 0].text(t0 / 2, 0.0, "noise-only lead-in", ha="center", va="center", fontsize=7,
                    rotation=90, color=INK_SECONDARY)
    cbar = fig.colorbar(mesh, ax=axes[1, :], shrink=0.95)
    cbar.set_label("Magnitude (dB)")
    fig.suptitle(suptitle)
    save_fig(fig, path)
    return vmin, vmax


# ---------------------------------------------------------------------------
# Step 4: musical noise and the alpha x beta sweep
# ---------------------------------------------------------------------------
# Diverging map for improvements (Delta): red = worse, neutral grey at 0, blue = better.
DIVERGING = matplotlib.colors.LinearSegmentedColormap.from_list(
    "delta", ["#a52a2a", "#e66767", "#f0efec", "#6da7ec", "#184f95"]
)


def plot_musical_noise(
    panels: Sequence[tuple[str, np.ndarray]],
    n_lead: int,
    t_region: tuple[float, float],
    path: Path,
    suptitle: str,
    zoom_notes: Sequence[str] | None = None,
    fs: int = config.FS,
) -> tuple[float, float]:
    """Top row: full spectrograms, with the zoomed region outlined. Bottom row:
    the same panels zoomed on the speech-free region ``t_region`` (s), where
    musical noise is isolated, each optionally annotated (``zoom_notes``).
    All eight panels share one dB colour scale."""
    apply_style()
    t0 = -n_lead / fs
    specs = [spectrogram_db(x) for _, x in panels]
    vmin, vmax = shared_db_limits(specs)
    ncols = len(panels)
    fig, axes = plt.subplots(2, ncols, figsize=(3.1 * ncols, 5.8), squeeze=False,
                             sharey=True, layout="constrained")
    mesh = None
    for col, ((title, _), spec) in enumerate(zip(panels, specs, strict=True)):
        top, zoom = axes[0, col], axes[1, col]
        mesh = draw_spectrogram(top, spec, vmin, vmax, fs, t0)
        top.set_title(title)
        top.add_patch(matplotlib.patches.Rectangle(
            (t_region[0], 0.0), t_region[1] - t_region[0], fs / 2,
            fill=False, edgecolor="#ffffff", ls="--", lw=1.0))
        draw_spectrogram(zoom, spec, vmin, vmax, fs, t0)
        zoom.set_xlim(*t_region)
        zoom.set_title("zoom: speech-free region (dashed box)", fontsize=8)
        if zoom_notes and zoom_notes[col]:
            zoom.text(0.03, 0.97, zoom_notes[col], transform=zoom.transAxes, ha="left",
                      va="top", fontsize=7, color=INK,
                      bbox={"boxstyle": "round,pad=0.25", "fc": "#ffffff", "ec": "none",
                            "alpha": 0.85})
        if col > 0:
            top.set_ylabel("")
            zoom.set_ylabel("")
    cbar = fig.colorbar(mesh, ax=axes, shrink=0.9)
    cbar.set_label("Magnitude (dB)")
    fig.suptitle(suptitle)
    save_fig(fig, path)
    return vmin, vmax


def plot_alpha_beta_heatmap(summary: Sequence[dict], path: Path, suptitle: str) -> None:
    """Annotated heatmaps of mean ΔPESQ and mean ΔsegSNR over alpha (rows) x beta (cols)."""
    apply_style()
    alphas = sorted({float(r["alpha"]) for r in summary})
    betas = sorted({float(r["beta"]) for r in summary})
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), layout="constrained")
    for ax, (col, label, fmt) in zip(
        axes,
        (("d_pesq_mean", "Mean ΔPESQ (MOS-LQO)", "{:+.2f}"),
         ("d_segsnr_mean", "Mean ΔsegSNR (dB)", "{:+.1f}")),
        strict=True,
    ):
        grid = np.full((len(alphas), len(betas)), np.nan)
        for r in summary:
            grid[alphas.index(float(r["alpha"])), betas.index(float(r["beta"]))] = r[col]
        lim = float(np.nanmax(np.abs(grid))) or 1.0
        img = ax.imshow(grid, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto",
                        origin="lower")
        for (a_i, b_i), v in np.ndenumerate(grid):
            ax.text(b_i, a_i, fmt.format(v), ha="center", va="center", fontsize=8,
                    color="#ffffff" if abs(v) > 0.6 * lim else INK)
        ax.set_xticks(range(len(betas)), [f"{b:g}" for b in betas])
        ax.set_yticks(range(len(alphas)), [f"{a:g}" for a in alphas])
        ax.set_xlabel("Spectral floor β (fraction of |X|, dimensionless)")
        ax.set_ylabel("Over-subtraction α (dimensionless)")
        ax.set_title(label)
        ax.grid(False)
        cbar = fig.colorbar(img, ax=ax, shrink=0.9)
        cbar.set_label(label)
    fig.suptitle(suptitle)
    save_fig(fig, path)


# ---------------------------------------------------------------------------
# Step 5: Wiener eta sweep
# ---------------------------------------------------------------------------
NOISE_STYLES = {
    "white": {"ls": "-", "mfc": None},  # filled markers
    "babble": {"ls": "--", "mfc": "#ffffff"},  # open markers
}


def plot_eta_sweep(summary: Sequence[dict], path: Path, suptitle: str) -> None:
    """ΔPESQ, ΔSTOI and ΔsegSNR against eta, one line per noise type (mean ± 1 std).

    The x axis is log(1 - eta), reversed, so the grid values 0.5 ... 0.99
    are spread out and eta still increases to the right.
    """
    apply_style()
    color = METHOD_COLORS["wiener"]
    metrics = (("d_pesq", "ΔPESQ (MOS-LQO)"), ("d_stoi", "ΔSTOI"), ("d_segsnr", "ΔsegSNR (dB)"))
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), layout="constrained")
    etas = sorted({float(r["eta"]) for r in summary})
    for ax, (col, label) in zip(axes, metrics, strict=True):
        for noise_type, style in NOISE_STYLES.items():
            sel = sorted((r for r in summary if r["noise"] == noise_type),
                         key=lambda r: float(r["eta"]))
            if not sel:
                continue
            x = [1.0 - float(r["eta"]) for r in sel]
            ax.errorbar(
                x, [r[f"{col}_mean"] for r in sel], yerr=[r[f"{col}_std"] for r in sel],
                color=color, ls=style["ls"], marker=METHOD_MARKERS["wiener"], ms=6,
                mfc=style["mfc"] or color, mec=color, capsize=3, lw=1.5,
                label=f"{noise_type} noise",
            )
        ax.axhline(0.0, color=INK_MUTED, lw=0.8)
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xticks([1.0 - e for e in etas], [f"{e:g}" for e in etas])
        ax.minorticks_off()
        ax.set_xlabel("Decision-directed smoothing η (log scale in 1 − η)")
        ax.set_ylabel(label)
        ax.set_title(label)
    axes[0].legend(loc="best")
    fig.suptitle(suptitle)
    save_fig(fig, path)


# ---------------------------------------------------------------------------
# Step 6: transients and the wavelet sweep
# ---------------------------------------------------------------------------
TRACE_ORDER = ("clean", "noisy", "spectral_sub", "wiener", "wavelet", "wavelet_ld")
CLEAN_GHOST = "#c3c2b7"  # faint clean reference drawn behind each processed trace


def plot_transient_closeup(
    traces: dict, pair: int, path: Path, envelopes: dict, suptitle: str, fs: int = config.FS
) -> None:
    """Row 1: two consecutive clicks; row 2: a real plosive onset; row 3: the
    click-synchronous energy envelope (dB) of every signal, overlaid.

    ``traces[case] = (signals, onsets, description)`` where ``signals`` maps a
    TRACE_ORDER key to the lead-in-trimmed waveform. Each processed panel
    shows the clean waveform faintly behind it; onsets are dashed and the
    10 ms pre-echo windows are shaded. One amplitude scale per row.
    ``envelopes[key] = (t_ms, dB)`` from experiments.click_envelope_db.
    """
    apply_style()
    keys = [k for k in TRACE_ORDER if k in traces["click_train"][0]]
    pre = int(round(config.TRANSIENT_PRE_S * fs))
    margin_c, before_p, after_p = int(0.030 * fs), int(0.040 * fs), int(0.060 * fs)
    fig = plt.figure(figsize=(2.7 * len(keys), 8.4), layout="constrained")
    grid = fig.add_gridspec(3, len(keys), height_ratios=[1, 1, 1.25])
    axes = np.empty((2, len(keys)), dtype=object)
    for row in range(2):
        for col in range(len(keys)):
            share = axes[row, 0] if col else None
            axes[row, col] = fig.add_subplot(grid[row, col], sharey=share)
            if col:
                axes[row, col].tick_params(labelleft=False)
    for row, case in enumerate(("click_train", "plosive")):
        signals, onsets, desc = traces[case]
        if case == "click_train":
            shown = onsets[pair : pair + 2]
            a, b = shown[0] - margin_c, shown[-1] + margin_c
        else:
            shown = onsets[:1]
            a, b = shown[0] - before_p, shown[0] + after_p
        ref = shown[0]
        t_ms = (np.arange(a, b) - ref) * 1000.0 / fs
        amp = 1.1 * max(float(np.max(np.abs(signals[k][a:b]))) for k in keys)
        for col, key in enumerate(keys):
            ax = axes[row, col]
            for o in shown:
                ax.axvspan((o - pre - ref) * 1000.0 / fs, (o - ref) * 1000.0 / fs,
                           color=GRID, alpha=0.9, lw=0)
                ax.axvline((o - ref) * 1000.0 / fs, color=INK_MUTED, ls="--", lw=0.8)
            if key != "clean":
                ax.plot(t_ms, signals["clean"][a:b], color=CLEAN_GHOST, lw=0.8)
            ax.plot(t_ms, signals[key][a:b], color=METHOD_COLORS[key], lw=0.9)
            ax.set_ylim(-amp, amp)
            ax.set_xlim(t_ms[0], t_ms[-1])
            ax.set_xlabel("Time re onset (ms)")
            if row == 0:
                ax.set_title(METHOD_LABELS[key], fontsize=9)
        axes[row, 0].set_ylabel(f"{desc}\nAmplitude (full scale)", fontsize=8)

    ax = fig.add_subplot(grid[2, :])
    ax.axvspan(-1000.0 * config.TRANSIENT_PRE_S, 0.0, color=GRID, alpha=0.9, lw=0)
    ax.axvline(0.0, color=INK_MUTED, ls="--", lw=0.8)
    for key in keys:
        t_ms, env = envelopes[key]
        ax.plot(t_ms, env, color=METHOD_COLORS[key], lw=2.2 if key == "clean" else 1.4,
                label=METHOD_LABELS[key])
    finite = np.concatenate([e[np.isfinite(e)] for _, e in envelopes.values()])
    ax.set_ylim(max(float(finite.min()), -60.0) - 3.0, 5.0)
    ax.set_xlim(t_ms[0], t_ms[-1])
    ax.set_xlabel("Time re click onset (ms)")
    ax.set_ylabel("Energy (dB re clean click peak)")
    ax.set_title("Click-synchronous energy envelope, averaged over all clicks "
                 f"({1000 * config.ENVELOPE_SMOOTH_S:g} ms smoothing): pre-echo = energy "
                 "before 0 ms, smearing = width of the peak", fontsize=9)
    ax.legend(loc="upper right", ncols=2, fontsize=8)
    fig.suptitle(suptitle)
    save_fig(fig, path)


def plot_wavelet_sweep(summary: Sequence[dict], path: Path, suptitle: str) -> None:
    """Mean ΔPESQ heatmaps (family x level) for each noise type (rows) and variant (cols)."""
    apply_style()
    families = list(config.WAVELET_FAMILIES)
    levels = sorted({int(r["level"]) for r in summary})
    noises = [n for n in config.NOISE_TYPES if any(r["noise"] == n for r in summary)]
    variants = list(config.WAVELET_VARIANTS)
    grids = {}
    for n in noises:
        for v in variants:
            g = np.full((len(families), len(levels)), np.nan)
            for r in summary:
                if r["noise"] == n and r["variant"] == v:
                    g[families.index(r["wavelet"]), levels.index(int(r["level"]))] = \
                        r["d_pesq_mean"]
            grids[n, v] = g
    lim = max(float(np.nanmax(np.abs(g))) for g in grids.values()) or 1.0
    fig, axes = plt.subplots(len(noises), len(variants), figsize=(9.0, 3.0 * len(noises)),
                             squeeze=False, layout="constrained")
    img = None
    for i, n in enumerate(noises):
        for j, v in enumerate(variants):
            ax, g = axes[i, j], grids[n, v]
            img = ax.imshow(g, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
            for (fi, li), val in np.ndenumerate(g):
                ax.text(li, fi, f"{val:+.2f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if abs(val) > 0.6 * lim else INK)
            ax.set_xticks(range(len(levels)), [str(lev) for lev in levels])
            ax.set_yticks(range(len(families)), families)
            ax.set_xlabel("Decomposition level L (number of scales)")
            ax.set_ylabel("Wavelet family")
            ax.set_title(f"{n} noise, {v.replace('_', '-')} threshold")
            ax.grid(False)
    cbar = fig.colorbar(img, ax=axes, shrink=0.9)
    cbar.set_label("Mean ΔPESQ (MOS-LQO)")
    fig.suptitle(suptitle)
    save_fig(fig, path)


# ---------------------------------------------------------------------------
# Step 7: cross-method figures
# ---------------------------------------------------------------------------
METRIC_AXES = (("segsnr", "segSNR (dB)"), ("pesq", "PESQ (MOS-LQO)"), ("stoi", "STOI"))
QUADRANT = "#f6d9d9"  # light red wash for the "fidelity up, perceptual down" quadrant


def plot_metric_vs_snr(summary: Sequence[dict], path: Path, suptitle: str,
                       perceptual: bool = True) -> None:
    """segSNR, PESQ and STOI against input SNR: one line per method (mean ± 1 std)
    and the noisy input as a dashed baseline. Methods are offset horizontally by
    a fraction of a dB so the error bars do not overlap."""
    apply_style()
    methods = list(dict.fromkeys(r["method"] for r in summary))
    snrs = sorted({r["snr_in"] for r in summary})
    axes_spec = METRIC_AXES if perceptual else METRIC_AXES[:1]
    fig, axes = plt.subplots(1, len(axes_spec), figsize=(4.0 * len(axes_spec), 3.8),
                             layout="constrained", squeeze=False)
    step = 0.25
    for ax, (m, label) in zip(axes[0], axes_spec, strict=True):
        base = [next(r for r in summary if r["snr_in"] == s) for s in snrs]
        ax.errorbar(snrs, [b[f"{m}_noisy_mean"] for b in base],
                    yerr=[b[f"{m}_noisy_std"] for b in base], color=METHOD_COLORS["noisy"],
                    ls="--", marker=METHOD_MARKERS["noisy"], ms=6, capsize=3,
                    label=METHOD_LABELS["noisy"])
        for i, method in enumerate(methods):
            sel = [next(r for r in summary if r["method"] == method and r["snr_in"] == s)
                   for s in snrs]
            x = [s + (i - (len(methods) - 1) / 2) * step for s in snrs]
            ax.errorbar(x, [r[f"{m}_out_mean"] for r in sel],
                        yerr=[r[f"{m}_out_std"] for r in sel], color=METHOD_COLORS[method],
                        marker=METHOD_MARKERS[method], ms=6, capsize=3,
                        label=METHOD_LABELS[method])
        ax.set_xticks(snrs)
        ax.set_xlabel("Input SNR (dB)")
        ax.set_ylabel(label)
        ax.set_title(label)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(labels), fontsize=8)
    fig.suptitle(suptitle)
    save_fig(fig, path)


def plot_spectrogram_grid(
    grid: dict[str, Sequence[tuple[str, np.ndarray]]], n_lead: int, path: Path, suptitle: str,
    fs: int = config.FS,
) -> None:
    """Rows: noise types; columns: clean, noisy, methods. One dB scale for every panel."""
    apply_style()
    t0 = -n_lead / fs
    specs = {n: [spectrogram_db(x) for _, x in panels] for n, panels in grid.items()}
    vmin, vmax = shared_db_limits([s for ss in specs.values() for s in ss])
    ncols = max(len(p) for p in grid.values())
    fig, axes = plt.subplots(len(grid), ncols, figsize=(3.0 * ncols, 2.7 * len(grid)),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    mesh = None
    for i, (noise_type, panels) in enumerate(grid.items()):
        for j, ((title, _), spec) in enumerate(zip(panels, specs[noise_type], strict=True)):
            ax = axes[i, j]
            mesh = draw_spectrogram(ax, spec, vmin, vmax, fs, t0)
            ax.set_title(f"{title}" if i else title)
            if j:
                ax.set_ylabel("")
            else:
                ax.set_ylabel(f"{noise_type} noise\nFrequency (Hz)")
            if i < len(grid) - 1:
                ax.set_xlabel("")
    cbar = fig.colorbar(mesh, ax=axes, shrink=0.9)
    cbar.set_label("Magnitude (dB)")
    fig.suptitle(suptitle)
    save_fig(fig, path)


def plot_fidelity_vs_perceptual(rows: Sequence[dict], path: Path, suptitle: str) -> None:
    """ΔsegSNR (x) against ΔPESQ (y) for every sweep configuration and case.

    Small multiples: rows are noise types, columns are methods (colour and
    marker per method). The quadrant ΔsegSNR > 0, ΔPESQ < 0 is shaded, its
    points are drawn opaque (others faint), and each panel reports its count.
    """
    apply_style()
    keys = [method_key(r["method"], r["variant"]) for r in rows]
    methods = [k for k in ("spectral_sub", "wiener", "wavelet", "wavelet_ld") if k in keys]
    noises = [n for n in config.NOISE_TYPES if any(r["noise"] == n for r in rows)]
    xs = np.array([float(r["d_segsnr"]) for r in rows])
    ys = np.array([float(r["d_pesq"]) for r in rows])
    pad_x, pad_y = 0.05 * np.ptp(xs), 0.05 * np.ptp(ys)
    xlim, ylim = (xs.min() - pad_x, xs.max() + pad_x), (ys.min() - pad_y, ys.max() + pad_y)
    fig, axes = plt.subplots(len(noises), len(methods), figsize=(3.3 * len(methods),
                             3.1 * len(noises)), sharex=True, sharey=True, squeeze=False,
                             layout="constrained")
    for i, noise_type in enumerate(noises):
        for j, method in enumerate(methods):
            ax = axes[i, j]
            sel = [r for r, k in zip(rows, keys, strict=True)
                   if r["noise"] == noise_type and k == method]
            x = np.array([float(r["d_segsnr"]) for r in sel])
            y = np.array([float(r["d_pesq"]) for r in sel])
            dis = (x > 0) & (y < 0)
            ax.add_patch(matplotlib.patches.Rectangle((0, ylim[0]), xlim[1], -ylim[0],
                                                      color=QUADRANT, lw=0, zorder=0))
            ax.axhline(0.0, color=INK_MUTED, lw=0.8)
            ax.axvline(0.0, color=INK_MUTED, lw=0.8)
            style = {"color": METHOD_COLORS[method], "marker": METHOD_MARKERS[method],
                     "s": 5, "linewidths": 0, "rasterized": True}
            ax.scatter(x[~dis], y[~dis], alpha=0.2, **style)
            ax.scatter(x[dis], y[dis], alpha=0.8, **style)
            ax.text(0.97, 0.03, f"{dis.sum()} / {dis.size}\n({100 * dis.mean():.1f} %)",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            if i == 0:
                ax.set_title(METHOD_LABELS[method], fontsize=9)
            if i == len(noises) - 1:
                ax.set_xlabel("ΔsegSNR (dB)")
            if j == 0:
                ax.set_ylabel(f"{noise_type} noise\nΔPESQ (MOS-LQO)")
    fig.suptitle(suptitle)
    save_fig(fig, path)


# ---------------------------------------------------------------------------
# Step 1: COLA figure
# ---------------------------------------------------------------------------
def plot_cola(
    path: Path,
    n_len: int = config.FRAME_LEN,
    hop: int = config.HOP,
    n_windows: int = config.COLA_FIG_WINDOWS,
) -> dict[str, float]:
    """Shifted periodic Hann windows and their overlap-added sum.

    Top: each shifted window (thin) and the sum (thick), flat at 1 in the
    interior. Bottom: the interior sum on a zoomed scale, periodic Hann
    (used) against the symmetric np.hanning (not used), showing why the
    periodic form is required for exact reconstruction.
    Returns the max interior deviation |sum - 1| for both windows.
    """
    apply_style()
    w = periodic_hann(n_len)
    env = window_sum(n_windows, n_len, hop)
    n = np.arange(env.size)
    interior = slice(n_len // 2, env.size - n_len // 2)

    # Symmetric Hann, for comparison only (it is never used for processing).
    w_sym = np.hanning(n_len)
    env_sym = np.zeros(env.size)
    for m in range(n_windows):
        env_sym[m * hop : m * hop + n_len] += w_sym

    dev_periodic = float(np.max(np.abs(env[interior] - 1.0)))
    dev_symmetric = float(np.max(np.abs(env_sym[interior] - 1.0)))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.0, 5.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    for m in range(n_windows):
        seg = np.arange(m * hop, m * hop + n_len)
        ax1.plot(
            seg, w, color=METHOD_COLORS["spectral_sub"], lw=0.8, alpha=0.8,
            label="Shifted windows w[n − mH]" if m == 0 else None,
        )
    ax1.plot(n, env, color=INK, lw=2.2, label="Sum Σₘ w[n − mH]")
    for edge in (interior.start, interior.stop - 1):
        ax1.axvline(edge, color=INK_MUTED, ls=":", lw=1.0)
    ax1.text(
        (interior.start + interior.stop) / 2, 1.04,
        f"interior (between dotted lines): max |Σw − 1| = {dev_periodic:.1e}",
        ha="center", va="bottom", color=INK,
    )
    ax1.set_ylim(-0.05, 1.45)
    ax1.set_ylabel("Amplitude")
    ax1.set_title(
        f"COLA check: periodic Hann, N = {n_len}, H = {hop} (50 % overlap), {n_windows} frames"
    )
    ax1.legend(loc="upper center", ncols=2)

    ax2.plot(n[interior], env[interior], color=INK, lw=2.2,
             label=f"Periodic Hann (used): max dev {dev_periodic:.1e}")
    ax2.plot(n[interior], env_sym[interior], color=METHOD_COLORS["wiener"], lw=1.4, ls="--",
             label=f"Symmetric np.hanning (not used): max dev {dev_symmetric:.1e}")
    ax2.set_xlim(0, env.size - 1)
    ax2.set_ylim(1 - 1.6 * dev_symmetric, 1 + 1.6 * dev_symmetric)
    ax2.set_xlabel("Sample index")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Interior window sum, zoomed")
    # The symmetric sum only dips below 1, so the upper half is free for the legend.
    ax2.legend(loc="upper center", fontsize=8, ncols=1)
    fig.tight_layout()
    save_fig(fig, path)
    return {"max_dev_periodic": dev_periodic, "max_dev_symmetric": dev_symmetric}
