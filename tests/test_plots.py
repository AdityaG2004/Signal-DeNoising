"""Plotting helpers: COLA figure and shared-dB-scale spectrograms."""

import numpy as np
import pytest

import config
from src import plots
from src.stft import spectrogram_db


def test_plot_cola_writes_file_and_reports_deviation(tmp_path):
    path = tmp_path / "cola.png"
    devs = plots.plot_cola(path)
    assert path.exists() and path.stat().st_size > 0
    assert devs["max_dev_periodic"] < 1e-15
    assert devs["max_dev_symmetric"] > 1e-3


def test_shared_db_limits_span_all_panels():
    rng = np.random.default_rng(config.SEED)
    loud = rng.standard_normal(4000)
    quiet = 0.01 * rng.standard_normal(4000)
    specs = [spectrogram_db(loud), spectrogram_db(quiet)]
    vmin, vmax = plots.shared_db_limits(specs)
    assert vmax >= max(s.max() for s in specs)
    assert vmax - vmin == pytest.approx(config.SPEC_DYN_RANGE_DB)


def test_plot_spectrograms_uses_one_scale(tmp_path):
    rng = np.random.default_rng(config.SEED)
    sigs = [rng.standard_normal(4000), 0.1 * rng.standard_normal(4000)]
    path = tmp_path / "spec.png"
    vmin, vmax = plots.plot_spectrograms(sigs, ["a", "b"], path, "test")
    assert path.exists()
    assert (vmin, vmax) == plots.shared_db_limits([spectrogram_db(s) for s in sigs])


def test_plot_spectrograms_rejects_mismatched_titles(tmp_path):
    with pytest.raises(ValueError):
        plots.plot_spectrograms([np.zeros(500)], ["a", "b"], tmp_path / "x.png", "t")


def test_method_palette_complete():
    for key in ("spectral_sub", "wiener", "wavelet", "wavelet_ld", "noisy"):
        assert key in plots.METHOD_COLORS
        assert key in plots.METHOD_MARKERS
        assert key in plots.METHOD_LABELS
    colors = [plots.METHOD_COLORS[k] for k in ("spectral_sub", "wiener", "wavelet", "wavelet_ld")]
    assert len(set(colors)) == len(colors)
