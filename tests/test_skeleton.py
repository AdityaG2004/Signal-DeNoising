"""Step 0 gate: the CLI parses and the config is self-consistent."""

import subprocess
import sys

import numpy as np
import pytest

import config
import main


def test_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=config.ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    for flag in ("--quick", "--synthetic", "--step", "--no-perceptual", "--workers",
                 "--resume", "--download"):
        assert flag in result.stdout


def test_cli_flags_parse():
    args = main.build_parser().parse_args(
        ["--quick", "--synthetic", "--step", "4", "--no-perceptual", "--workers", "2"]
    )
    assert args.quick and args.synthetic and args.no_perceptual
    assert args.step == 4
    assert args.workers == 2


def test_step_out_of_range_rejected():
    with pytest.raises(SystemExit):
        main.build_parser().parse_args(["--step", "10"])


def test_unimplemented_step_exits_zero(capsys, monkeypatch):
    # Every step now has a runner or is test-only, so remove one to reach the path.
    runners = dict(main.STEP_RUNNERS)
    runners.pop(7)
    monkeypatch.setattr(main, "STEP_RUNNERS", runners)
    assert main.main(["--step", "7"]) == 0
    assert "not implemented yet" in capsys.readouterr().out


def test_test_only_step_exits_zero(capsys):
    assert main.main(["--step", "0"]) == 0
    assert "verified by pytest" in capsys.readouterr().out


def test_config_values():
    assert config.FS == 8000
    assert config.FRAME_LEN == config.NFFT == 256
    assert config.HOP == config.FRAME_LEN // 2
    assert config.N_LEAD == 2000
    assert config.SNRS_DB == (-5, 0, 5, 10, 15)
    assert set(config.QUICK_SNRS_DB) <= set(config.SNRS_DB)
    assert config.SS_ALPHA in config.SS_ALPHA_GRID
    assert config.SS_BETA in config.SS_BETA_GRID
    assert config.WIENER_ETA in config.WIENER_ETA_GRID
    assert config.WAVELET in config.WAVELET_FAMILIES
    assert config.WAVELET_LEVEL in config.WAVELET_LEVELS


def test_output_dirs_by_mode(tmp_path, monkeypatch):
    for name in ("RESULTS_DIR", "FIGURES_DIR", "AUDIO_DIR"):
        monkeypatch.setattr(config, name, tmp_path / name.lower())
    assert config.output_dirs() == (
        tmp_path / "results_dir", tmp_path / "figures_dir", tmp_path / "audio_dir"
    )
    assert config.output_dirs(quick=True)[1] == tmp_path / "figures_dir" / "quick"
    assert config.output_dirs(synthetic=True)[0] == tmp_path / "results_dir" / "synthetic"
    both = config.output_dirs(quick=True, synthetic=True)
    assert both[2] == tmp_path / "audio_dir" / "synthetic_quick"
    assert all(d.is_dir() for d in both)
    fast = config.output_dirs(perceptual=False)
    assert fast[0] == tmp_path / "results_dir" / "no_perceptual"
    looked_up = config.output_dirs(quick=True, perceptual=False, create=False)
    assert not any(d.exists() for d in looked_up)  # a lookup creates nothing


def test_case_rng_reproducible_and_distinct():
    a = config.case_rng(3, 1, 2).standard_normal(1000)
    b = config.case_rng(3, 1, 2).standard_normal(1000)
    c = config.case_rng(3, 1, 3).standard_normal(1000)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
