"""Single entry point for the EE 317 speech-denoising study.

    python main.py [--quick] [--synthetic] [--step N] [--no-perceptual] [--workers K]

With no ``--step``, every implemented step runs in order and regenerates all
CSVs, figures and WAVs.
"""

import argparse
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

from src import experiments
from src.data import download_noizeus

ALL_STEPS = tuple(range(10))

# Steps that are verified by pytest alone and write no CSV, figure or WAV.
TEST_ONLY_STEPS = frozenset({0, 8, 9})

# Step number -> callable(args). Filled in as each build step lands.
STEP_RUNNERS: dict[int, Callable[[argparse.Namespace], None]] = {
    1: experiments.run_step1,
    2: experiments.run_step2,
    3: experiments.run_step3,
    4: experiments.run_step4,
    5: experiments.run_step5,
    6: experiments.run_step6,
    7: experiments.run_step7,
}


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Speech denoising in the STFT domain: spectral subtraction, Wiener "
            "filtering and wavelet thresholding on NOIZEUS (8 kHz)."
        )
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="fast iteration: 3 sentences and input SNRs {0, 5, 10} dB only",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="force the seeded synthetic speech fallback instead of NOIZEUS",
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=ALL_STEPS,
        metavar="N",
        help="run only the experiment/figures for build step N (0-9)",
    )
    parser.add_argument(
        "--no-perceptual",
        action="store_true",
        help="skip PESQ and STOI (fast debugging)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        metavar="FILE",
        help="denoise one recording with every method instead of running the experiments "
        "(converted to mono 8 kHz; its first 0.25 s must be noise only); "
        "writes audio_out/input/<file name>/",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="download the NOIZEUS clean sentences into data/clean/ first (if missing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep rows already in results/runs_*.csv and compute only the missing ones",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        metavar="K",
        help="number of worker processes (default: CPU count minus 1)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        build_parser().error("--workers must be >= 1")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.download:
        download_noizeus()
    if args.input is not None:
        if not args.input.is_file():
            build_parser().error(f"--input: no such file: {args.input}")
        folder = experiments.denoise_file(args.input)
        print(f"Denoised versions of {args.input.name} written to {folder}")
        return 0

    steps = (args.step,) if args.step is not None else ALL_STEPS
    for step in steps:
        runner = STEP_RUNNERS.get(step)
        if runner is not None:
            runner(args)
        elif step in TEST_ONLY_STEPS:
            if args.step is not None:
                print(f"Step {step}: no experiment or figure (verified by pytest)")
        else:
            print(f"Step {step}: not implemented yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
