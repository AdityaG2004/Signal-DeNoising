# Speech Denoising in the STFT Domain: A Comparative Study of Classical DSP Methods

This repository implements three classical speech denoisers directly from their equations:

- spectral subtraction with over-subtraction and a spectral floor (Boll 1979; Berouti et al. 1979);
- a short-time Wiener filter with the decision-directed a-priori SNR (Ephraim & Malah 1984; Scalart & Vieira Filho 1996);
- wavelet soft thresholding with universal and level-dependent thresholds (Donoho 1995; Johnstone & Silverman 1997).

It compares them on the 30 NOIZEUS sentences (8 kHz) corrupted by white and babble noise at −5 to 15 dB input SNR, using segmental SNR, narrowband PESQ and STOI. The two Fourier methods share a perfect-reconstruction STFT framework (periodic Hann, 32 ms frames, 50 % overlap). Everything is seeded: one command regenerates every CSV, figure and WAV, byte-identically.

## Setup

Python 3.12 or newer (developed on 3.14.2, macOS arm64). The pinned NumPy, SciPy and PyWavelets require 3.12; check with `python3 --version`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

`pesq` compiles a small C extension. If that fails on macOS, run `xcode-select --install` and retry. `requirements.txt` pins exact versions; the full environment of the reference run is in `results/environment.txt`.

## Getting NOIZEUS

The clean NOIZEUS sentences (Hu & Loizou 2007) are not redistributed here. Download them from the official site into `data/clean/` with:

```bash
python main.py --download --step 0
```

This fetches `clean.zip` from <https://ecs.utdallas.edu/loizou/speech/noizeus/> using the system `curl`, extracts `sp01.wav` … `sp30.wav`, and checks there are 30 mono 8 kHz files. You can also download `clean.zip` by hand and unzip the WAV files into `data/clean/`. Speaker and gender labels (used to build babble noise) are already in `data/speakers.csv`.

Without NOIZEUS the pipeline falls back to a seeded synthetic speech-like signal (see `--synthetic`), so every step still runs.

## Running

```bash
python main.py                  # everything: steps 1-7 on 30 sentences x 5 SNRs
python main.py --quick          # 3 sentences, SNRs {0, 5, 10} dB
python main.py --synthetic      # synthetic speech instead of NOIZEUS
python main.py --step 4         # one build step's experiment and figures (1-7)
python main.py --no-perceptual  # skip PESQ and STOI (fast debugging)
python main.py --workers 4      # worker processes (default: CPU count - 1)
python main.py --resume         # keep finished sweep rows, compute only the missing ones
python main.py --input rec.wav  # denoise one recording of your own (see below)
```

Runtime on an 8-core Apple silicon laptop: about 1 min 50 s for the full run (17,700 denoiser evaluations, each with segSNR, PESQ and STOI) and about 25 s for `--quick`.

`--quick`, `--synthetic` and `--no-perceptual` write to a sub-folder of `results/`, `figures/` and `audio_out/` named after the flags (for example `quick/`, `synthetic_quick/`, `no_perceptual/`), so they never overwrite the full-run outputs.

### Denoising your own recording

```bash
python main.py --input path/to/recording.wav
```

This denoises one file with all four methods at their default settings and writes the results to `audio_out/input/<file name>/`:

- `input.wav` is your file as processed.
- `spectral_sub.wav`, `wiener.wav`, `wavelet.wav` and `wavelet_ld.wav` are the denoised versions.

All five share one volume scale, so you can compare them fairly. The experiments are not run in this mode. Things to know:

- **Any sample rate and channel count works.** The file is mixed down to mono and resampled to 8 kHz (the project's rate), so the output sounds like telephone audio, band-limited to 4 kHz.
- **Formats:** WAV, FLAC, OGG and MP3 are read directly. M4A (iPhone Voice Memos, QuickTime) is not; convert it first with macOS's built-in `afconvert`:
  ```bash
  afconvert -f WAVE -d LEI16 memo.m4a memo.wav
  ```
- **Start the recording with at least 0.25 s of background noise and no speech.** Spectral subtraction and the Wiener filter learn the noise from that stretch. The wavelet method doesn't need it.
- **No scores.** There is no clean reference for your recording, so PESQ, STOI and segSNR cannot be computed. Judge by listening.

What each step produces:

| Step | Produces |
|---|---|
| 1 | STFT perfect-reconstruction checks, `cola.png` |
| 2 | dataset summary, exact-SNR mixing check, `mixing_example.png` |
| 3 | metrics of the unprocessed noisy input |
| 4 | spectral subtraction: musical-noise demo and WAVs, α × β sweep, heatmap |
| 5 | Wiener: η sweep and figure |
| 6 | wavelet: transient experiment, family × level × variant sweep |
| 7 | `all_runs.csv`, summaries, cross-method figures, example WAVs, `sanity_checks.md` |

Tests and lint:

```bash
pytest -q
ruff check .
```

## Layout

```
main.py              single entry point (CLI above)
config.py            every parameter (no magic numbers in src/)
requirements.txt     pinned dependencies
pyproject.toml       pytest and ruff configuration
src/
  stft.py            periodic Hann, STFT / iSTFT with half-frame padding, COLA check
  data.py            NOIZEUS loader and downloader, synthetic speech, white and babble noise
  mixing.py          exact-SNR mixing with a 0.25 s noise-only lead-in
  spectral_sub.py    method 1
  wiener.py          method 2
  wavelet.py         method 3
  metrics.py         segSNR, global SNR, PESQ-NB, STOI, alignment check
  experiments.py     step runners, parallel sweep engine, summaries, sanity checks
  plots.py           all figures (one colour-blind-safe colour per method)
tests/               pytest, one file per module (157 tests)
data/speakers.csv    NOIZEUS speaker and gender per sentence
data/clean/          NOIZEUS WAVs (downloaded, not in the repository)
results/             CSVs, summary tables, FINDINGS.md, sanity checks
figures/             report figures (PNG, 300 dpi)
audio_out/           WAVs for listening (regenerated, not in the repository)
```

The repository keeps only the full-run outputs. Intermediates that `python main.py` regenerates are git-ignored (see `.gitignore`): the per-sweep `runs_*.csv` (their rows are all in `all_runs.csv`), raw timings, and the `quick/`, `synthetic/` and `no_perceptual/` debug folders.

## Outputs and the objectives they support

Objectives O1–O6 are defined in `midsem-report.pdf` §2. All paths below are under `results/` or `figures/`.

| Output | What it shows | Objective |
|---|---|---|
| `cola.png`, `stft_checks.csv` | STFT → iSTFT round-trip error (worst 8.9e-16); periodic vs symmetric Hann overlap-add | O1 |
| `dataset_summary.csv`, `mixing_check.csv`, `mixing_example.png` | 30 sentences; every mixture within 2e-15 dB of its target SNR; lead-in | O3 |
| `noisy_baseline.csv` | segSNR, PESQ and STOI of the unprocessed input | O3, O4 |
| `musical_noise.png`, `musical_noise.csv` | musical noise at α = 1, β = 0 vs the defaults; surviving-bin fraction | O2, O5 |
| `alpha_beta_heatmap.png`, `ss_sweep_summary*.csv` | spectral subtraction over α × β | O2, O5 |
| `eta_sweep.png`, `wiener_sweep_summary*.csv` | Wiener filter over η | O2 |
| `wavelet_sweep.png`, `wavelet_sweep_summary*.csv`, `wavelet_band_check.csv` | wavelet family × level × variant; what the thresholds keep per band | O2, O5 |
| `transient_closeup.png`, `transient_metrics.csv`, `transient_envelope.csv` | click train and a real plosive: onset concentration, pre-echo, click-synchronous energy envelope | O5 |
| `noise_estimate_check.csv` | how well the lead-in noise estimate represents white vs babble noise | O5 |
| `all_runs.csv`, `summary.csv`, `summary.md`, `metric_vs_snr_{white,babble}.png` | every run; means ± std at the defaults against input SNR | O4 |
| `spectrogram_grid.png` | clean / noisy / three methods at 5 dB, both noise types | O4, O5 |
| `fidelity_vs_perceptual.png`, `metric_disagreement*.csv`, `pesq_stability_check.csv` | cases where segSNR improves while PESQ worsens; unstable PESQ scores | O4 |
| `sanity_checks.md`, `config_used.json`, `environment.txt`, `csv_sha256.txt` | result checks, parameters, environment, reproducibility hashes | O6 |
| `FINDINGS.md`, `../AUDIT.md` | findings with sources; equation and reproducibility audit | all |

## Known limitations

- **Static noise estimate.** Spectral subtraction and the Wiener filter estimate the noise once, from the 0.25 s lead-in. That works for stationary white noise but not for babble, whose level varies by about 4.7 dB from frame to frame (`noise_estimate_check.csv`). Minimum-statistics tracking (Martin 2001) would be the natural extension.
- **SNR definition.** Mixing uses plain signal power over the whole sentence, not the ITU-T P.56 active speech level that NOIZEUS used for its own noisy files. At the same nominal SNR our mixtures contain less noise, so absolute scores are not directly comparable with published NOIZEUS results.
- **The wavelet method does not use the lead-in.** It estimates the noise blindly (MAD of the wavelet coefficients), while the two Fourier methods get a noise-only lead-in. With the universal threshold it removes much of the speech along with the noise (see `FINDINGS.md`).
- **PESQ is narrowband.** At 8 kHz only ITU-T P.862 narrowband mode applies; the scores are P.862.1 MOS-LQO. PESQ is unreliable on outputs with almost no in-band speech: a few such scores change by more than 0.5 when the output is delayed by one sample (`pesq_stability_check.csv`). Those rows are kept and flagged, not removed.
- **Timings are separate.** Denoiser runtimes are summarised per configuration in `timing_summary.csv` (raw per-run times in `timing_<sweep>.csv`, not committed), not in `all_runs.csv`, because wall-clock times cannot be byte-reproducible.

## References

1. S. F. Boll, "Suppression of acoustic noise in speech using spectral subtraction," *IEEE Trans. Acoustics, Speech, and Signal Processing*, 27(2), 113–120, 1979.
2. M. Berouti, R. Schwartz and J. Makhoul, "Enhancement of speech corrupted by acoustic noise," *Proc. IEEE ICASSP*, 208–211, 1979.
3. Y. Ephraim and D. Malah, "Speech enhancement using a minimum mean-square error short-time spectral amplitude estimator," *IEEE Trans. ASSP*, 32(6), 1109–1121, 1984.
4. P. Scalart and J. Vieira Filho, "Speech enhancement based on a priori signal to noise estimation," *Proc. IEEE ICASSP*, 629–632, 1996.
5. D. L. Donoho, "De-noising by soft-thresholding," *IEEE Trans. Information Theory*, 41(3), 613–627, 1995.
6. I. M. Johnstone and B. W. Silverman, "Wavelet threshold estimators for data with correlated noise," *J. Royal Statistical Society B*, 59(2), 319–351, 1997.
7. R. Martin, "Noise power spectral density estimation based on optimal smoothing and minimum statistics," *IEEE Trans. Speech and Audio Processing*, 9(5), 504–512, 2001.
8. J. B. Allen and L. R. Rabiner, "A unified approach to short-time Fourier analysis and synthesis," *Proc. IEEE*, 65(11), 1558–1564, 1977.
9. Y. Hu and P. C. Loizou, "Subjective comparison and evaluation of speech enhancement algorithms," *Speech Communication*, 49(7–8), 588–601, 2007 (NOIZEUS).
10. P. C. Loizou, *Speech Enhancement: Theory and Practice*, 2nd ed., CRC Press, 2013.
11. C. H. Taal, R. C. Hendriks, R. Heusdens and J. Jensen, "An algorithm for intelligibility prediction of time-frequency weighted noisy speech," *IEEE Trans. Audio, Speech, and Language Processing*, 19(7), 2125–2136, 2011.
12. ITU-T Recommendation P.862, "Perceptual evaluation of speech quality (PESQ)," 2001; P.862.1 (MOS-LQO mapping), 2003.
