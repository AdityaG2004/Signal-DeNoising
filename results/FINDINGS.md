# Findings

Raw material for the final report. Every number below comes from a CSV that `python main.py` writes; the source file is named in each section (all in `results/`). Conventions:

- 30 NOIZEUS sentences, white and babble noise, input SNR −5 to 15 dB. Metrics are computed on the sentence only (the 0.25 s lead-in is trimmed off).
- Δ is output minus noisy input. PESQ is narrowband P.862.1 MOS-LQO.
- "Mean over SNRs" is the mean of the five per-SNR means in `summary.csv`.
- **Defaults:**
  - spectral subtraction (SS): α = 2, β = 0.02;
  - Wiener: η = 0.98, ξ_min = −25 dB;
  - wavelet: db8, level 5, universal threshold ("universal"), plus the level-dependent threshold ("level-dependent").

## 1. Framework and data (O1, O3)

- **STFT round trip.** Worst max-abs error 8.9e-16 over 35 signals (5 white-noise lengths down to 100 samples, plus all 30 sentences); 2.2e-16 on the sentences alone (`stft_checks.csv`).
- **Window sum.** The periodic Hann sum deviates from 1 by 4.4e-16 in the interior; the symmetric `np.hanning` deviates by 6.2e-3 (`stft_checks.csv`, `cola.png`).
- **Dataset.** 30 sentences, 2.12–3.51 s each, 80.0 s in total (`dataset_summary.csv`).
- **Mixing.** Every one of the 300 mixtures is within 1.9e-15 dB of its target SNR (`mixing_check.csv`).
- **Noisy-input segSNR sits below the global SNR** (`noisy_baseline.csv`), because quiet speech frames count as much as loud ones:
  - white: by 1.4 dB at −5 dB up to 6.6 dB at 15 dB;
  - babble: by 0.2 dB up to 5.0 dB.
- **Babble looks "easier" than white at the same SNR** (`noisy_baseline.csv`). At 5 dB, noisy segSNR is 1.45 vs 0.10 dB and noisy PESQ is 1.66 vs 1.51.
  - Babble power fluctuates over time, and segSNR averages log ratios, so frames where the babble dips score high.
  - The SNRs themselves are exact.

## 2. The three methods at their defaults (O2, O4)

Mean over the five input SNRs (`summary.csv`):

| Method | White: ΔsegSNR / ΔPESQ / ΔSTOI | Babble: ΔsegSNR / ΔPESQ / ΔSTOI |
|---|---|---|
| Spectral subtraction | +5.30 dB / +0.551 / +0.006 | +0.92 dB / −0.013 / −0.043 |
| Wiener | +5.81 dB / +0.595 / −0.004 | +0.87 dB / −0.022 / −0.071 |
| Wavelet, universal | −0.17 dB / −0.350 / −0.164 | +0.03 dB / −0.152 / −0.064 |
| Wavelet, level-dependent | −1.04 dB / −0.441 / −0.230 | −1.35 dB / −0.586 / −0.254 |

By input SNR (`summary.csv`, mean ± std over 30 sentences):

- **White noise, Wiener.** It has the largest segSNR gain at every SNR, from +7.92 ± 0.59 dB at −5 dB down to +3.25 ± 0.48 dB at 15 dB.
- **White noise, PESQ gains grow with input SNR:**
  - SS: +0.15 → +0.93;
  - Wiener: +0.22 → +0.89;
  - at 15 dB SS edges ahead (PESQ 3.03 vs 2.99).
- **White noise, STOI** barely changes for either Fourier method (ΔSTOI from −0.022 to +0.019).
- **Babble, both Fourier methods:**
  - The segSNR gain shrinks with SNR and turns negative at 15 dB (SS −0.63 dB, Wiener −0.61 dB).
  - ΔPESQ is negative up to 5 dB and small but positive at 10–15 dB (SS +0.022 and +0.096; Wiener +0.014 and +0.069).
  - ΔSTOI is negative at every SNR: SS −0.094 to −0.012, Wiener −0.122 to −0.032.
- **Wavelet, universal, white noise.** It raises segSNR only at low SNR (+3.48 dB at −5 dB, +1.87 dB at 0 dB) and lowers it above 5 dB (−4.26 dB at 15 dB). PESQ and STOI drop at every SNR:
  - ΔPESQ −0.245 at −5 dB to −0.539 at 15 dB;
  - ΔSTOI −0.218 to −0.117.

## 3. Musical noise in spectral subtraction (O5)

Source: `musical_noise.csv`, `musical_noise.png`. Case: sp01, white noise at 5 dB. The statistics use all speech-free frames after the lead-in.

- **α = 1, β = 0:** 45.5 % of bins survive. The residual is −8.5 dB relative to the noisy input, and the rest of the bins are zeroed. This is the classic pattern of isolated peaks switching on and off.
- **α = 2, β = 0.02 (defaults):** 5.6 % of bins survive and the residual is −20.4 dB.
- **The defaults reduce musical noise but do not remove it.** The surviving peaks are fewer and weaker, but they stand out as isolated specks against the −34 dB floor (visible in the spectrogram).
- **Metrics for the two settings:**
  - α = 1, β = 0: ΔsegSNR +4.72 dB, ΔPESQ +0.315, ΔSTOI +0.004.
  - Defaults: ΔsegSNR +6.16 dB, ΔPESQ +0.507, ΔSTOI −0.018.
- **Theory check.** On pure white noise, the zeroed-bin fraction at α = 1, β = 0 matches the Rayleigh prediction 1 − e^(−π/4) = 0.544 (tested in `tests/test_spectral_sub.py`).

## 4. Spectral subtraction: the α × β trade-off (O5)

Source: `ss_sweep_summary_all_snr.csv` (mean over 30 sentences × 5 SNRs), `ss_sweep_summary.csv`, `alpha_beta_heatmap.png`.

- **White noise:**
  - Best ΔPESQ: α = 2, β = 0.05 (+0.562), barely above the defaults (+0.551).
  - Best ΔsegSNR: α = 2, β = 0.01 (+5.30 dB).
  - At 5 dB only, the best heatmap cell is again α = 2, β = 0.05 (+0.584).
- **Over-subtraction past α = 2 hurts every metric:**
  - α = 5, β = 0: ΔsegSNR +2.92 dB, ΔPESQ +0.186, ΔSTOI −0.092 (the worst STOI in the grid).
  - Raising the floor to β = 0.1 at α = 5 recovers STOI to −0.022.
- **At α = 1 the floor barely matters:** ΔPESQ is +0.324 at β = 0 and +0.322 at β = 0.1. With 45 % of bins surviving, a floor 34 dB down is masked.
- **Babble:**
  - Only α = 1 and settings with a high floor are positive, and only slightly. The best is α = 5, β = 0.1 (+0.078); α = 2, β = 0.1 gives +0.065.
  - The defaults give −0.013; α = 5, β = 0 gives −0.174.
  - ΔSTOI is negative for every setting (−0.014 to −0.134).

## 5. Wiener: η, smoothing vs responsiveness (O2)

Source: `wiener_sweep_summary_all_snr.csv`, `eta_sweep.png`.

- **White noise:**
  - ΔPESQ peaks at η = 0.95 (+0.617), against +0.595 at the default 0.98.
  - ΔsegSNR also peaks at 0.95 (+6.07 dB).
  - η = 0.5 is clearly worse (+0.352, +4.05 dB).
- **ΔSTOI falls as η rises,** for white (+0.022 at 0.5 to −0.022 at 0.99) and babble (−0.018 to −0.096). A slow-tracking ξ lags speech onsets, which is the trade-off predicted in report §3.3.
- **Babble:** ΔPESQ stays within ±0.05 for every η (best +0.002 at 0.9).

## 6. Wavelet thresholding: universal vs level-dependent (O2, O5)

Sources: `wavelet_sweep_summary_all_snr.csv`, `wavelet_band_check.csv`, `wavelet_sweep.png`.

- **Every one of the 48 configurations lowers PESQ** (3 families × 4 levels × 2 variants × 2 noises; the best mean ΔPESQ is −0.105).
  - Level 3 is the least harmful in every case. The best is db8 L3 for white noise (universal −0.148, level-dependent −0.189) and sym8 L3 for babble (universal −0.105).
  - The choice of family (db4 / db8 / sym8) matters little.
- **Level-dependent is worse than universal for both noise types:**
  - white: −0.189 vs −0.148 at best;
  - babble: −0.335 vs −0.105.
- **Why the universal threshold removes speech.** Band breakdown for sp01, white noise at 5 dB, db8 L5 (`wavelet_band_check.csv`):
  - λ = 0.0906 applies to every level.
  - Only 0.02 % of cD₁ (2–4 kHz), 2.3 % of cD₂, 5.0 % of cD₃, 21 % of cD₄ and 8.5 % of cD₅ coefficients are kept. In cD₂ and cD₃ the clean speech carries 6.2 and 6.3 dB of energy, so most speech above 500 Hz is removed.
  - The untouched approximation cA₅ (0–125 Hz) holds noise at −5.3 dB against speech at −22.2 dB. It passes 17 dB more noise than speech.
  - This is correction 6's expected finding: λ = σ√(2 ln n_samples) with n_samples ≈ 2.4 × 10⁴ is about 4.5σ, which is too high for speech at these SNRs.
- **Why level-dependent is worse.** In speech-dominated bands, the MAD of the noisy coefficients measures speech rather than noise. On sp01 at 5 dB, λ for cD₄ is 0.168 vs 0.091 universal, and only 11.6 % of cD₄ is kept vs 21.3 %.
  - `sanity_checks.md` records the same effect across 30 sentences: at 15 dB, λⱼ is 1.25–3.0× the value the true noise would give.
- **Wavelet on babble.** The universal threshold is set from cD₁, where babble is weak, so it changes babble mixtures much less than white ones. Its babble ΔPESQ (−0.152) is therefore "better" than its white ΔPESQ (−0.350).
- **Method difference to state in the report.** The wavelet method never sees the noise-only lead-in (σ comes blindly from the MAD), while SS and Wiener estimate the noise from it.

## 7. Transient preservation (O5)

Source: `transient_metrics.csv`, `transient_closeup.png`.

**Click train** (20 identical 2 ms bursts, white noise at 10 dB):

| Signal | Concentration (±2.5 ms / ±25 ms) | Pre-echo (10 ms before onset, dB re clean click) | Core energy (dB re clean) |
|---|---|---|---|
| clean | 1.000 | −∞ | 0.0 |
| noisy | 0.957 | −20.1 | 0.0 |
| spectral subtraction | 0.979 | −21.2 | −3.0 |
| Wiener | 0.944 | −19.1 | −4.6 |
| wavelet, universal | 0.992 | −22.1 | −1.5 |
| wavelet, level-dependent | 0.990 | −21.8 | −1.6 |

- As theory predicts, the wavelet method keeps clicks the sharpest, with the least pre-echo and the least attenuation.
- Wiener's pre-echo is worse than the unprocessed input.
- The click-synchronous energy envelope (`transient_envelope.csv`, bottom row of the figure; mean dB relative to the clean click peak) shows two things:
  - **Pre-echo.** Both Fourier methods' residual rises from −44.6 dB (SS) and −46.8 dB (Wiener) at 25–15 ms before the onset to −33.1 and −33.5 dB in the last 10–2 ms. That's energy spread across the 32 ms frame. The wavelet output rises only from −44.6 to −41.4 dB.
  - **Post-click.** Over 5–30 ms after the onset, Wiener stays at −31.8 dB, within 2 dB of the unprocessed noise (−29.9 dB), because the decision-directed recursion keeps the gain open after a burst. SS is at −41.7 dB and the wavelet method at −44.0 dB.

**Real plosive:** the /k/ in "crack", sp13 at 2.257 s, found automatically as the largest energy jump (33.8 dB) after a quiet gap of at least 20 ms; white noise at 5 dB.

- Every method loses most of the burst: core energy is −13.4 dB for SS, −17.3 dB for Wiener and −9.7 dB for both wavelet variants.
- Pre-echo is −8.7 dB (SS), −15.9 dB (Wiener) and −6.9 dB (wavelet), against −31.0 dB for clean and +10.4 dB for noisy.
- Both wavelet variants zero **every** detail coefficient within ±25 ms of the onset (`wavelet_band_check.csv`, `pct_kept_near_onset` = 0). What remains is the 0–125 Hz approximation, so the two outputs are identical there.
- **Finding: the wavelet advantage holds for strong synthetic clicks but disappears for a real plosive at 5 dB, where all methods lose the burst.**

## 8. Fidelity vs perceptual metric disagreement (O4)

Sources: `metric_disagreement.csv`, `metric_disagreement_examples.csv`, `pesq_stability_check.csv`, `fidelity_vs_perceptual.png`.

- **Overall.** 5,421 of 17,700 sweep points (30.6 %) have ΔsegSNR > 0 but ΔPESQ < 0, and 9,441 (53.3 %) have ΔsegSNR > 0 but ΔSTOI < 0.
- **By method and noise** (ΔsegSNR > 0 with ΔPESQ < 0):

  | Method | White | Babble |
  |---|---|---|
  | SS | 356 / 4,500 | 1,704 / 4,500 |
  | Wiener | 6 / 750 | 340 / 750 |
  | Wavelet, universal | 833 / 1,800 | 842 / 1,800 |
  | Wavelet, level-dependent | 722 / 1,800 | 618 / 1,800 |

- **At the defaults** the disagreement is almost entirely a babble effect:
  - SS: 65 of 150 babble cases vs 2 of 150 white;
  - Wiener: 66 of 150 babble vs 1 of 150 white.
- **Largest-ΔsegSNR examples:**
  - sp25, white −5 dB, SS α = 3, β = 0.005: ΔsegSNR +8.10 dB but ΔPESQ −0.050.
  - sp21, white −5 dB, Wiener η = 0.99: +7.42 dB but ΔPESQ −0.089 and ΔSTOI −0.039.
  - sp03, babble −5 dB, level-dependent wavelet db8 L6: +6.45 dB but ΔPESQ −0.213 and ΔSTOI −0.188.
- **A second kind of disagreement: PESQ up, STOI down.** 24 points have ΔPESQ > +0.5 with ΔSTOI < −0.1.
  - 19 give stable PESQ scores, among them 11 SS points on white noise at 5–15 dB. That's a genuine disagreement: noise removed, speech envelope damaged.
  - 5 are **PESQ failures:** the score changes by more than 0.5 when the output is delayed by one sample.

    | Sentence | Condition | Method | PESQ → PESQ after 1-sample delay | STOI |
    |---|---|---|---|---|
    | sp09 | babble −5 dB | level-dependent db4 L6 | 4.40 → 1.04 | 0.19 |
    | sp18 | white −5 dB | level-dependent db4 L6 | 2.97 → 4.17 | 0.28 |
    | sp18 | white −5 dB | level-dependent db8 L5 | 3.09 → 1.71 | 0.25 |
    | sp24 | white −5 dB | level-dependent db8 L6 | 4.34 → 1.05 | 0.29 |
    | sp28 | white −5 dB | universal sym8 L6 | 4.25 → 1.04 | 0.32 |

    These outputs contain almost no speech in the 300–3400 Hz band that PESQ models.
  - Three more stable sp18 wavelet points are implausible too (PESQ 2.88–3.64 with STOI 0.16–0.28), so the one-sample test does not catch every failure.
  - All these rows are kept in `all_runs.csv`. They inflate the level-dependent wavelet's white −5 dB cell (PESQ 1.12 ± 0.39).

## 9. Babble vs white

Sources: `noise_estimate_check.csv`, `summary.csv`.

- **The lead-in noise estimate is unbiased on average for both noise types, but only white noise is stationary enough for it to hold:**

  | | Mean bias | Per-sentence bias range | Per-bin spread | Frame-energy fluctuation |
  |---|---|---|---|---|
  | White | −0.05 dB | −0.27 to +0.13 dB | 1.2 dB | 0.54 dB |
  | Babble | −0.90 dB | −5.71 to +3.49 dB | 4.0 dB | 4.69 dB |

- **Consequence:** the Fourier methods gain about 5–6 dB segSNR and 0.55–0.60 PESQ on white noise but under 1 dB and about 0 PESQ on babble, and they lower STOI on babble at every SNR (Section 2).
- **Babble-specific artefacts.** The spectrogram grid shows broadband vertical streaks: frames where the static estimate over- or under-subtracts as the babble level swings.
- **The natural next step** is a noise tracker such as minimum statistics (Martin 2001; correction 7).

## 10. Flags and caveats

Source: `sanity_checks.md` (40 PASS, 19 FLAG, each explained).

- **STOI below 0 in 2 of 17,700 rows** (−0.017 and −0.006): the level-dependent wavelet at L6 on sp25, babble −5 dB. STOI's range is [−1, 1].
- **"Improvements shrink in spread at 15 dB" does not hold** for several method × metric cells. It's a metric floor effect: at −5 dB, 45–48 % of noisy-input frames sit at the −10 dB segSNR clamp (vs 7–8 % at 15 dB), and noisy PESQ is 1.28 (floor ≈ 1.0) vs 2.10–2.36 at 15 dB.
- **The PESQ failures** in Section 8.
- **SNR is plain power over the whole sentence, not ITU-T P.56 active level,** so scores are not directly comparable with published NOIZEUS results (see `README.md`).
- **Denoiser runtime per 2–3 s utterance** (mean over 300 runs at the defaults, `timing_summary.csv`, wall clock, machine-dependent): SS 1.7 ms, Wiener 3.0 ms, wavelet universal 1.0 ms, wavelet level-dependent 1.2 ms.
