# Resolution-based review flag for overlapping peaks

Study date: 2026-11-05. **Study only:** PeakFit gains no flag, warning or output
from it. **Conclusion:** none of the eight measures tested meets the
pre-registered criteria for an advisory review flag. Separation significance
discriminates risk well (AUC 0.90) but cannot certify that an unflagged fit is
reliable ([Conclusion](#conclusion)). Tool: `tools/validate_resolution_flag.py`.
Results: [`resolution-flag.json`](resolution-flag.json).

## Question

The [sandwich-error validation](sandwich-errors.md) found that, for a pair 0.4
points apart, 87 of 150 fits carry no uncertainty warning yet cover their true
positions only 0.65 (68%) and 0.82 (95%). Those fits are biased: the fit separates
the peaks by more than they are, and the local errors shrink around that solution.
Amplitude correlation does not identify them.

Can any measure of resolution, computed from a completed fit, flag such fits as a
separate, named, advisory review flag, without flagging well-resolved or
calibrated fits? Such a flag would be a review heuristic: it would change no fitted
value, published error or warning, and it would be separate from the covariance
withholding rule.

## Set-up

Each realization is a 2D pseudo-3D pair (four planes) drawn with PeakFit's own
lineshapes. Its noise comes from the exact processing operators of
`tools/validate_sandwich_errors.py`, and its NMRPipe header records that
processing. `run_pipeline` fits it with the noise model that `read_noise_model`
reads from the header, so the errors are the published sandwich errors.

- **Width unit:** separations are given in *apodized widths*: the FWHM, in points,
  of the processed lineshape along each axis. The linewidth parameter is the
  unapodized FWHM; for these short acquisitions the apodized width is dominated by
  the window. At 20 Hz, `sp2` is 4.31 points wide in ¹⁵N and 3.75 in ¹H.
- **Sweep:** each cell has 80 realizations.

  | Factor | Levels |
  |---|---|
  | Separation (Euclidean, in apodized widths) | 0 (coincident), 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.5 |
  | Geometry | along ¹H, along ¹⁵N, diagonal (equal in widths on both axes) |
  | Linewidth, both axes | 10, 20, 40 Hz (`sp2`); 20 Hz otherwise |
  | Height of the second peak over the first | 1, 0.3 |
  | Noise σ (larger peak height 1) | 0.01, 0.03, 0.1 |
  | Lineshape and processing | `sp2`; `sp1` and `no_apod` at 20 Hz |
  | Start | jittered peak list; the truth for `sp2` at 20 Hz |

  - **Peak heights over the planes:** (1, 0.8, 0.6, 0.9) and the second peak's
    (0.7, 0.9, 1, 0.5) times the ratio.
  - **Jittered peak-list start:** each position is moved by a normal error of SD 0.2
    apodized widths per axis, and every linewidth starts at PeakFit's default
    20 Hz. Position bounds centre on the start, as they do for a peak list.
  - **Window:** a rectangle of half-width ⌈|Δ|/2 + 1.25 apodized widths⌉ points per
    axis around the pair's midpoint.
  - **Reference cell:** the near-coalescent pair of the sandwich study, with its seeds
    and window (150 realizations), included to tie the two studies together.
- **Size:** 901 cells, 72,150 realizations.

## Candidate measures

Every measure is computed from the completed fit alone, as a user would have it.

| Measure | Definition | Flags when |
|---|---|---|
| Fitted separation / apodized width | √Σᵢ(Δᵢ/wᵢ)², with Δᵢ the fitted separation on axis i and wᵢ the mean apodized width of the two fitted shapes | small |
| … largest axis | maxᵢ \|Δᵢ\|/wᵢ | small |
| Fitted shape overlap | cosine between the two fitted 2D shapes, over a window holding both | large |
| Amplitude correlation | \|ρ\| between the two amplitudes, from the published covariance | large |
| Scaled Jacobian condition | log₁₀ of the condition number of the column-scaled Jacobian (the quantity the covariance gate tests) | large |
| Separation significance | √(Δᵀ Σ_Δ⁻¹ Δ), with Σ_Δ the published (sandwich) covariance of the separation vector | small |
| Profile nonlinearity | the separation is fixed at its published 95% interval edges, (1 ∓ 1.96/z)Δ̂, the other parameters refitted, and the χ² rise compared with the local quadratic prediction: max \|ln(actual/predicted)\| | large |
| Alternative-start disagreement | refits from the pair collapsed to ¼ and spread to 2× the fitted separation; the largest position shift, in published SEs, to a refit no worse by Δχ² > 1 | large |

Two oracle measures use the truth, which a user does not have: the true
separation in apodized widths, and the true separation significance (separation
over its sandwich SE at the truth, noise-free). They test whether resolution
itself predicts the problem, so that a fitted estimate of it could.

The profile is computed in the independent-point χ² metric, where both the actual
and the predicted rise are defined. The step uses the published sandwich SE, so
the check covers the region the published interval claims.

## Pre-registered definitions and criteria

Fixed in the tool before the sweep ran.

- **Problematic fit:** an unwarned fit that misses the true value of at least one
  position by more than 3 published SEs. Under calibration that happens to about
  1% of fits (four positions at 0.27% each).
- **Biased fit (secondary):** an unwarned fit with a position error above 3 times
  the sandwich SE at the truth, noise-free: the precision the experiment supports.
  It is undefined for coincident peaks, whose covariance at the truth is withheld.
- **Calibrated configuration:** a cell whose unwarned fits cover their positions,
  pooled, no worse than 3 binomial SE below 68% and 95%. n is the number of fits,
  not of positions, which is conservative. Over-coverage is not a failure. Cells
  with fewer than 20 unwarned fits are not judged.
- **False positive:** an unwarned fit in a calibrated configuration that is flagged.
  Also reported: flagged fits that are not problematic, and flagged fits of
  calibrated configurations at a separation of at least one apodized width.
- **Threshold:** chosen on the even-numbered realizations (development half) as the
  one of most sensitivity with at most 5% false positives. Every rate below is on
  the odd-numbered half (test half), with that threshold.

**A measure qualifies as an advisory review flag only if, on the test half:**
1. it flags at least 80% of problematic fits;
2. it flags at most 5% of fits of calibrated configurations (plus 3 binomial SE);
3. after flagging, no configuration's unflagged, unwarned fits cover below 0.85
   at 95% (with at least 20 such fits);
4. pooled over the configurations that were miscalibrated, the unflagged,
   unwarned fits are calibrated.

Criteria 3 and 4 are the ones a reader relies on: the absence of the flag must
mean the published errors can be taken at face value. As context, the analysis
also reports a whole-configuration flag, which flags every fit of a miscalibrated
configuration (see [Expected outliers and systematic failures](#expected-outliers-and-systematic-failures)).

**Pilot:** before these criteria were committed, the tool was debugged on 48
`sp2` diagonal cells at 20 Hz (40 realizations each) and on the reference cell.
Those fits use the sweep's seeds, so they are a subset of it. The definitions and
criteria were written into the tool before the pilot ran and were not changed
after it. The whole-configuration flag, a context figure and not a criterion, was
added after the pilot.

**Corrections after review.** Two parts of the first analysis did not match the
pre-registration's intent, and are corrected. The stored fit records are
reanalysed; nothing is refitted.
- **Held-out labels:** the first analysis labelled configurations calibrated or
  miscalibrated from all realizations, so the test half had helped decide its own
  labels. Configurations are now labelled from the development half alone. That
  half has 40 realizations per cell, so fewer cells can be judged and fewer
  register as miscalibrated: 45 instead of 92.
- **ROC:** the first AUC compared problematic fits with every fit of a calibrated
  configuration, including the problematic ones, so the two groups overlapped. The
  negatives are now the unproblematic fits of calibrated configurations, and the
  AUC is computed on the test half. The AUC against every unproblematic fit is
  also given.

The first analysis reproduces with the tool at commit `3938f36`.

## Results

72,150 fits; every one was usable. 31,495 (44%) carry the uncertainty warning; the
analysis concerns the other 40,655.

**Reproduction of the sandwich study.** The reference cell reproduces it exactly:
63 warned fits, and 87 unwarned ones covering 0.65 · 0.82, of which the 37 with
|ρ| < 0.9 cover 0.37 · 0.67.

### Where unwarned fits under-cover

Descriptive, over all realizations and the pre-registered labels. The rows pool all
lineshapes, starts, linewidths, ratios and geometries. Each entry gives
miscalibrated / judged cells · problematic / unwarned fits · share of fits warned.

| Separation (apodized widths) | σ = 0.01 | σ = 0.03 | σ = 0.1 |
|---|---|---|---|
| 0 (coincident) | 0/0 · 10/92 · 0.90 | 0/0 · 7/88 · 0.91 | 0/0 · 9/57 · 0.94 |
| 0.1 | 12/26 · 419/1203 · 0.58 | 2/4 · 213/480 · 0.83 | 0/0 · 51/223 · 0.92 |
| 0.2 | 22/35 · 599/2078 · 0.28 | 9/17 · 310/891 · 0.69 | 0/0 · 108/244 · 0.92 |
| 0.3 | 15/36 · 434/2539 · 0.12 | 16/27 · 341/1342 · 0.53 | 1/1 · 135/284 · 0.90 |
| 0.45 | 2/36 · 185/2728 · 0.05 | 4/33 · 215/1887 · 0.34 | 4/6 · 145/418 · 0.85 |
| 0.6 | 0/36 · 60/2808 · 0.03 | 1/34 · 103/2238 · 0.22 | 2/15 · 125/685 · 0.76 |
| 0.8 | 0/36 · 24/2852 · 0.01 | 0/35 · 45/2491 · 0.14 | 1/18 · 68/971 · 0.66 |
| 1.0 | 0/36 · 28/2868 · 0.00 | 0/36 · 31/2610 · 0.09 | 0/20 · 60/1227 · 0.57 |
| 1.5 | 0/36 · 30/2880 · 0.00 | 0/36 · 46/2727 · 0.05 | 0/28 · 51/1657 · 0.42 |

- **Miscalibration is a regime of small separations.** Over all realizations, 92 of
  588 judged cells are miscalibrated: 87 at 0.45 apodized widths or less, plus the
  reference cell. From 0.8 widths up, one judged cell is miscalibrated (σ = 0.1).
- **In those regimes most fits are fine.** At 0.2 widths and σ = 0.01, 29% of
  unwarned fits are problematic; the rest are not.
- **The fit both over- and under-separates.** For problematic fits of separated
  pairs, the fitted over true separation has quantiles 0.73, 0.97, 1.14, 1.77 and
  3.31 (10–90%). For unproblematic fits the range is 0.94–1.12.
- **Alternative starts reconverge.** In 243 of 40,655 unwarned fits a refit found a
  χ² lower by more than 10⁻³. The bias is the estimator's, not a local minimum.

### Expected outliers and systematic failures

A problematic fit is either a statistical outlier that exact calibration produces
at its nominal rate, or a systematic failure. Under exact calibration a fit misses
one of its four positions by more than 3 SE with probability between 0.27% (the
positions perfectly correlated) and 1.08% (independent). The excess over the upper
bound is therefore a lower bound on the systematic failures.

Test half, configurations labelled on the development half:

| Configurations | Unwarned fits | Problematic | Expected by chance | Systematic, at least |
|---|---:|---:|---:|---:|
| Calibrated | 15,870 | 683 | 43–171 | 512 |
| Miscalibrated | 1,544 | 446 | 4–17 | 429 |
| Not judged (fewer than 20 unwarned fits) | 2,915 | 802 | 8–31 | 771 |
| All | 20,329 | 1,931 | 55–219 | 1,712 |

- **The direct baseline matches the chance rate.** At one width or more and
  σ ≤ 0.03, 68 of 5,555 test fits (1.2%) are problematic, close to the 1.08% of
  independent positions. At σ = 0.1 it is 57 of 1,481 (3.8%): the tails are heavier
  at low signal-to-noise, even where 68% and 95% coverage hold.
- **At least 88% of problematic fits are systematic** (1,712 of 1,931). Most of them
  lie outside the configurations labelled miscalibrated.

By mechanism, in this order (test half):

| Mechanism | Pre-registered labels | Label-invariant |
|---|---:|---:|
| Coincident truth | 15 | 15 |
| Peaks swapped by the start | 1,028 | — |
| Over-separated (fitted > 1.5 × true separation) | 287 | 475 |
| Under-separated (fitted < ⅔ × true) | 67 | 102 |
| Separation within ⅔–1.5 × true | 534 | 664 |

The last row holds all of the 55–219 expected chance outliers, plus heavy-tailed
misses.

**The whole-configuration flag** flags every fit of a configuration labelled
miscalibrated. On the test half it catches 23% of problematic fits (9%
label-invariant), with no false positives.
- It bounds what flagging whole configurations can achieve: no measure that is a
  function of the configuration alone can catch the problematic fits of
  calibrated configurations.
- It does not bound a per-fit diagnostic, which can flag individual fits anywhere.
  Most measures tested here exceed its sensitivity, at 5% false positives.
- What makes criterion 1 hard is the mix above: chance outliers that no
  diagnostic should flag, and systematic failures spread through calibrated cells.

### The candidate measures (pre-registered analysis)

The threshold comes from the development half; every rate is on the test half.
- **AUC:** problematic fits against the unproblematic fits of calibrated
  configurations; in brackets, against every unproblematic fit.
- **FP:** the share of fits of calibrated configurations flagged.
- **Resolved FP:** the same, at separations of at least one width.
- **After flagging:** cells whose unflagged, unwarned fits cover below 0.85 at 95%
  (of those judged), and pooled coverage over the cells labelled miscalibrated.

| Measure | Threshold | AUC | Sensitivity | FP | Resolved FP | Flagged | Gross cells after | Pooled after |
|---|---|---|---:|---:|---:|---:|---|---|
| No flag | — | — | — | — | — | — | 61/489 | 0.54 · 0.78 |
| Fitted separation / apodized width | ≤ 0.28 | 0.82 (0.76) | 0.35 | 0.049 | 0.000 | 0.13 | 23/422 | 0.55 · 0.79 |
| … largest axis | ≤ 0.26 | 0.82 (0.76) | 0.36 | 0.049 | 0.000 | 0.14 | 13/411 | 0.58 · 0.83 |
| Fitted shape overlap | ≥ 0.88 | 0.82 (0.75) | 0.31 | 0.049 | 0.000 | 0.12 | 20/417 | 0.53 · 0.77 |
| Amplitude correlation | ≥ 0.85 | 0.82 (0.76) | 0.32 | 0.049 | 0.000 | 0.13 | 20/419 | 0.53 · 0.78 |
| Scaled Jacobian condition (log₁₀) | ≥ 1.19 | 0.62 (0.58) | 0.16 | 0.046 | 0.000 | 0.11 | 22/418 | 0.50 · 0.75 |
| **Separation significance** | ≤ 4.16 | **0.90 (0.83)** | **0.54** | 0.047 | 0.000 | 0.15 | 26/426 | 0.56 · 0.79 |
| Profile nonlinearity | ≥ 0.71 | 0.86 (0.79) | 0.46 | 0.048 | 0.007 | 0.15 | 29/425 | 0.55 · 0.79 |
| Alternative-start disagreement | ≥ 0.001 | 0.82 (0.76) | 0.26 | 0.045 | 0.011 | 0.10 | 46/450 | 0.54 · 0.78 |
| True separation (oracle) | ≤ 0.30 | 0.87 (0.81) | 0.48 | 0.047 | 0.000 | 0.15 | 24/431 | 0.56 · 0.80 |
| True separation significance (oracle) | ≤ 4.30 | 0.89 (0.82) | 0.55 | 0.048 | 0.000 | 0.16 | 31/436 | 0.57 · 0.79 |

**Verdict: no measure meets all four criteria.** Every measure meets criterion 2
(5% false positives), by construction of its threshold, and fails criteria 1, 3
and 4. So do both oracles.

- **Separation significance discriminates risk well.** A problematic fit has a
  lower significance than an unproblematic fit of a calibrated configuration 90%
  of the time. At 5% false positives it flags 54% of problematic fits, as well as
  an oracle that knows the true significance (0.89, 0.55).
- **It cannot certify reliability.** The fits it leaves unflagged in miscalibrated
  configurations still cover 0.56 · 0.79, as without a flag (0.54 · 0.78).
- **The amplitude correlation, the fitted overlap and the fitted separation are one
  measure in three forms** (AUC 0.82), below the significance.
- **The Jacobian condition carries little information** (AUC 0.62).
- **The alternative-start threshold is numerical noise.** The selected threshold is
  10⁻³ SE, because the refits return to the same optimum.

**What the corrections changed** (pre-registered labels):

| | First analysis | Corrected |
|---|---:|---:|
| Configurations labelled miscalibrated | 92 (all realizations) | 45 (development half) |
| Separation significance: threshold · AUC · sensitivity | ≤ 3.40 · 0.88 · 0.47 | ≤ 4.16 · 0.90 · 0.54 |
| Profile nonlinearity: AUC · sensitivity | 0.84 · 0.37 | 0.86 · 0.46 |
| Amplitude correlation: AUC · sensitivity | 0.82 · 0.30 | 0.82 · 0.32 |
| Pooled coverage after flagging by separation significance | 0.57 · 0.81 | 0.56 · 0.79 |
| No flag: gross cells · pooled coverage | 61/489 · 0.56 · 0.79 | 61/489 · 0.54 · 0.78 |
| Whole-configuration flag, sensitivity | 0.39 | 0.23 |

Most AUCs rise by 0.01–0.02, and the held-out labels move the thresholds. No
verdict changes.

### Label switching (post hoc)

This analysis is post hoc, not pre-registered. The pre-registered labels count a
fit as wrong when the start swapped the peaks. With free amplitudes on every
plane, the swapped solution is the same model, so such a fit is wrong only about
which name goes with which position. Here each fit is compared with the truth
under the better labelling (smaller Σz²).

| | Pre-registered | Label-invariant |
|---|---:|---:|
| Problematic unwarned fits (all realizations) | 3,874 | 2,463 |
| Configurations labelled miscalibrated (development half) | 45 | 13 |
| Whole-configuration flag, sensitivity | 0.23 | 0.09 |
| No flag: gross cells · pooled coverage | 61/489 · 0.54 · 0.78 | 16/489 · 0.57 · 0.80 |
| Separation significance: threshold · AUC · sensitivity · FP | ≤ 4.16 · 0.90 · 0.54 · 0.047 | ≤ 3.60 · 0.88 · 0.53 · 0.046 |
| … gross cells · pooled coverage after flagging | 26/426 · 0.56 · 0.79 | 3/435 · 0.59 · 0.91 (24 fits) |
| Profile nonlinearity: AUC · sensitivity | 0.86 · 0.46 | 0.84 · 0.41 |
| Fitted separation, overlap, amplitude correlation: AUC | 0.82 | 0.73–0.74 |

- **No measure meets all four criteria here either.**
- Separation significance, profile nonlinearity and the largest-axis separation now
  meet criterion 4. That rests on the 24–32 unflagged fits that remain in the 13
  miscalibrated configurations: weak evidence. All three still fail criteria 1
  and 3.
- Without the swaps, the fitted separation, overlap and amplitude correlation drop
  to AUC 0.73–0.74: much of their apparent skill was detecting swaps.

### Operating points beyond 5% (exploratory)

A flag might accept more false positives. Separation significance, test half,
development-half labels, label-invariant:

| Flag when significance ≤ | FP | Sensitivity | Unwarned fits flagged | Gross cells | Pooled, labelled miscalibrated |
|---:|---:|---:|---:|---|---|
| 2 | 0.018 | 0.28 | 0.07 | 10/461 | 0.57 · 0.80 |
| 3 | 0.036 | 0.45 | 0.12 | 3/442 | 0.59 · 0.91 |
| 4 | 0.055 | 0.58 | 0.15 | 2/429 | 0.57 · 0.89 |
| 8 | 0.146 | 0.79 | 0.25 | 1/383 | — |
| 16 | 0.286 | 0.87 | 0.39 | 0/309 | — |

- Under the pre-registered labels the same thresholds leave 43, 33, 27, 13 and 6
  gross cells. Pooled coverage stays 0.55–0.63 · 0.78–0.83.
- Gross under-coverage disappears only by flagging a quarter or more of unwarned
  fits. The falling count of judged cells shows that part of it is cells losing
  too many fits to be judged.
- 18% of problematic fits (20% pre-registered, all realizations) have a separation
  significance of 10 or more: confidently separated and still wrong. No threshold
  that spares calibrated fits reaches them.

## Conclusion

**None of the tested measures meets the pre-registered criteria for a flag that
both identifies unreliable fits and leaves the remaining fits trustworthy.** The
recommendation is to add no resolution-based flag now. This does not show that no
resolution-based advisory diagnostic could be useful.

- **Direct evidence:** no measure meets all four criteria, nor does either oracle,
  in the pre-registered or the post-hoc label-invariant analysis. Under the
  pre-registered labels, every measure leaves the unflagged fits of miscalibrated
  configurations at 0.50–0.58 coverage at 68%.
- **Risk discrimination is real.** Separation significance ranks problematic fits
  below unproblematic ones 90% of the time (83% against every unproblematic fit).
  At 5% false positives it flags about half of them, as well as knowing the true
  significance.
- **Certification is not.** A fit without the flag is not thereby reliable:
  - the unflagged fits of miscalibrated configurations still under-cover;
  - a fifth of the problematic fits have a significance of 10 or more.
- **The absence of a warning is not proof of reliable uncertainties.** At least
  1,712 of the 1,931 problematic test fits are systematic failures, not chance
  outliers. Most lie outside the configurations labelled miscalibrated: 683 in
  configurations whose 68% and 95% coverage pass.
- **Interpretation:** resolution identifies the regime and part of the risk, not
  the fit. Knowing the true resolution does no better than estimating it.
- **Not tested:** diagnostics that use more than the local fit, such as posterior
  sampling or replicate spectra.
- **Unchanged:** the published values and errors, the uncertainty warning and the
  covariance withholding rule. The caveat in [outputs.md](../outputs.md) on
  strongly overlapping or near-coalescent peaks stays as written.

## Scope

- Synthetic pairs only: no neighbours, a rectangular window, four planes, the
  phase fixed, and the processing chains of the sandwich study.
- The jittered start is one model of a peak list (SD 0.2 widths, linewidths at the
  20 Hz default). Starts from real peak pickers are not modelled.
- The 3-SE miss is one definition of a problematic fit; the coverage criteria (3
  and 4) do not depend on it.

## Reproduce

```bash
uv run python tools/validate_resolution_flag.py fit --output /tmp/resolution-fits.json.gz
```

```bash
uv run python tools/validate_resolution_flag.py analyze /tmp/resolution-fits.json.gz --output docs/validation/resolution-flag.json
```

```bash
uv run python tools/validate_resolution_flag.py explore /tmp/resolution-fits.json.gz
```

The fit takes about 5 minutes on 16 workers and writes a 32 MB record that is not
committed. `analyze` reanalyses that record: it writes the pre-registered and the
post-hoc label-invariant analyses to the results file. `explore` prints the
exploratory tables. Every realization is seeded from its cell and index, so a rerun
reproduces the record exactly.
