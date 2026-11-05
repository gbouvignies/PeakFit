# Resolution-based review flag for overlapping peaks

Study date: 2026-11-05. **Study only:** PeakFit gains no flag, warning or output
from it. **Conclusion:** no resolution-based measure is useful as an advisory
review flag ([Conclusion](#conclusion)). Tool: `tools/validate_resolution_flag.py`.
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

**A measure is useful as an advisory review flag only if, on the test half:**
1. it flags at least 80% of problematic fits;
2. it flags at most 5% of fits of calibrated configurations (plus 3 binomial SE);
3. after flagging, no configuration's unflagged, unwarned fits cover below 0.85
   at 95% (with at least 20 such fits);
4. pooled over the configurations that were miscalibrated, the unflagged,
   unwarned fits are calibrated.

Criteria 3 and 4 are the ones a reader relies on: the absence of the flag must
mean the published errors can be taken at face value. As context, the analysis
also reports a configuration oracle that flags every fit of a miscalibrated
configuration. Problematic fits in calibrated configurations are beyond any flag
that depends only on the configuration, so it bounds what a pure measure of
resolution can reach.

**Pilot:** before these criteria were committed, the tool was debugged on 48
`sp2` diagonal cells at 20 Hz (40 realizations each) and on the reference cell.
Those fits use the sweep's seeds, so they are a subset of it. The definitions and
criteria were written into the tool before the pilot ran and were not changed
after it. The configuration oracle, a context figure and not a criterion, was added
after the pilot.

## Results

72,150 fits; every one was usable. 31,495 (44%) carry the uncertainty warning; the
analysis concerns the other 40,655.

**Reproduction of the sandwich study.** The reference cell reproduces it exactly:
63 warned fits, and 87 unwarned ones covering 0.65 · 0.82, of which the 37 with
|ρ| < 0.9 cover 0.37 · 0.67.

### Where unwarned fits under-cover

Pre-registered labels, all lineshapes, starts, linewidths, ratios and geometries.
Each entry: miscalibrated / judged cells · problematic / unwarned fits · share of
fits warned.

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

- **Miscalibration is a regime of small separations.** 92 of 588 judged cells are
  miscalibrated: 87 at 0.45 apodized widths or less, plus the reference cell. From
  0.8 widths up, one judged cell is miscalibrated (σ = 0.1).
- **In those regimes most fits are fine.** At 0.2 widths and σ = 0.01, 29% of
  unwarned fits are problematic; the rest are not.
- **Problematic fits occur everywhere.** At one width or more, 3-SE misses run at
  1.0–1.7% for σ ≤ 0.03, near the 1% of exact Gaussian errors, but at 3–5% for
  σ = 0.1: the tails are heavier at low signal-to-noise, though 68% and 95%
  coverage hold.
  - Of the 3,874 problematic fits, 1,401 lie in calibrated cells, 1,530 in
    miscalibrated cells, and 943 in cells too often warned to judge.
  - The configuration oracle therefore catches only 39% of them (test half), with
    no false positives. That bounds what any per-configuration measure of
    resolution can reach, and puts criterion 1 out of reach by construction.
- **The fit both over- and under-separates.** For problematic fits of separated
  pairs, the fitted over true separation has quantiles 0.73, 0.97, 1.14, 1.77 and
  3.31 (10–90%); for unproblematic ones, 0.94–1.12.
- **Alternative starts reconverge.** In 243 of 40,655 unwarned fits a refit found a
  χ² lower by more than 10⁻³. The bias is the estimator's, not a local minimum.

### The candidate measures (pre-registered analysis)

Threshold from the development half; every rate on the test half. FP: share of
fits of calibrated cells flagged. Resolved FP: the same, at separations of at least
one width. After flagging: cells whose unflagged, unwarned fits cover below 0.85
at 95% (of those judged), and pooled coverage over the cells that were
miscalibrated.

| Measure | Threshold | AUC | Sensitivity | FP | Resolved FP | Flagged | Gross cells after | Pooled after |
|---|---|---:|---:|---:|---:|---:|---|---|
| No flag | — | — | — | — | — | — | 61/489 | 0.56 · 0.79 |
| Fitted separation / apodized width | ≤ 0.27 | 0.82 | 0.33 | 0.050 | 0.000 | 0.12 | 23/424 | 0.59 · 0.83 |
| … largest axis | ≤ 0.26 | 0.82 | 0.35 | 0.050 | 0.000 | 0.14 | 14/412 | 0.58 · 0.84 |
| Fitted shape overlap | ≥ 0.88 | 0.81 | 0.29 | 0.050 | 0.000 | 0.12 | 21/421 | 0.58 · 0.82 |
| Amplitude correlation | ≥ 0.86 | 0.82 | 0.30 | 0.050 | 0.000 | 0.12 | 21/422 | 0.58 · 0.82 |
| Scaled Jacobian condition (log₁₀) | ≥ 1.20 | 0.62 | 0.16 | 0.048 | 0.000 | 0.10 | 24/421 | 0.55 · 0.80 |
| **Separation significance** | ≤ 3.40 | **0.88** | **0.47** | 0.047 | 0.001 | 0.13 | 30/438 | 0.57 · 0.81 |
| Profile nonlinearity | ≥ 0.88 | 0.84 | 0.37 | 0.049 | 0.009 | 0.12 | 34/436 | 0.57 · 0.81 |
| Alternative-start disagreement | ≥ 0.001 | 0.80 | 0.23 | 0.051 | 0.017 | 0.09 | 48/457 | 0.55 · 0.79 |
| True separation (oracle) | ≤ 0.30 | 0.86 | 0.46 | 0.049 | 0.000 | 0.14 | 25/435 | 0.59 · 0.83 |
| True separation significance (oracle) | ≤ 3.16 | 0.87 | 0.47 | 0.050 | 0.000 | 0.13 | 40/452 | 0.57 · 0.80 |

**Verdict: no measure passes.** Every measure meets criterion 2 (5% false
positives), by construction of its threshold, and fails criteria 1, 3 and 4. So do
both oracles.

- **Separation significance is the most informative measure,** and as good as
  knowing the truth: its AUC (0.88) and sensitivity (0.47) match the oracle
  significance (0.87, 0.47).
- **The amplitude correlation, the fitted overlap and the fitted separation are one
  measure in three forms** (AUC 0.81–0.82), and below the significance.
- **The Jacobian condition carries little information** (AUC 0.62).
- **The alternative-start threshold is numerical noise.** The selected threshold is
  10⁻³ SE, because the refits return to the same optimum.
- **Flagging does not make the remaining fits trustworthy.** Pooled over the cells
  that were miscalibrated, the unflagged fits still cover 0.55–0.59 · 0.79–0.84,
  little better than without a flag (0.56 · 0.79).

### Label switching (post hoc)

The pre-registered labels count a fit as wrong when the start swapped the peaks.
With free amplitudes on every plane, the swapped solution is the same model, so
such a fit is wrong only about which name goes with which position. At jittered
starts below 0.3 widths, about half of the problematic fits are swaps. This
deviation from the pre-registration repeats the analysis with each fit compared
with the truth under the better labelling (smaller Σz²).

| | Pre-registered | Label-invariant |
|---|---:|---:|
| Problematic unwarned fits | 3,874 | 2,463 |
| Miscalibrated / calibrated cells | 92 / 496 | 33 / 555 |
| Configuration oracle sensitivity | 0.39 | 0.17 |
| Gross cells / pooled coverage, no flag | 61/489 · 0.56 · 0.79 | 16/489 · 0.58 · 0.82 |
| Separation significance: threshold · AUC · sensitivity · FP | ≤ 3.40 · 0.88 · 0.47 · 0.047 | ≤ 2.62 · 0.85 · 0.38 · 0.048 |
| … gross cells / pooled coverage after flagging | 30/438 · 0.57 · 0.81 | 5/450 · 0.57 · 0.85 |
| Best other AUC (profile nonlinearity) | 0.84 | 0.81 |

No measure passes here either. Without the swaps, the fitted separation, overlap
and amplitude correlation drop to AUC 0.71–0.72: much of their apparent skill was
detecting swaps, which happen when the pair is barely separated.

### Operating points beyond 5% (exploratory)

A flag might accept more false positives. Separation significance, all
realizations, label-invariant:

| Flag when significance ≤ | FP | Sensitivity | Unwarned fits flagged | Gross cells | Pooled, previously miscalibrated |
|---:|---:|---:|---:|---|---|
| 2 | 0.033 | 0.29 | 0.07 | 24/541 | 0.57 · 0.83 |
| 3 | 0.058 | 0.46 | 0.12 | 11/511 | 0.57 · 0.84 |
| 4 | 0.082 | 0.58 | 0.15 | 5/486 | 0.55 · 0.86 |
| 8 | 0.177 | 0.79 | 0.26 | 1/423 | 0.59 · 0.90 |
| 16 | 0.319 | 0.87 | 0.39 | 0/338 | — |

- Gross under-coverage disappears only by flagging a quarter or more of all
  unwarned fits, including 18% or more of the fits of calibrated cells. The falling
  count of judged cells shows that part of it is cells losing too many fits to be
  judged.
- Wherever they can still be judged, the unflagged fits of formerly miscalibrated
  cells stay below 68% coverage (0.55–0.59).
- 18% of problematic fits (20% pre-registered) have a separation significance of
  10 or more: confidently separated and still wrong. No threshold that spares
  calibrated fits reaches them.
- Profile nonlinearity and fitted separation behave alike or worse (`explore`
  output). The closest to calibration is profile nonlinearity ≥ 0.2, at
  0.67 · 0.94, which flags 42% of unwarned fits and 35% of those of calibrated
  cells.

## Conclusion

**No resolution-based measure is useful as an advisory review flag.** The
recommendation is to add none.

- **Direct evidence:** none of eight candidate measures, nor either oracle,
  satisfies the pre-registered criteria, in the pre-registered or the
  label-invariant analysis. The best, separation significance, flags 38–47% of
  problematic fits at 5% false positives. The fits it leaves unflagged in bad
  regimes still cover only 0.57 at 68%.
- **Interpretation:** resolution identifies the regime, not the fit. Within a
  miscalibrated regime most fits are sound, and neither the fitted separation, the
  overlap nor the local curvature tells the biased ones from them reliably.
  Knowing the true resolution does no better than estimating it.
- **What this does not show:** that no per-fit diagnostic can work. Measures that
  use more than the local fit (for example posterior sampling, or replicate
  spectra) were not tested.
- **Separation significance as a descriptive review measure** would be a separate
  decision. It would have to say that a high value does not certify a fit: a fifth
  of the problematic fits have a significance of 10 or more.
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
committed. `analyze` writes both analyses to the results file; `explore` prints the
exploratory tables. Every realization is seeded from its cell and index, so a rerun
reproduces the record exactly.
