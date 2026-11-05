# Resolution-based review flag for overlapping peaks

Study date: 2026-11-05. **Study only:** PeakFit gains no flag, warning or output
from it. Tool: `tools/validate_resolution_flag.py`. Results:
[`resolution-flag.json`](resolution-flag.json).

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
configuration. Problematic fits in calibrated configurations are beyond any flag,
so it bounds the sensitivity any measure can reach.

**Pilot:** before these criteria were committed, the tool was debugged on 48
`sp2` diagonal cells at 20 Hz (40 realizations each) and on the reference cell.
Those fits use the sweep's seeds, so they are a subset of it. The definitions and
criteria were written into the tool before the pilot ran and were not changed
after it.

## Results

To be added from the recorded run.

## Reproduce

```bash
uv run python tools/validate_resolution_flag.py fit --output /tmp/resolution-fits.json.gz
```

```bash
uv run python tools/validate_resolution_flag.py analyze /tmp/resolution-fits.json.gz --output docs/validation/resolution-flag.json
```
