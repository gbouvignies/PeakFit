# Sandwich errors with PeakFit's fitting engine

Validation date: 2026-11-04. This is the gate of step 2 in
[correlated-noise.md](../correlated-noise.md) §11: it must pass before sandwich errors
are published. It uses `tools/validate_sandwich_errors.py`, with results in
[`sandwich-errors.json`](sandwich-errors.json).

## Set-up

- **Clusters:** 2D pseudo-3D clusters over four planes, drawn with PeakFit's own `sp2`
  lineshapes in both dimensions. Peak heights are 0.5–1 and σ = 0.02.
- **Noise:** from the exact, separable processing operators an NMRPipe header
  describes:
  - ¹⁵N: 32 → 128 points, sine bell² (off 0.4), P1 180°;
  - ¹H: 64 → 256 points, sine bell² (off 0.3), first point 0.5.
- **Correlation:** `read_noise_model` reads that header, and `cluster_correlation`
  gives R over each 11 × 13 fitting window.
- **Fits:** each realization is fitted by `run_pipeline`, starting at the truth.
- **Nonlinear errors:** σ-only errors (identical to those published in every
  realization) and sandwich errors, both with the bound-crossing rule.
- **Conditional amplitude errors:** amplitudes solved at the true lineshape, so the
  comparison is of the conditional error itself.

**Pass criteria, fixed in advance:**
- For the isolated and overlapping cases, sandwich coverage within 3 binomial SE of
  68% and 95%, and scatter / SD within 0.9–1.1 for every quantity.
- Identical covariance withholding in every case.
- R = I reproduces σ-only to 10⁻¹⁰.

## Results

Scatter / median stated SD is given per quantity (range over the parameters, or
over the peaks and planes); coverage is pooled through z.

| Case (realizations) | Quantity | σ-only scatter/SD · 68% · 95% | Sandwich scatter/SD · 68% · 95% |
|---|---|---|---|
| Isolated peak (400) | positions | 2.86–3.56 · 0.25 · 0.45 | 1.01–1.08 · 0.65 · 0.95 |
| | linewidths | 2.80–3.01 · 0.27 · 0.48 | 1.02–1.04 · 0.65 · 0.95 |
| | conditional amplitudes | 3.40–3.53 · 0.22 · 0.43 | 0.97–1.01 · 0.69 · 0.96 |
| Overlapping pair, 2 points apart in each dimension (400) | positions | 2.43–2.57 · 0.30 · 0.56 | 1.00–1.03 · 0.67 · 0.95 |
| | linewidths | 2.41–2.86 · 0.30 · 0.54 | 0.94–1.02 · 0.68 · 0.95 |
| | conditional amplitudes | 3.09–3.53 · 0.24 · 0.44 | 0.92–1.05 · 0.67 · 0.95 |
| Near-coalescent pair, 0.4 points apart (150) | positions | 3.24–5.94 · 0.24 · 0.44 | 1.22–2.28 · 0.55 · 0.76 |
| | linewidths | 3.91–6.06 · 0.30 · 0.62 | 1.41–2.19 · 0.77 · 0.98 |
| | conditional amplitudes | 2.85–3.11 · 0.26 · 0.49 | 0.91–0.99 · 0.71 · 0.96 |
| White-noise control, R = I (150) | all | identical | identical, to 1.3 × 10⁻¹⁵ |

**Withholding:**
- The covariance gate was identical under both conventions in every realization of
  every case. For duplicated peaks, both withheld the nonlinear covariance in 133 of
  150 realizations, and the amplitude covariance in 132.
- The bound-crossing rule applies to the published SE. The larger sandwich SE
  crosses a bound more often: in the near-coalescent case, 98 parameter errors were
  withheld against 51.

**Verdict:** every pre-registered criterion passes.

## Reading

- **Separated and overlapping peaks:** sandwich errors are calibrated for positions,
  linewidths and conditional amplitudes. σ-only errors understate the scatter
  2.4–3.6-fold with this processing.
- **Near-coalescent peaks:**
  - Conditional amplitude errors stay calibrated.
  - The nonlinear sandwich under-covers (positions 0.55 · 0.76), because the
    linearization it rests on fails there. It is better than σ-only (0.24 · 0.44),
    but not calibrated.
  - The total spread of the fitted amplitudes is about 10³ × the conditional SD, as
    the two amplitudes trade off. Conditional errors say nothing about total
    uncertainty in this regime.
- **Total against conditional:** with a 2D lineshape fitted over four planes, the
  jointly fitted amplitudes scatter 1.4 × (isolated) and 2.5 × (overlapping) the
  conditional sandwich SD. That is the lineshape uncertainty conditional errors
  exclude by definition.

## Corrections made before the verdict

The first run reported two criteria as failing. Both were errors in the tool's
measurements; the criteria themselves were unchanged:
- **Positions:** scatter / SD pooled ¹⁵N and ¹H positions, whose ppm scales differ
  about eightfold. It is now computed for each parameter. Per parameter it was
  1.01–1.08 in that first run too, with no outliers (largest |z| 3.2).
- **Withholding:** the first run compared per-parameter NaNs, which include the
  bound-crossing rule. That rule is meant to act on the published, larger SE. The
  criterion concerns the covariance gate, which is now compared directly; bound-rule
  counts are reported separately.

## Scope

- 2D `sp2` clusters on synthetic spectra with the stated processing. Other
  apodizations are not modelled (§2.3).
- The fits start at the truth, so this tests local errors and does not search for
  other optima.
- Real-data calibration needs replicates (C6), and MCMC is unchanged.

```bash
uv run python tools/validate_sandwich_errors.py --output /tmp/sandwich-errors.json
```
