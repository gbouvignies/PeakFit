# 0001: σ-only uncertainties; reduced χ² is a diagnostic

**Date:** 2026-11-04 · **Status:** accepted

## Decision

Every primary uncertainty PeakFit publishes uses the supplied or estimated
statistical noise σ:
- nonlinear-parameter errors;
- conditional amplitude errors, σ√diag((ΦΦᵀ)⁻¹);
- MCMC intervals.

The automatic √max(1, χ²ᵣ) inflation of amplitude errors is removed, with no option to
restore it. Its fields (`scaled_amplitude_standard_errors`,
`amplitude_uncertainty_scale`) are deleted rather than kept beside the σ-only values.

Reduced χ² stays prominent as a goodness-of-fit diagnostic: in `clusters.csv`, in
`intensities.csv` beside `intensity_err`, in the JSON statistics, in the GUI
navigator, profile and review, and in MCMC output.

## Why

- **One convention, not three.** Nonlinear errors and MCMC already used σ alone, while
  amplitude errors were inflated. One table therefore mixed conventions, and MCMC
  intervals looked narrower than the fit's errors for no scientific reason.
- **The inflation treats χ²ᵣ > 1 as an unknown noise scale.** PeakFit's σ is
  supplied, or estimated from thousands of points, so χ²ᵣ > 1 more often reflects
  model misfit, locally varying noise or correlated noise. These affect the errors
  differently, and only an underestimated σ is corrected by a larger error. In
  synthetic tests:
  - the inflation corrected an underestimated σ;
  - under lineshape misfit, it overstated the scatter by 41% without representing
    the bias;
  - under correlated noise, it never responded.
- **The real case the inflation might have served is not served by it.** On the
  bundled CEST series, the errors understate the plane-to-plane scatter mainly
  because zero filling and apodization correlate neighbouring points.
  - That shortfall is about 2.7× with σ-only errors and was about 2× with the
    inflation.
  - It is unrelated to χ²ᵣ: the correlation is 0.18.
  - Its remedy is a noise model, not a χ²ᵣ factor.

Evidence: [validation/sigma-uncertainties.md](../validation/sigma-uncertainties.md).

## Consequences

- **Smaller errors.** `intensity_err` and the GUI error bars are smaller by √χ²ᵣ
  wherever χ²ᵣ > 1. On the CEST example, every peak's error shrinks: by a median of
  1/1.41, and up to 1/3.1.
- **Normalized ratios.** Their propagated σ shrinks too, so more references pass the
  σ_R/|R| ≤ 0.1 reliability limit.
- **Breaking formats.** Scientific archive 7.0.0, fit JSON 6.0.0, analysis project
  10.0.0. Earlier files are refused; refit, or re-export from a refit. Projects embed
  an archive and record the fit statistics, hence their bump. MCMC records (1.1.0)
  are unchanged: MCMC already used σ alone.
- **Correlated noise remains.** On zero-filled spectra, σ-only errors understate the
  actual scatter. Modelling the noise covariance, or estimating noise locally, is a
  separate decision.
