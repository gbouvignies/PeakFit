# 0002: Sandwich errors from the processing-derived noise correlation

**Date:** 2026-11-05 · **Status:** accepted, pending review of its scientific outputs
· **Refines:** [ADR 0001](0001-sigma-only-uncertainties.md)
· **Refined by:** [ADR 0003](0003-ols-and-profile-uncertainty.md)

## Decision

Zero filling and apodization correlate the noise of neighbouring spectral points.
Where a run's NMRPipe header records its processing, PeakFit derives that
correlation, R, and publishes the least-squares sandwich errors under σ²R. That
processing is a sine bell or no apodization, first-point scale, zero fill and
first-order phase, and the header must pass a consistency check.
- Nonlinear errors: J⁺(R ⊗ I)J⁺ᵀ.
- Conditional amplitude errors: σ²G⁻¹ΦRΦᵀG⁻¹.

Anything else publishes independent-point errors, with the run recorded as "noise
correlation not modelled" and the reason.

Unchanged:
- **The fit:** the least-squares estimates and fitted values, σ estimation, and the
  reduced-χ² definition.
- **Identifiability:** decided on the shapes and the Jacobian, as before.
- **MCMC:** the sampler keeps its independent-noise likelihood. Every MCMC output
  says its intervals are not calibrated for correlated noise, even when the
  convergence checks pass.

There is no measured-correlation fallback, no other noise model, and no
compatibility reader.

## Why

- **The errors were too small.** Independent-point errors understated the scatter of
  fitted values 1.8–3.6-fold for isolated and overlapping 2D peaks, with `sp1`,
  `sp2` and `no_apod` lineshapes, in PeakFit's own fitting engine. The sandwich
  errors were calibrated there ([validation](../validation/sandwich-errors.md)).
- **The model is exact given its assumptions.** The correlation follows in closed
  form from the processing. It matches the exact processing operator to 10⁻¹²,
  needs no fitted noise parameters, and changes no fitted value.
- **Keeping least squares avoids the fragile alternatives.** Generalized least
  squares would move the published estimates and weight the directions the
  apodization emptied of noise. A measured correlation is biased by residual signal.
  The specification gives the comparison ([correlated-noise.md](../correlated-noise.md)).

## Consequences

- **Larger published errors on processed data.** On the bundled example (sine bell,
  zero fill to about 2.7× on both axes), amplitude errors grow 2.04–2.40-fold
  (median 2.34).
  - 643 of 644 parameter errors grow 1.25–2.47-fold (median 1.90).
  - One shrinks 0.55-fold: a narrow ¹H linewidth (4.6 Hz). The sandwich can be
    smaller where a parameter's information lies in frequencies the apodization
    suppressed.
- **The golden reference was regenerated** for this, with fitted values,
  classifications, warnings and χ² bit-identical.
- **Breaking formats:** scientific archive 8.0.0, fit JSON 7.0.0 and analysis
  project 11.0.0, which record the noise model. The CSV files gain
  `noise_correlation`. Earlier files are refused; refit. The MCMC record is
  unchanged (1.1.0).
- **Remaining limits:**
  - The errors are local. For strongly overlapping or near-coalescent peaks the fit
    can separate them by more than they are, with smaller errors around that
    biased solution. Unwarned fits of a pair 0.4 points apart covered their
    positions only 0.65 · 0.82; neither the existing warnings nor amplitude
    correlation identify them. A resolution-based advisory flag is a separate
    investigation, with no threshold until one is shown useful.
  - The model describes the recorded processing, not t1 noise or other unrecorded
    effects.
  - Calibration on real spectra needs replicates (validation C6). MCMC calibration
    under correlated noise (C5) is future work.
