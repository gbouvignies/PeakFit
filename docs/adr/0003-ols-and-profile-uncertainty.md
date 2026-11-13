# 0003: OLS on replicate evidence; three amplitude uncertainties

**Date:** 2026-11-13 · **Status:** accepted; implementation pending
· **Refines:** [ADR 0002](0002-processing-derived-sandwich-errors.md)

## Decision

**Estimator.** Ordinary least squares with sandwich errors under the
processing-derived noise correlation stays. Generalized least squares (GLS) and
shrinkage towards it are rejected.

**Reported quantities.** For one peak, write Σ for the first-order covariance of
its amplitudes over the planes, c for its conditional variance, L = Σ − cI for the
lineshape part, and P = I − uuᵀ with u the unit profile. PeakFit reports, each
under its own name:
- the **marginal σ**, √Σ_ss: the uncertainty of one absolute amplitude;
- the **relative-profile covariance**, Σ_rel = cI + PLP: exactly equivalent to Σ
  for every downstream analysis that fits a free scale;
- the **profile σ**, √(Σ_rel)_ss: the per-plane error of the profile, published as
  `intensity_err`. For an isolated peak it equals the conditional σ.

A **correlated-profile flag** marks peaks for which independent per-plane errors
misstate the downstream uncertainty: the largest correlated mode of Σ_rel exceeds
τ·c, or the amplitudes correlate strongly with a cluster partner. The thresholds
come from validation T3. The flag states a limit of the approximation, not a
judgment on the data.

Ratios and R2eff use the covariance between planes. No "common scale uncertainty"
is reported: its split from Σ is not unique.

The specification is [profile-uncertainty.md](../profile-uncertainty.md).

## Why

- **ADR 0002 rejected GLS on theory; the evidence is now empirical.** On the bundled
  CEST example, the baseline planes act as replicates (§5.1 of the specification):
  - OLS gives the smallest plane-to-plane scatter of relative intensities;
    shrinkage gives 1.07–1.34× and GLS 1.30–1.95× that scatter, and 1.9–2.7× for
    overlapping peaks.
  - The OLS conditional sandwich σ is calibrated: empirical / stated 1.06 under the
    header R, 1.01 under the measured R.
- **Correcting R does not rescue GLS.** The example's ¹⁵N axis was linearly
  predicted, invisibly to the header. With the correlation measured in noise-only
  columns, GLS's whitened χ²ᵣ rises from 10.9 to 18.4: GLS up-weights the late,
  LP-extended part of the FID, where neither the signal model nor the noise
  covariance near peaks is known.
- **One σ cannot serve both absolute amplitudes and profiles.** For an isolated peak
  the lineshape uncertainty is exactly a common scale error, Cov = δ_st c + a_s a_t k,
  which cancels in relative intensities. Using the marginal σ as independent
  per-plane errors overstates a downstream relaxation rate's error up to 1.47×, and
  ChemEx kinetic errors 1.08× (resolved) to 2–3× (strong overlap). The profile σ is
  calibrated within 5% except under strong overlap with differing profiles, which
  the flag marks.

## Consequences

- **No change to fitted values** or to the noise model of ADR 0002.
- **`intensity_err` changes meaning** from the conditional σ to the profile σ. It is
  identical for isolated peaks and can grow for overlapping ones; the marginal and
  conditional σ get their own columns.
- **Breaking formats:** the archive, the analysis project and the fit JSON store the
  factors that rebuild Σ (a size independent of the number of planes), the flag and
  a noise-power check, under new major versions. Reopening never refits.
- **Release gate:** the new reporting is not released until validation T3 passes.
- **The noise-power check** compares each axis's time-domain noise power with the
  header's prediction and warns on unrecorded processing such as LP. It never
  changes the noise model or an error.
- **Remaining limits:**
  - Replicates measure repeatability, not accuracy. A bias common to every plane,
    such as LP extrapolation error, is invisible to them; a known-truth test through
    real processing (T7) is needed to measure it for overlapping peaks.
  - The real-data evidence is one dataset.
  - MCMC stays deferred: OLS is not a likelihood estimator under R, and the GLS
    likelihood is contradicted on real data.
- **When to revisit GLS:** only for spectra whose late FID the model describes, and
  then decided by the replicate test, not by theory.
