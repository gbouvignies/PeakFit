# σ-only uncertainties

Validation date: 2026-11-04. Evidence for [ADR 0001](../adr/0001-sigma-only-uncertainties.md).
Amplitude errors are σ√diag((ΦΦᵀ)⁻¹), from the supplied or estimated noise σ alone.
They are no longer multiplied by √max(1, χ²ᵣ). Intensities themselves are unchanged.

`tools/validate_sigma_uncertainties.py`; results in
[`sigma-uncertainties.json`](sigma-uncertainties.json).

## Synthetic spectra

**Set-up:**
- A 1D Gaussian cluster over four planes, noise σ = 0.05, the amplitudes of the MCMC
  coverage study ([mcmc-coverage.md](mcmc-coverage.md)), fitted by the ordinary
  pipeline.
- 400 noise realizations per case.
- **Scatter** is the observed SD of the fitted amplitudes over the realizations.
- **Coverage** is of the noise-free fitted value: the truth, except under misfit.
- MCMC runs on the first 40 realizations of each well-behaved case, with 16 walkers
  × 3,000 steps.

| Case | median χ²ᵣ | scatter / σ-only error | scatter / former rescaled error | σ-only coverage 68% · 95% | former rescaled 68% · 95% |
|---|---:|---:|---:|---|---|
| Single peak, well-behaved | 1.00 | 1.06 | 1.04 | 0.658 · 0.940 | 0.662 · 0.943 |
| Resolved pair (10 points), well-behaved | 0.99 | 1.06 | 1.05 | 0.645 · 0.929 | 0.650 · 0.934 |
| σ supplied at half its value | 3.99 | 1.98 | 0.99 | 0.388 · 0.673 | 0.684 · 0.944 |
| Lorentzian data fitted as Gaussian | 2.58 | 1.14 | 0.71 | 0.611 · 0.903 | 0.822 · 0.989 |
| Correlated noise, ρ₁ = 0.67 | 0.98 | 1.66 | 1.64 | 0.452 · 0.764 | 0.458 · 0.771 |

**Well-behaved cases:**
- The σ-only errors match the observed scatter to about 6%. The shortfall is the
  lineshape uncertainty that conditional errors omit by design.
- MCMC adds it back: its amplitude SD is 1.04 × the σ-only error, and the observed
  scatter is 0.99 (single) and 1.01 (pair) × the MCMC SD. All 80 MCMC runs passed.
- The rescaling made no material difference here, because χ²ᵣ ≈ 1.

**Elevated χ²ᵣ:** the cause matters.
- **Underestimated σ:** this is the one case the rescaling corrected. The σ-only
  errors are then too small by the factor σ was underestimated.
- **Lineshape misfit:**
  - The rescaling overstated the scatter by 41% (0.71). The σ-only errors understate
    it by 14%.
  - Neither represents the systematic error. The fitted height is biased by −10.7%,
    and the true height is covered at 95% in only 9% (σ-only) or 22% (rescaled) of
    realizations.
- **Correlated noise:** χ²ᵣ stays near 1, so the rescaling never responds. Both
  conventions understate the scatter by about 1.65×.

## Real spectra

The bundled CEST example (`examples/04-uncertainty-analysis`) has 166 peaks, fitted
over 131 planes: one unsaturated reference and 130 saturation offsets in 25 Hz steps,
none repeated.

### Direct effect of the change (exact)

- Intensities are identical under both conventions.
- Every peak has χ²ᵣ > 1 (median 2.00, IQR 1.70–2.48), so every `intensity_err`
  shrinks by exactly √χ²ᵣ: a median of 1/1.41, and at most 1/3.09.

### Measurements, and what each rests on

| Quantity | Value | Assumptions |
|---|---|---|
| Each plane's noise SD / the global σ | 1.03–1.06 | Points with \|x\| < 3σ only, corrected for that truncation as if the noise were Gaussian. |
| Lag-1 noise autocorrelation, ¹⁵N · ¹H | 0.77 · 0.61 | The same points. Truncation, and signal leaking below the cut, bias it somewhat. Processing: ¹⁵N 92 → 256 points and ¹H 768 → 2,048 points, both sine bell. |
| Robust SD of successive-plane differences / σ-only error | median 2.72 (IQR 2.53–3.00) | The SD is divided by √2 and taken over the saturation planes. See below for what it contains. |
| The same / former rescaled error | median 1.95 | As above. |
| Correlation of log χ²ᵣ with log of that ratio | 0.18 | As above. |

### What these measurements do not establish

- **The successive-difference scatter is not an estimate of amplitude uncertainty.**
  - Successive planes are different saturation offsets, at which the CEST intensity
    is expected to change. Their differences therefore contain genuine profile
    variation as well as noise, in unknown proportion.
  - A robust statistic limits the effect of a few planes inside the dips, but not of
    gradual changes.
  - The 2.72 and 1.95 ratios are thus not a calibration of either convention.
  - Their weak correlation with χ²ᵣ does not show what χ²ᵣ reflects here.
- **A correlation-only prediction is an approximation, not a measurement.**
  - The integrated autocorrelations predict a factor of about 2.4
    (`integrated_autocorrelation_inflation_approximation` in the JSON). That figure treats
    each lineshape as broad relative to the correlation length, and the noise as
    stationary and separable.
  - The variance of a fitted amplitude actually depends on the noise covariance
    projected through the cluster's lineshape matrix, (ΦΦᵀ)⁻¹ΦRΦᵀ(ΦΦᵀ)⁻¹.
  - The prediction's closeness to 2.72 is not evidence that correlated noise
    explains that ratio.
- **The cause of χ²ᵣ ≈ 2 is not identified.** σ measures close to each plane's own
  noise, and correlated noise alone does not raise χ²ᵣ. Lineshape misfit is
  therefore plausible, but non-Gaussian artifacts or other causes are not excluded.

**What would validate the calibration:**
- independent replicates, which this series lacks: duplicate planes or a repeated
  experiment;
- the projected-covariance calculation, checked on synthetic spectra processed with
  the same zero filling and apodization.

Neither is part of this change. A draft assessment of correlated noise is on branch
`correlated-noise-assessment`, awaiting its own review.

## Reading

- **Well-behaved synthetic data:** σ-only errors match the observed scatter to 6%,
  and MCMC adds the lineshape uncertainty they omit.
- **What χ²ᵣ means:** it is a goodness-of-fit diagnostic, not an error correction.
  - In the synthetic tests, a large value arose from an underestimated σ and from
    misfit, and correlated noise left it near 1.
  - Only an underestimated σ is fixed by a larger error.
- **This CEST series:** the change lowers every `intensity_err` by √χ²ᵣ, exactly.
  - This study does not establish how either convention compares with the true
    uncertainty of these amplitudes.
  - The synthetic correlated-noise case shows the mechanism by which independent-point
    errors can be too small without any rise in χ²ᵣ (1.66× there). Its size for this
    spectrum is untested.
  - Downstream fits that need calibrated errors should estimate them from replicate
    data.

Reproduce, with `RUN` a fit of the bundled example:

```bash
uv run python tools/validate_sigma_uncertainties.py --output /tmp/sigma-uncertainties.json --results RUN --spectrum examples/04-uncertainty-analysis/data/pseudo3d.ft2
```
