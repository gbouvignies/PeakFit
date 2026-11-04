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

The bundled CEST example (`examples/04-uncertainty-analysis`) has 166 peaks, all with
χ²ᵣ > 1 (median 2.00, IQR 1.70–2.48). Its intensities are identical under both
conventions. Each `intensity_err` shrinks by √χ²ᵣ: a median of 1/1.41, and at most
1/3.09.

**Empirical check:** each peak's plane-to-plane scatter is a robust SD of successive
differences over the 130 saturation planes (25 Hz steps), divided by √2. It is
independent of the fit's error model.

| | median | IQR |
|---|---:|---|
| scatter / σ-only error | 2.72 | 2.53–3.00 |
| scatter / former rescaled error | 1.95 | |

**The shortfall does not track χ²ᵣ.** The correlation of log χ²ᵣ with log(scatter /
error) is only 0.18. The rescaling therefore did not correct it; it reduced it by a
factor unrelated to its cause.

**Most of it is correlated noise:**
- The spectrum's noise is strongly correlated between neighbouring points. Lag-1
  autocorrelation is 0.77 along ¹⁵N (92 → 256 points, sine bell) and 0.61 along ¹H
  (768 → 2,048 points, sine bell).
- Treating points as independent then understates amplitude errors by about 2.4×.
- σ is close: each plane's own noise is 1.03–1.06 × the global estimate.
- The median χ²ᵣ of about 2 therefore mostly reflects lineshape misfit, and the
  former √χ²ᵣ partly offset the correlated-noise shortfall by coincidence.

## Reading

- **Well-behaved data:** σ-only errors are correct, and MCMC adds the lineshape
  uncertainty they omit.
- **What χ²ᵣ means:** it is a goodness-of-fit diagnostic, not an error correction. A
  large value can mean underestimated or locally varying σ, model misfit or
  correlated noise. Only the first is fixed by a larger error.
- **Zero-filled, apodized spectra:** a χ²ᵣ near 1 does not validate the errors.
  On this CEST series, both conventions understate the plane-to-plane scatter: by
  about 2.7× (σ-only) or 2× (former).
  - For downstream fits that need realistic errors, estimate them from the data
    themselves, such as duplicate planes or the scatter of a flat baseline.
  - Modelling correlated noise in PeakFit is a separate scientific question.

Reproduce, with `RUN` a fit of the bundled example:

```bash
uv run python tools/validate_sigma_uncertainties.py --output /tmp/sigma-uncertainties.json --results RUN --spectrum examples/04-uncertainty-analysis/data/pseudo3d.ft2
```
