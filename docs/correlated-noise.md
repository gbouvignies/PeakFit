# Correlated spectral noise

**Status:** approved specification, implemented in three steps (§11).
- Steps 1 and 2 (#129, #130) are merged.
- Step 3, which publishes the errors, is under review.

Decision record: [ADR 0002](adr/0002-processing-derived-sandwich-errors.md), building
on [ADR 0001](adr/0001-sigma-only-uncertainties.md).

**Evidence:** `tools/validate_correlated_noise.py`, with results in
[`validation/correlated-noise.json`](validation/correlated-noise.json). The run is
deterministic, and every number below comes from it.

**Question:** zero filling and apodization correlate the noise of neighbouring
spectral points. PeakFit's analytical errors, its reduced χ² and its MCMC likelihood
all assume independent points. What should change?

**Scope:** only the noise *correlation* R, a unit-diagonal matrix, with Cov = σ²R.
The noise *level* σ, and whether to estimate it locally or per plane, is a separate
question (§9).

## 1. Decisions

These were settled on 2026-11-04.

1. **Estimator unchanged.** Fits stay ordinary least squares, so every fitted value
   is unchanged.
2. **Correlation from the recorded processing only.** It is exact under the
   assumptions of §2.1, but not necessarily a description of the actual spectrum's
   noise.
   - Processing the model does not cover, and inconsistent headers, give σ-only
     errors with a recorded warning, "noise correlation not modelled".
   - There is no fallback that measures the correlation from the spectrum.
3. **Sandwich errors.** Where the correlation is modelled, the published analytical
   errors are the covariance of the least-squares estimates under σ²R (§5).
4. **Reduced χ² unchanged.** It keeps today's definition. The docs say its
   expectation can differ from 1 under correlated noise. There is no alternative
   definition and no effective-degrees-of-freedom threshold.
5. **MCMC unchanged.** The sampler keeps the independence likelihood.
   - Its intervals are conditional on that likelihood, and are never presented as
     calibrated for correlated noise, even when the convergence checks pass (§5.5).
   - The horizontal (Chandler–Bate) adjustment is rejected (§6).
   - Scalar tempering is not adopted unless C5 (§7) demonstrates its behaviour with
     the real sampler.
6. **Real-data validation needs genuine replicates (C6).** The bundled CEST series
   has none, so it cannot establish empirical coverage. Synthetic evidence and the
   processing model are reported separately from any real-data comparison.

## 2. The noise model

### 2.1 Assumptions

The correlation is exact when all of these hold:
- thermal noise is white and circular complex in the time domain of every acquired
  dimension, and independent between planes;
- each axis is processed exactly as its header records:
  - a sine-bell apodization (`SP`: `APODQ1`–`APODQ3`) or none;
  - first-point scaling `C1 + 1`;
  - zero filling to `FTSIZE`;
  - a Fourier transform and the first-order phase `P1`;
  - keeping the real part;
- axes are processed independently, so R is separable.

It does not cover:
- t1 noise or spectrometer instability;
- linear prediction, non-uniform-sampling reconstruction, solvent filters, baseline
  correction, or other window functions;
- input that is not NMRPipe.

Extraction does not matter, because the noise is stationary along each axis.

### 2.2 Correlation

For an axis with N acquired complex points, window w_t (first point scaled), FT size M
and first-order phase φ₁, the noise correlation of the real points at lag k is

  ρ(k) = Re[C(k) e^{iφ₁k/M}] / C(0),  C(k) = Σ_t w_t² e^{−2πikt/M}.

This follows from E[X_k X_l*] = 2s²C(k − l) and the zero pseudo-covariance of circular
noise.

For a cluster, R is the elementwise product, over spectral axes, of ρ evaluated at the
lags between its fitting samples. It is the identity between planes, and is never
stored: it is regenerated from the descriptor and the recorded fitting samples.

**Verification (C1):** the formula matches Monte Carlo processing of white noise to
within 0.007 at lags 0–5. This holds for both CEST chains and for zero fill ×2 and ×4
without apodization.

### 2.3 Header fields and consistency

For each transformed spectral axis (`FTFLAG` = 1) with S stored points:

| Quantity | Header field | Required |
|---|---|---|
| N, acquired complex points | `TDSIZE` | integer > 0 |
| M, Fourier-transform size | `FTSIZE` | integer ≥ max(N, S) |
| Window | `APODCODE` | 0 (none) or 1 (`SP`) |
| Sine-bell parameters | `APODQ1`, `APODQ2`, `APODQ3` | finite, Q3 > 0, when `APODCODE` = 1 |
| First-point scale | `C1` + 1 | finite |
| First-order phase | `P1` | finite |
| Apodized length | `APOD` | the consistency check below |

**Consistency check:** |`APOD` − N·S/M| < 1.
- NMRPipe records the apodized length, rescaled by any extraction and stored as a
  whole number. Without extraction (S = M), `APOD` must equal N exactly.
- With extraction, rounding down or to the nearest integer both pass.
  - The CEST ¹H axis has N·S/M = 768 × 546/2048 = 204.75, and `APOD` = 204.
  - The ¹⁵N axis has S = M and `APOD` = N = 92.
- A window applied over fewer points than were acquired fails the check. This
  happens with `SP -size`, for example; that processing reading `APOD` alone would
  imply ρ₁ = 0.97 instead of 0.66 on the CEST ¹H axis.

**Safe fallback:** any of these leaves the run "not modelled", with the reason
recorded:
- a missing or non-finite field, or an unsupported window;
- a failed check;
- an untransformed spectral axis;
- input that is not NMRPipe.

Its errors are then σ-only, as today (§5.5). There is no partial model: if one axis
fails, the whole run is not modelled.

### 2.4 Comparison with a real spectrum

This is not a validation. On the bundled CEST example, the predicted noise
autocorrelation is compared with one measured on noise-only points (|x| < 3σ):

| Lag | ¹⁵N predicted | ¹⁵N measured | ¹H predicted | ¹H measured |
|---:|---:|---:|---:|---:|
| 1 | 0.764 | 0.772 | 0.656 | 0.608 |
| 2 | 0.295 | 0.392 | 0.043 | 0.079 |
| 3 | −0.015 | 0.048 | −0.233 | −0.155 |
| 4 | −0.071 | −0.118 | −0.162 | −0.103 |

- Lag 1 agrees closely along ¹⁵N, and to within 0.05 along ¹H. Differences of up to
  0.10 remain at higher lags.
- These differences are far larger than the measurement's sampling error. Since C1
  rules out the formula, they come from the measurement or from the spectrum.
  Candidates are residual signal in the mask, the effect of truncation on the mask,
  and noise sources outside §2.1.
- They are not established, and PeakFit does not use the measured curve.

## 3. Size of the effect

The table gives the SD inflation of a single line's least-squares amplitude, with all
points used and 92 acquired points. "sine-bell²" is NMRPipe `SP -end 0.98 -pow 2`, and
R2·aq is the decay over the acquisition time.

| Processing | ρ₁ | Inflation, R2·aq = 0.3 · 1.3 · 5 |
|---|---:|---|
| No zero fill, sine-bell² (off 0.5) | 0.16 | 1.03 · 1.08 · 1.21 |
| Zero fill ×2, no apodization | 0.01 | 1.01 · 1.03 · 1.09 |
| Zero fill ×2, sine-bell² (off 0.4) | 0.57 | 1.33 · 1.38 · 1.47 |
| Zero fill ×2, sine-bell² (off 0.5) | 0.65 | 1.45 · 1.54 · 1.71 |
| Zero fill ×4, no apodization | 0.64 | 1.43 · 1.45 · 1.54 |
| Zero fill ×4, sine-bell² (off 0.5) | 0.90 | 2.06 · 2.17 · 2.42 |

- **Monte Carlo confirms three rows:** 1.544, 1.553 and 1.086.
- **Twofold zero fill without apodization** is only nearly independent. Odd lags keep
  ρ = 1/N, so the predicted inflation is small but nonzero: 1.014 · 1.029 · 1.089. C3
  tests this: Monte Carlo gives 1.022 · 1.035 · 1.093, within 2.5 SE of 0.003.
- **Apodization with zero filling** is what correlates the points strongly.
- **2D clusters:** the inflation is roughly the product of the per-axis factors.

## 4. What correlation does

- **Estimates:** least squares stays unbiased, but it is no longer the
  minimum-variance linear estimator.
- **Covariance:** σ²(JᵀJ)⁻¹JᵀRJ(JᵀJ)⁻¹, the sandwich (Huber 1967; White 1980). It is
  exact to first order for the existing estimator.
- **χ²:** E[χ²] = tr((I − H)R) instead of n − p, with H the hat matrix. It is usually
  lower, and its variance is 2 tr(((I − H)R)²).
- **Independence likelihood:** its curvature, JᵀJ/σ², is too large. Its intervals
  are therefore too narrow, by a factor that differs per parameter.

## 5. Specification

### 5.1 Conditional amplitude errors (published)

- **Definition:** for each peak and plane, σ√diag(G⁻¹ΦRΦᵀG⁻¹), with G = ΦΦᵀ. This
  becomes `intensity_err` and the amplitude covariance.
- **What it means:** like today's error, it is *conditional* on the fitted lineshape
  parameters θ̂, which are held fixed.

### 5.2 Total amplitude uncertainty (not an analytical output)

- **Definition:** total uncertainty also includes the variability of the nonlinear
  parameters.
- **How large it is:** in the synthetic single-plane fits it is 1.48–1.63 × the
  conditional sandwich SD (§8). With many planes sharing θ, the difference is
  smaller.
- **Where it comes from:** PeakFit does not publish it analytically. MCMC provides
  it, but under the independence likelihood (§5.5).
- **Naming:** documentation and the GUI always say "conditional amplitude error" or
  "total amplitude uncertainty", never just "amplitude uncertainty".

### 5.3 Nonlinear-parameter errors

- **Definition:** the sandwich built from the VarPro Jacobian, with residuals ordered
  as points × planes and noise covariance R ⊗ I_planes.
- **Withholding:** the existing rank and conditioning rules of
  `identifiable_covariance` are decided first, on the independent covariance, and
  are unchanged. A covariance withheld there is withheld here too.
- **The bound-crossing rule** applies to the published sandwich SE.

### 5.4 Amplitude identifiability and withholding

- **Identifiability stays a property of the shapes,** and the rules do not change.
  `factorize_shapes` withholds the whole analytical amplitude covariance when the
  shapes lose numerical rank, or when any direction is weak (s ≤ √ε·s₁).
- **The sandwich adds no withholding rule.**
  - Write Φᵀ = USVᵀ for the retained directions. Where the independent covariance
    σ²VS⁻²Vᵀ is published, every direction is strong, and the sandwich is
    σ²VS⁻¹(UᵀRU)S⁻¹Vᵀ.
  - Where it is withheld, so is the sandwich. The two are withheld for exactly the
    same clusters.
- **R = I regression.** Then UᵀRU = I, so the sandwich reproduces today's covariance
  to round-off. The same holds for the nonlinear sandwich. This is the exact
  regression test of the current σ-only path.
- **Amplitude correlation:** the review measure "amplitude correlation" uses the
  sandwich covariance where R is modelled.
- **MCMC withholding and flags** are unchanged, and keep their per-combination
  weak-direction rule, because MCMC keeps the independence model.

### 5.5 Reporting and warnings

- **A run-level noise-model record:**
  - either "modelled from processing", with each axis's descriptor (N, M, window and
    its parameters, first point, P1);
  - or "not modelled", with the reason.

  It is stored with the run. A `noise_correlation` column ("modelled" or "not
  modelled") goes in `parameters.csv` and `intensities.csv`. The CLI summary and the
  GUI navigator, profile and review state it once.
- **Not modelled:** σ-only errors, with "Noise correlation not modelled (reason):
  uncertainties assume independent spectral points and can be too small for
  zero-filled or apodized data."
- **MCMC:** every MCMC output carries one fixed statement, on the terminal and on the
  report page:
  - its intervals use an independent-noise likelihood;
  - they are not calibrated for correlated noise, even when the convergence checks
    pass.

  "Passed the convergence checks" is never shown without it. No per-parameter
  sandwich-to-independence factors are shown. They are local covariance diagnostics,
  and could be misread as corrections to posterior intervals. The statement is
  independent of the run, so the MCMC record does not change.
- **χ²ᵣ:** reported as today. The documentation explains that under correlated noise
  its expectation is tr((I − H)R)/(n − p), which can be well below 1 (0.54–0.65 in
  §8).

### 5.6 Formats and cost

- **Formats:** scientific archive 8.0.0, fit JSON 7.0.0 and analysis project 11.0.0,
  for the noise-model record, plus the CSV column. The MCMC record is unchanged.
  Earlier files are refused; refit.
- **Cost:** O(n²) per cluster once at finalization. R is never stored.

### 5.7 Documentation

- `docs/outputs.md` (uncertainty section) and `docs/mcmc.md`.
- `GLOSSARY.md` gains three terms: noise correlation, sandwich error, and conditional
  amplitude error against total amplitude uncertainty.
- ADR 0002 records the approach, and supersedes ADR 0001's statement that correlated
  noise is unmodelled.

## 6. Alternatives rejected

- **A global effective-sample-size factor, √(1 + 2Σρ).**
  - It is the inflation for the mean of a flat signal, not for a peak.
  - Over all lags its value is M·w₀²/Σw², set by the first point alone: 4.70 or 1.19
    for the ¹⁵N chain, with first-point scaling 1 or ½.
  - Truncated, it depends on the truncation.
- **Measuring the correlation from the spectrum** (decision 2). It is biased by
  residual signal and the noise mask (§2.4), and would be a second estimator.
- **Generalized least squares.**
  - It changes the published estimates.
  - It depends on R⁻¹, whose condition number is 1.8 × 10⁶ even for a 13-point
    window, in directions where the apodization removed nearly all noise.
  - In the committed misfit test (an unapodized model fitted to apodized data), it
    gained nothing: its position SD was 1.46 × that of least squares, and neither was
    biased.
  - An earlier draft reported large GLS biases under misfit. They are not
    reproduced here and are not relied on.
- **A correlated-Gaussian likelihood for MCMC.** It is coherent, but it centres on the
  GLS estimate and depends on R⁻¹ for the same reasons. Not now.
- **The horizontal (Chandler–Bate 2007) adjustment,** ℓ_p(θ̂ + C(θ − θ̂)). It is not
  the likelihood of any data model, and is calibrated only near θ̂, where the
  analytical sandwich already suffices. Elsewhere:
  - its fixed linear map moves every other mode, so peak exchanges are misplaced;
  - it evaluates the likelihood at transformed points while the bounds stay on θ;
  - it is undefined wherever the sandwich is withheld;
  - it depends on an arbitrary choice of matrix square root.
- **Scalar tempering,** exp(k·ℓ_p): a generalized-Bayes update (Bissiri, Holmes and
  Walker 2016). It keeps modes, bounds and exchange detection, and equals sampling
  with σ/√k. But one k cannot match per-parameter inflations. In the grid study (§8)
  the amplitude under-covered at 68% (0.64), while position and R2 over-covered
  (0.77, about five SE high). It is not adopted unless C5 shows otherwise.
- **Empirical scatter as the published error.** It needs genuine replicates. It is the
  validation of C6, not an error model.

## 7. Validation

| Test | What it checks | Status |
|---|---|---|
| C1 | ρ against Monte Carlo processing; the inflation table | Done (§2.2, §3) |
| C2 | Coverage of σ-only and sandwich errors: conditional amplitude, and the joint fit | Done for a 1D single line with the exact processed lineshape. Pairs, 2D clusters and PeakFit's `sp1`, `sp2` and `no_apod` lineshapes are the implementation's tests. |
| C3 | Twofold zero fill, no apodization: the small predicted inflation (§3), by Monte Carlo. The exact regression test of today's σ-only path is R = I (§5.4), a unit test of the implementation. | Prediction done |
| C4 | E[χ²] and Var[χ²] against tr((I − H)R) and 2 tr(((I − H)R)²) | Done |
| C5 | MCMC under correlated noise | Preliminary grid study done (§8). The gate for any change to MCMC, the real sampler on overlapping peaks with individual amplitudes and identifiable sums, is not run. |
| C6 | Real data with genuine replicates (duplicate planes or a repeated experiment): sandwich errors against the replicate scatter | Not possible with the bundled data |

The processing-model comparison with the CEST spectrum (§2.4) is reported separately,
and is not part of this table.

```bash
uv run python tools/validate_correlated_noise.py --output /tmp/correlated-noise.json --results RUN --spectrum examples/04-uncertainty-analysis/data/pseudo3d.ft2
```

`RUN` is a fit of the bundled example. Without `--results`, the CEST comparison is
skipped.

## 8. Evidence

**Synthetic set-up:**
- a line through the ¹⁵N chain: 92 points, sine-bell² (off 0.4), zero fill to 256;
- a 13-point window, peak SNR 30;
- 1,000 realizations, with the fit starting at the truth.

Each cell gives the observed scatter / the stated SD, then the 68% · 95% coverage. The
coverage SE is 0.015 · 0.007.

| Case | Quantity | σ-only | Sandwich |
|---|---|---|---|
| Narrow (R2·aq 1.3) | Conditional amplitude | 1.60 · 0.48 · 0.78 | 1.00 · 0.69 · 0.95 |
| | Joint: amplitude | 1.58 · 0.48 · 0.78 | 1.00 · 0.68 · 0.95 |
| | Joint: position | 1.38 · 0.54 · 0.84 | 1.00 · 0.69 · 0.94 |
| | Joint: R2 | 1.36 · 0.55 · 0.86 | 1.00 · 0.69 · 0.95 |
| Broad (R2·aq 5.2) | Conditional amplitude | 1.65 · 0.46 · 0.78 | 0.99 · 0.70 · 0.95 |
| | Joint: amplitude | 1.57 · 0.45 · 0.79 | 0.98 · 0.69 · 0.95 |
| | Joint: position | 1.60 · 0.48 · 0.78 | 1.00 · 0.69 · 0.95 |
| | Joint: R2 | 1.51 · 0.50 · 0.81 | 0.98 · 0.70 · 0.95 |
| ×2 zero fill, no apodization | Conditional amplitude | 0.97 · 0.69 · 0.96 | 0.96 · 0.70 · 0.96 |
| | Joint: amplitude | 1.01 · 0.69 · 0.95 | 1.00 · 0.69 · 0.95 |

**Total against conditional:** the joint-fit amplitude SD is 1.63 (narrow), 1.48
(broad) and 1.57 (×2 zero fill) × the conditional SD. This is one plane, so θ is fitted
from the same data as the amplitude.

**χ² (C4):**

| Case | n − p | tr((I − H)R) | Observed mean | Predicted · observed variance | Nominal χ²ᵣ |
|---|---:|---:|---:|---|---:|
| Narrow | 10 | 6.70 | 6.47 | 24.2 · 22.7 | 0.65 |
| Broad | 10 | 5.32 | 5.37 | 16.2 · 15.8 | 0.54 |
| ×2 zero fill | 10 | 9.97 | 10.05 | 19.9 · 20.4 | 1.01 |

**C5, preliminary.** This is the exact grid posterior of (position, R2), with the
amplitude drawn conditionally: narrow line, 600 realizations. The coverage SE is 0.019
· 0.009.

| Likelihood | Amplitude | Position | R2 |
|---|---|---|---|
| Independence (today) | 0.46 · 0.80 | 0.61 · 0.86 | 0.62 · 0.88 |
| Tempered, k = 0.48 | 0.64 · 0.94 | 0.77 · 0.97 | 0.77 · 0.96 |

**The CEST prediction** is not a validation. Under the processing model, the
conditional sandwich factor for each peak is 2.335 (IQR 2.325–2.344) × σ-only. Whether
it matches the true uncertainty is untested (C6).

## 9. Limitations

- **The processing model only.** Real noise may include t1 noise, instabilities or
  processing outside §2.1. The CEST comparison shows unexplained differences at
  higher lags, and no real-data validation is available.
- **Synthetic scope:** 1D, single lines, the exact lineshape, one plane. Overlapping
  pairs and 2D clusters are untested.
- **First order.** The sandwich is exact only in the locally linear regime.
  Near-degenerate clusters stay withheld, or are approximate.
- **Bias.** The sandwich describes variance, not misfit bias.
- **MCMC remains uncalibrated for correlated noise.** On processed data its intervals
  will often be narrower than the published conditional errors.
- **χ²ᵣ** keeps a definition whose expectation depends on the noise correlation.

## 10. Separate question: the noise level σ

This document does not change σ. Whether to estimate σ per plane or locally, for
example for t1 noise near intense peaks, needs its own assessment and its own
validation against replicates. That noise is neither white nor stationary, so it
would also break §2.1.

## 11. Implementation sequence

The direction was approved on 2026-11-04. Each step is its own pull request, reviewed
before merging. None changes the least-squares estimates, σ estimation, the χ²ᵣ
definition or the MCMC sampler, and none adds another noise model.

1. **Noise model.** Read the per-axis processing from the NMRPipe header (§2.3),
   compute ρ and a cluster's R (§2.2), and return "not modelled", with the reason,
   whenever a check fails.
   - Nothing is stored or published.
   - Tests cover ρ against an exact processing operator, the CEST header, and each
     fallback.
2. **Sandwich calculations.** The conditional amplitude and nonlinear-parameter
   sandwiches (§5.1, §5.3), with the withholding of §5.4.
   - Tests: R = I reproduces today's errors to round-off, and the sandwich matches a
     direct formula.
   - A validation study with PeakFit's own fitting engine must pass before step 3:
     isolated and overlapping 2D clusters with correlated noise, and the
     identifiability cases (near-coalescent and duplicated peaks).
   - Nothing is published yet.
3. **Publishing.**
   - The noise-model record and the sandwich errors in the exports.
   - Format bumps (§5.6), the CSV column, and the CLI and GUI statements.
   - The MCMC warning (§5.5).
   - Documentation and ADR 0002.
   - The engine validation extends to `sp1` and `no_apod`, and to the published
     errors themselves.
   - Strongly overlapping or near-coalescent fits can be biased with small local
     errors and no warning
     ([validation](validation/sandwich-errors.md#near-coalescent-peaks-and-warnings)).
     This is documented as a limitation. A resolution-based advisory flag is a
     separate investigation, with no threshold until one is shown useful.

C5 (with the real sampler) and C6 (with replicates) remain future validation. They are
not prerequisites for these steps.

## References

- E. Bartholdi and R. R. Ernst (1973). Fourier spectroscopy and the causality
  principle. *Journal of Magnetic Resonance* 11, 9–19.
- P. G. Bissiri, C. C. Holmes and S. G. Walker (2016). A general framework for
  updating belief distributions. *Journal of the Royal Statistical Society B* 78,
  1103–1130.
- R. E. Chandler and S. Bate (2007). Inference for clustered data using the
  independence loglikelihood. *Biometrika* 94, 167–183.
- P. J. Huber (1967). The behavior of maximum likelihood estimates under nonstandard
  conditions. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics
  and Probability* 1, 221–233.
- H. White (1980). A heteroskedasticity-consistent covariance matrix estimator and a
  direct test for heteroskedasticity. *Econometrica* 48, 817–838.
