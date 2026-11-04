# Correlated spectral noise

**Status:** assessment and specification for a decision. No code follows until the
open decisions (§9) are settled. It builds on
[ADR 0001](adr/0001-sigma-only-uncertainties.md) and its evidence,
[validation/sigma-uncertainties.md](validation/sigma-uncertainties.md).

**Question:** how should PeakFit account for spectral noise that zero filling and
apodization correlate between neighbouring points? Today these four quantities all
assume independent points:
- the conditional amplitude errors;
- the nonlinear-parameter errors;
- reduced χ² and its degrees of freedom;
- the MCMC profile likelihood.

**Scope:** this document is about the noise *correlation*, a unit-diagonal matrix R.
The noise *level* σ is a separate matter. Whether σ should be estimated locally or per
plane is a separate question, kept apart in §10. The two compose as Cov = σ²R, so
either can change without the other.

## 1. Recommendation in brief

1. **Keep the estimator.** Fits stay ordinary (unweighted) least squares, and every
   fitted value, intensities included, is unchanged.
2. **Model the correlation from the processing.** Derive R from each spectral axis's
   recorded apodization, acquired points and Fourier-transform size (§3). On the
   bundled CEST example this reproduces the measured noise autocorrelation.
3. **Publish sandwich errors.** Compute the covariance of the least-squares estimates
   under σ²R: amplitudes and nonlinear parameters alike. This is exact for the
   existing estimator in the locally linear regime, and gives nominal coverage on
   synthetic spectra.
   - On the CEST example it raises amplitude errors by 2.34× over σ-only, nearly
     uniformly across peaks.
   - Whether that matches the true uncertainty there cannot be tested: the series
     has no replicates (§7.4).
4. **Report χ²ᵣ against its correlated-noise expectation,** with effective degrees of
   freedom (§5.3).
5. **Adjust the MCMC likelihood** with the Chandler–Bate (2007) horizontal adjustment.
   It makes the profile likelihood's curvature match the sandwich, so MCMC stays
   centred on the published fit (§5.4).
6. **Reject generalized least squares as the default.** It gains a little efficiency,
   but under even mild lineshape misfit it turns a small bias into a large one (§6.3).
7. **Do not use a global effective-sample-size factor.** It weights the correlation
   wrongly for peaks, and the inflation it should represent differs by parameter
   (§6.1).

Empirical scatter from duplicate planes or flat profile regions stays a validation
and downstream practice, not PeakFit's published error (§6.4).

## 2. What is assumed today

For one cluster with shapes Φ (peaks × points), data y_s in plane s, and noise σ:

| Quantity | Current formula | Assumption |
|---|---|---|
| Conditional amplitude errors | σ√diag(G⁻¹), G = ΦΦᵀ | Cov(y_s) = σ²I |
| Nonlinear errors | σ√diag((JᵀJ)⁻¹), J the VarPro Jacobian | Cov(y) = σ²I |
| χ²ᵣ | χ² / max(1, n_obs − n_par) | E[χ²] = n_obs − n_par |
| MCMC | exp(−½ χ²_min(θ)), the profile likelihood | independent Gaussian points |

All four are correct for independent points and too optimistic for positively
correlated ones. The estimates themselves stay unbiased under correlated noise. Only
their stated precision and the likelihood's curvature are wrong.

## 3. Where the correlation comes from

Thermal noise is white in the time domain of every acquired dimension, and
independent between planes, since each plane is a separate experiment. Take a complex
FID of N points, apodized by w_t, zero filled to M points and Fourier transformed,
keeping the real part after phasing. The noise covariance between real points k and
l of that axis is σ²ρ(k − l), with

  ρ(k) = Re[C(k) e^{iφ₁k/M}] / C(0),  C(k) = Σ_t w_t² e^{−2πikt/M},

where φ₁ is the first-order phase. This follows from E[X_k X_l*] = 2s²C(k − l) and the
zero pseudo-covariance of circular complex noise. The noise is stationary along each
axis: ρ depends on lag only, and extraction does not change it. In a pseudo-ND
spectrum, R is the elementwise product of the per-axis lag matrices over a cluster's
fitting samples. It is the identity between planes.

**Zero filling and apodization act together.** For the real part:
- zero filling up to twofold alone leaves the points uncorrelated, because the
  mirrored component fills the extra points (Bartholdi and Ernst 1973);
- apodization without zero filling correlates them only weakly;
- together they correlate them strongly.

The table gives the inflation of a fully covered line's amplitude SD (exact R; 92
acquired points; "sine-bell²" is NMRPipe `SP -end 0.98 -pow 2`; R2·aq is the decay
over the acquisition time). It is the factor by which σ-only errors understate the
true SD.

| Processing | ρ₁ | Inflation, R2·aq = 0.3 · 1.3 · 5 |
|---|---:|---|
| No zero fill, no apodization | −0.01 | 1.00 · 1.00 · 0.98 |
| No zero fill, sine-bell² (off 0.5) | 0.16 | 1.04 · 1.08 · 1.21 |
| Zero fill ×2, no apodization | 0.00 | 1.00 · 1.00 · 0.99 |
| Zero fill ×2, sine-bell² (off 0.4) | 0.57 | 1.33 · 1.38 · 1.47 |
| Zero fill ×2, sine-bell² (off 0.5) | 0.66 | 1.45 · 1.54 · 1.71 |
| Zero fill ×4, no apodization | 0.64 | 1.42 · 1.41 · 1.40 |
| Zero fill ×4, sine-bell² (off 0.5) | 0.90 | 2.06 · 2.17 · 2.42 |

The inflation depends on the linewidth as well as the processing. In a 2D cluster it
is roughly the product of the per-axis values. The same mechanism governs the
precision of spectral integrals (Bourg and Nuzillard 1998; Ebel, Dreher and Leibfritz
2006). Zero filling and apodization are also known to make in vivo MRS Cramér–Rao
bounds misleading (Landheer and Juchem 2021).

### Check on the bundled CEST example

`examples/04-uncertainty-analysis` has these processing parameters:
- ¹⁵N: 92 → 256 points, SP off 0.4, first point ×1.0, P1 180°;
- ¹H: 768 → 2,048 points, SP off 0.3, first point ×0.5, extracted to 546 points.

The NMRPipe header records all of them. The measured curve uses the noise-only points
(|x| < 3σ), as in `tools/validate_sigma_uncertainties.py`.

| Lag | ¹⁵N predicted | ¹⁵N measured | ¹H predicted | ¹H measured |
|---:|---:|---:|---:|---:|
| 1 | 0.764 | 0.772 | 0.656 | 0.608 |
| 2 | 0.295 | 0.392 | 0.043 | 0.079 |
| 3 | −0.015 | 0.048 | −0.233 | −0.155 |
| 4 | −0.071 | −0.118 | −0.162 | −0.103 |

**The header alone predicts the correlation** to within the scatter of the measured
curve, whose estimate is biased slightly by residual signal. Moving the first-point
scaling between axes changes the predicted amplitude inflation by 0.5% (2.335 against
2.322). The model needs no fitted noise parameters.

## 4. What correlation does to each quantity

Let H be the hat matrix of the locally linear fit, and J the Jacobian.

- **Estimates:** least squares stays unbiased, but it is no longer the minimum-variance
  linear estimator.
- **Covariance:** the sandwich σ²(JᵀJ)⁻¹JᵀRJ(JᵀJ)⁻¹ (Huber 1967; White 1980). For
  conditional amplitudes it is σ²G⁻¹ΦRΦᵀG⁻¹.
- **Inflation:** each parameter has its own factor, √(sandwich / independent). It is
  near 1 for templates whose content lies where the noise spectrum is flat, and
  larger where the apodization concentrates it.
- **χ²:** E[χ²] = tr((I − H)R) rather than n − p. Its distribution is a weighted sum
  of χ²₁ variables, approximated by a scaled χ² with
  ν_eff = tr((I − H)R)² / tr(((I − H)R)²) (Satterthwaite 1946).
- **Profile likelihood:** its curvature is the independent-noise Fisher information,
  JᵀJ/σ², too large by the same per-parameter factors. Its posterior is too narrow.

## 5. Specification of the recommended approach

### 5.1 Noise correlation model

- **Per spectral axis:** a 1D lag-correlation vector ρ, from a recorded description.
  This is the apodization function and its parameters, the acquired complex points N,
  the Fourier-transform size M, the first-point scaling and the first-order phase.
  NMRPipe headers carry all of them. PeakFit already reads the apodization code,
  q1–q3, the acquired points (`APOD`, `TDSIZE`) and `FTSIZE`.
- **Supported apodizations:** start with those PeakFit already models as lineshapes:
  `SP` (`sp1`, `sp2`) and none (`no_apod`). EM and GM have closed forms and could
  follow.
- **Not modelled:** processing the header does not describe, such as linear
  prediction, non-uniform-sampling reconstruction, solvent filters or baseline
  correction. Neither is non-NMRPipe input. What happens then is decision O2.
- **Per cluster:** R = (ρ_axis1 gathered at the lags between fitting samples) ∘
  (ρ_axis2 likewise) ∘ …. It is the identity across planes, and is never stored:
  it is regenerated from the descriptor and the recorded fitting samples.

### 5.2 Analytical errors

- **Conditional amplitude errors:** σ√diag(G⁻¹ΦRΦᵀG⁻¹) per peak and plane, from the
  same rank-aware `factorize_shapes`. Withholding rules are unchanged. If G⁻¹ is
  withheld, so is the sandwich.
- **Nonlinear errors:** the sandwich built from the VarPro Jacobian, with residual
  order points × planes and noise covariance R ⊗ I_planes. The active-bound and
  conditioning rules of `identifiable_covariance` apply first, unchanged.
- **Amplitude covariance and correlations:** the review measure "amplitude
  correlation" comes from the sandwich covariance.
- **One convention:** the sandwich errors replace the σ-only ones, in keeping with ADR
  0001's single convention. Whether independent-point errors are also kept is
  decision O3.

### 5.3 Reduced χ² diagnostic

- **Normalization:** χ²ᵣ = χ² / (S·tr((I − H)R)), with ν_eff reported beside it. It
  then reads 1 for a correct σ, model and noise correlation.
- **Hat matrix:** the amplitude block alone is enough for the expectation. The
  nonlinear parameters are shared across S planes, so they change it by about
  n_nonlinear/(nS).
- **Reading it:** χ²ᵣ remains a diagnostic. A large value can still mean a wrong σ,
  locally varying noise or model misfit. Correlated noise does not raise it: §7.3
  shows that it *lowers* the nominal χ²ᵣ.
- **Alternative:** keeping today's nominal χ²ᵣ is decision O4.

### 5.4 MCMC likelihood

The sampler targets the Chandler–Bate horizontally adjusted profile likelihood:

  ℓ_A(θ) = ℓ_p(θ̂ + C(θ − θ̂)),  with CᵀH_I C = H_A,

where:
- H_I = JᵀJ/σ² is the independence curvature at θ̂;
- H_A is the inverse of the sandwich covariance of θ;
- C is fixed per cluster, for example from Cholesky factors.

Then:
- **Centre and shape:** the posterior stays centred on the published θ̂, and in the
  locally quadratic regime its covariance equals the sandwich. Away from it, the
  profile likelihood's shape (skewness, bounds) is retained.
- **Conditional amplitude draws:** from N(â(θ), σ²G⁻¹ΦRΦᵀG⁻¹).
- **Prior bounds:** they stay on θ itself, not on the transformed coordinates.
- **Cost:** one fixed linear transform per likelihood call.

**Simpler alternative, the vertical adjustment.** It scales ℓ_p by one factor
k = p / tr(H_I⁻¹V) (Chandler and Bate 2007; Ribatet, Cooley and Davison 2012). It is
exact only when every parameter has the same inflation. On the CEST example they do
not: 1.7–2.3 (§7.4). It would therefore over-widen linewidth intervals and
under-widen amplitude intervals. This is decision O5.

**Versioning:** the formulation version of
[mcmc-formulation.md](mcmc-formulation.md) becomes 2.0, and §4 "Noise" (D3) changes
from "conditional on σ" to "conditional on σ and R".

### 5.5 Stored formats and cost

- **New fields:**
  - the per-axis noise-correlation descriptor, about ten numbers per axis;
  - its source: "processing", or the alternatives of O2;
  - the formulation of the published errors.

  Published error arrays keep their shape; only their values change. No n × n
  matrices are stored.
- **Breaking bumps:** scientific archive 8.0.0, fit JSON 7.0.0, analysis project
  11.0.0, MCMC record 2.0.0. Earlier files are refused, as for ADR 0001: refit.
- **Cost:** O(n²p + S n² n_θ) per cluster, once at finalization. On the CEST example
  clusters have at most 154 points, so this is negligible beside the fit.
- **Large clusters:** if they make n² memory matter, R can be truncated at the lag
  where |ρ| < 10⁻³ and applied sparsely.

### 5.6 GUI and documentation

- **Error bars and intervals:** they use the sandwich errors. The navigator, profile
  and review label the noise model once, for example "noise correlation from
  processing (¹⁵N SP, 92→256; ¹H SP, 768→2048)".
- **Unmodelled axes:** when correlation could not be modelled, the GUI says so,
  explains what that means for the errors, and says what the user can do (O2).
- **Documents to update:** `docs/outputs.md` (uncertainty section) and
  `GLOSSARY.md`. The glossary gains: noise correlation, sandwich error, effective
  degrees of freedom.

### 5.7 Validation plan

| Test | What it checks |
|---|---|
| C1 | ρ from the descriptor matches the noise autocorrelation of synthetic spectra pushed through the stated processing, to Monte Carlo error |
| C2 | Synthetic coverage (as §7.1) for single peaks, overlapping pairs and 2D clusters, with `sp1`, `sp2` and `no_apod` lineshapes, within binomial bounds |
| C3 | Unapodized ×2 zero fill: errors equal the σ-only ones to round-off |
| C4 | E[χ²] and its variance match tr((I − H)R) and 2 tr(((I − H)R)²) |
| C5 | MCMC with the adjusted likelihood: V5 and V6 of the MCMC plan repeated under correlated noise |
| C6 | Real data with replicates (duplicate planes or a repeated experiment): sandwich errors against the replicate scatter, recorded in `docs/validation/`. The bundled CEST series has no replicates, so it cannot serve. |

## 6. Candidates compared

### 6.1 Effective-sample-size or variance-inflation factor

This means multiplying every error by √(1 + 2Σρ_k) (Bayley and Hammersley 1946), from
measured autocorrelation or from the processing.

- **Analytical errors:** wrong in principle.
  - 1 + 2Σρ is the inflation for the *mean of a flat signal*. A peak weights the noise
    spectrum by its own content.
  - Over all lags it equals M w₀²/Σw², which the first time-domain point alone sets:
    4.70 or 1.19 for ¹⁵N here, with first-point scaling 1 or ½.
  - Truncated at 12 lags, as in the validation tool, it gives 2.71. The result
    therefore depends on an arbitrary truncation.
  - One scalar also cannot represent per-parameter factors that differ by 35% on the
    CEST example (§7.4). In synthetic tests it over-covered positions and linewidths
    (95% coverage 0.98).
- **χ²ᵣ:** it suggests dividing the degrees of freedom by the factor. That is not the
  correct expectation, tr((I − H)R).
- **MCMC:** it amounts to the vertical adjustment, with the wrong k.
- **Cost and formats:** trivial, one number.
- **Verdict:** rejected. Its correct, template-specific form *is* the sandwich.

### 6.2 Sandwich errors with a modelled R (recommended)

- **Analytical errors:** exact for the existing estimator to first order. Fitted
  values are unchanged. The errors depend on R only through ΦRΦᵀ, so a modest error
  in R gives a modest error in the SD: measured instead of predicted R moves the CEST
  inflation from 2.34 to 2.40.
- **χ²ᵣ:** correct expectation and ν_eff (§5.3).
- **MCMC:** Chandler–Bate adjustment of the existing profile likelihood.
- **Cost and formats:** §5.5.

### 6.3 Generalized least squares (Aitken 1935)

This means fitting with R⁻¹ weighting. It is the maximum-likelihood estimator, and
under exact Gaussian correlated noise it is equivalent to fitting the FID in the time
domain (Bretthorst 1990).

- **Analytical errors:** σ√diag((JᵀR⁻¹J)⁻¹), which are smaller.
  - In synthetic tests GLS lowered the true SD by about 10% for amplitudes, and by up
    to 30% for positions and linewidths of narrow lines.
  - On the CEST example, the GLS amplitude error is 2.16× σ-only, against 2.34× for
    the least-squares sandwich: 8% smaller.
- **Fragility:**
  - R over a fitting window is badly conditioned: condition numbers up to 1.3 × 10¹⁰
    on the CEST example, median 8 × 10⁵. These are the directions where the
    apodization has removed nearly all noise.
  - GLS gives them great weight, and with them any lineshape misfit there. With a
    Gaussian or Lorentzian model of apodized data (the `gaussian`, `lorentzian` and
    `pvoigt` lineshapes), a height bias of 0.2–1 least-squares SD became 2.5–16 GLS
    SD (§7.2).
  - The bundled CEST fit uses the exact `sp2` lineshape and still has χ²ᵣ ≈ 2. Some
    misfit is present even there.
- **Changes the published values:** the estimates move from least squares, and they
  depend on R, so an error in R moves them too.
- **χ²ᵣ:** the whitened χ² has expectation n − p.
- **MCMC:** the exact likelihood under the model. But its posterior centres on the GLS
  estimate, not on the published fit.
- **Cost and formats:** one Cholesky factor per cluster, reused at every iteration.
  Moderate. Same descriptor as §6.2.
- **Verdict:** not the default. A possible later opt-in for exact processed
  lineshapes (decision O6).

### 6.4 Empirical errors from duplicate planes or baseline scatter

This means estimating errors from the scatter of intensities in duplicated planes, or
in flat regions of an intensity profile.

- **Analytical errors:** model-free *when the replicates are genuine*. Their scatter
  then includes every plane-to-plane source of error: correlated noise,
  plane-dependent σ, t1 noise, and fit variability.
  - Duplicate planes or repeated experiments qualify. "Flat regions" qualify only if
    the profile is truly constant there.
  - Successive offsets of a CEST profile do not qualify: the intensity changes
    physically between them, so their differences mix that variation with noise in
    unknown proportion. The 2.72 ratio in
    [validation/sigma-uncertainties.md](validation/sigma-uncertainties.md) is such a
    statistic, and is not an error estimate.
  - Even with genuine replicates it is noisy per peak: a robust SD from about 130
    differences has a relative SE of about 10%.
  - It cannot separate sources, or give per-plane or per-parameter errors. It says
    nothing about positions or linewidths.
- **χ²ᵣ and MCMC:** no role.
- **Cost and formats:** cheap, but it needs plane metadata saying which planes are
  duplicates, which PeakFit does not have.
- **Verdict:** keep it as the *validation* of whatever model PeakFit adopts (C6), and
  as downstream practice for programs fitting exchange models to intensity profiles.
  Adding a PeakFit feature for it needs its own concrete workflow (decision O7).

## 7. Evidence

The numbers come from exploratory scripts run on 2026-11-04. They are not in the
repository. An implementation turns them into the C1–C6 tools.

### 7.1 Synthetic coverage

The set-up:
- a 1D line through an exact processing chain: 92 complex points, sine-bell² (off
  0.4, end 0.98), zero filled to 256 (the CEST ¹⁵N chain);
- a 13-point fitting window, peak SNR 30;
- the exact processed lineshape fitted, with amplitude, position and R2 free;
- 1,000 realizations.

Each cell gives the observed scatter / median stated error, then the 68% · 95%
coverage of the truth. The coverage SE is 0.015 (68%) and 0.007 (95%).

| Case | Error | Amplitude | Position | R2 |
|---|---|---|---|---|
| Narrow line (R2·aq 1.3) | σ-only | 1.62 · 0.47 · 0.79 | 1.36 · 0.57 · 0.85 | 1.38 · 0.54 · 0.84 |
| | ESS factor, 12 lags | 0.99 · 0.70 · 0.95 | 0.83 · 0.80 · 0.98 | 0.84 · 0.76 · 0.98 |
| | **Sandwich** | **1.02 · 0.69 · 0.94** | **0.99 · 0.71 · 0.95** | **1.01 · 0.67 · 0.95** |
| | GLS (own errors) | 1.04 · 0.68 · 0.93 | 1.02 · 0.68 · 0.94 | 1.04 · 0.67 · 0.94 |
| Broad line (R2 ×4) | σ-only | 1.62 · 0.46 · 0.78 | 1.59 · 0.50 · 0.79 | 1.53 · 0.48 · 0.80 |
| | **Sandwich** | **1.01 · 0.67 · 0.95** | **0.99 · 0.73 · 0.94** | **0.99 · 0.68 · 0.96** |
| Zero fill ×2, no apodization | σ-only | 0.98 · 0.68 · 0.95 | 0.99 · 0.69 · 0.95 | 1.01 · 0.68 · 0.94 |
| | Sandwich | 0.98 · 0.68 · 0.95 | 0.99 · 0.69 · 0.95 | 1.01 · 0.68 · 0.94 |
| Narrow + unmodelled 0.5σ baseline | Sandwich | 1.03 · 0.49 · 0.84 | 0.98 · 0.71 · 0.95 | 1.03 · 0.54 · 0.88 |

**Reading the table:**
- **Per-parameter inflation:** for the narrow line it is 1.59 (amplitude), 1.38
  (position) and 1.36 (R2), and the sandwich matches each one. The global factor fits
  the amplitude and over-covers the others.
- **No correlation, no change:** unapodized ×2 zero fill leaves the points
  independent, and the sandwich reproduces the σ-only errors.
- **Bias stays bias:** with the baseline, the sandwich still matches the scatter. The
  coverage loss is bias (0.94 SD in amplitude), which no error model represents, as
  for misfit in ADR 0001.

### 7.2 GLS under misfit

Same chain and narrow line, 600 realizations. Each cell gives the bias in units of
that estimator's own SD. The last column is GLS SD / least-squares SD.

| Model fitted | Height: LS · GLS | Position: LS · GLS | GLS / LS SD (height, position) |
|---|---|---|---|
| Lorentzian, apodization not modelled | +0.99 · −15.9 | +0.71 · +15.7 | 0.59, 0.29 |
| Gaussian | +0.23 · −2.46 | −0.05 · +2.52 | 0.89, 0.73 |
| Exact shape, apodization offset misstated (0.45 for 0.4) | +0.05 · −0.24 | −0.01 · −0.01 | 0.90, 0.63 |

GLS's smaller SD does not compensate: in absolute terms its Lorentzian height bias is
about 9 least-squares SD, against 1.

### 7.3 Reduced χ² under correlated noise

Same synthetic narrow line, correct σ and model, 13 points and 3 parameters:

| | Value |
|---|---:|
| Nominal degrees of freedom | 10 |
| tr((I − H)R) | 6.68 |
| Observed mean χ² | 6.54 |
| ν_eff | 3.7 |
| Observed variance of χ² | 27.1 |
| Predicted variance, 2 tr(((I − H)R)²) | 24.1 |
| Variance if points were independent | 20 |

**Correlation lowers the nominal χ²ᵣ** (0.65 here) and widens its spread. In the
validation study's larger window it stayed near 1 (0.98). How much it drops depends
on the window size relative to the number of parameters.

On the CEST example:
- nominal χ²ᵣ: median 2.00 (IQR 1.70–2.48);
- against its correlated-noise expectation: median 2.25 (IQR 1.95–2.83);
- ν_eff / nominal degrees of freedom: 0.25.

The CEST χ²ᵣ ≈ 2 is therefore not explained by correlation. It points to misfit or
local σ.

### 7.4 The bundled CEST example

There are 166 peaks, fitted with `sp2` × `sp2` by `peakfit fit` at commit 889e09d.
Error ratios are relative to the σ-only error.

| | Median | IQR | Kind |
|---|---:|---|---|
| Sandwich, R from the processing | 2.335 | 2.325–2.344 | prediction under the stated noise model |
| Sandwich, R from the measured autocorrelation | 2.40 | 2.39–2.41 | prediction, R measured |
| GLS error | 2.16 | 2.14–2.19 | prediction under the stated noise model |
| Successive-difference spread (validation tool) | 2.72 | 2.53–3.00 | measurement, not an error estimate |

Sandwich inflation of the nonlinear errors:

| Parameter | Median | IQR |
|---|---:|---|
| ¹⁵N chemical shift | 2.01 | 1.96–2.03 |
| ¹⁵N linewidth | 1.68 | 1.60–1.78 |
| ¹H chemical shift | 2.32 | 2.22–2.37 |
| ¹H linewidth | 1.79 | 1.67–1.91 |

**Reading the tables:**
- **What the sandwich predicts:** under the processing-derived noise model, the
  least-squares errors on this series are 2.34× the σ-only errors for amplitudes, and
  1.7–2.3× for nonlinear parameters.
  - The factor is nearly uniform across peaks (IQR ±0.5%).
  - Using the measured autocorrelation instead changes it by 3%.
- **What it does not show:** that 2.34× is the true calibration error here, or that
  correlated noise explains the 2.72 successive-difference spread.
  - That spread mixes genuine CEST profile variation with noise (§6.4), so neither
    the gap between 2.34 and 2.72 nor their ratio is attributable to anything.
  - Validating the prediction needs replicates (C6), which this series lacks.

### 7.5 MCMC likelihood

This is an exact posterior on an 81 × 81 grid of (position, R2) per realization, with
the amplitude integrated as in [mcmc-formulation.md](mcmc-formulation.md) §2. No
sampler is involved, so only the likelihood is tested. Single peak, 600 realizations;
the coverage SE is 0.019 (68%) and 0.009 (95%). Each cell gives the 68% · 95% coverage
of the truth.

| Line | Likelihood | Position | R2 | Amplitude |
|---|---|---|---|---|
| Narrow | Profile, today | 0.48 · 0.82 | 0.53 · 0.86 | 0.45 · 0.79 |
| | Profile, vertical adjustment (k = 0.53) | 0.64 · 0.95 | 0.68 · 0.96 | 0.62 · 0.94 |
| | Gaussian with R (GLS) | 0.65 · 0.94 | 0.68 · 0.95 | 0.69 · 0.95 |
| Broad | Profile, today | 0.41 · 0.76 | 0.50 · 0.79 | 0.45 · 0.79 |
| | Profile, vertical adjustment (k = 0.40) | 0.64 · 0.94 | 0.72 · 0.96 | 0.68 · 0.95 |
| | Gaussian with R (GLS) | 0.66 · 0.95 | 0.69 · 0.96 | 0.68 · 0.95 |

- **Today's likelihood under-covers** badly.
- **The vertical adjustment** restores nominal coverage within about two SE, except
  narrow-line amplitude at 68% (0.62, three SE low). That is consistent with its single
  k being set by position and R2 (1.37) while the amplitude needs 1.59.
- **The horizontal adjustment** removes that compromise by construction. It was not
  run here. C5 must test it, and test it with the real sampler.

## 8. Limitations of this assessment

- **Synthetic scope:** the synthetic tests are 1D, with single peaks and the exact
  processed lineshape. 2D clusters and overlapping pairs enter only through the CEST
  example, where the truth is unknown.
- **Unmodelled processing:** the model covers thermal noise through the processing the
  header describes. Beyond lag 1 the measured curves depart slightly from the
  prediction: ¹⁵N lag 2 is larger, and ¹H lags 3–4 are smaller in magnitude. Residual
  signal in the noise mask is a candidate cause, but it has not been established.
- **No sampler:** MCMC was not run under any adjustment.
- **No implementation:** nothing here changes code, formats or the published results.

## 9. Open decisions

- **O1 Adopt the approach:** least squares kept, R from the processing, sandwich
  errors, adjusted MCMC likelihood. Recommended.
- **O2 Axes the model cannot describe:** an unsupported apodization, unrecorded
  processing or non-NMRPipe input. Options:
  - (a) Estimate ρ from noise-only points of the spectrum, as the validation tool
    does. It is general, but biased slightly by residual signal, and noisy for short
    axes.
  - (b) Publish σ-only errors with a recorded "noise correlation not modelled"
    warning.
  - (c) Refuse to publish errors.

  Recommended: (a), with the source recorded, and the predicted and measured curves
  compared whenever both exist.
- **O3 Independent-point errors:** keep them as a secondary field, or drop them as
  ADR 0001 did for the rescaled ones? Recommended: drop them, for one convention.
- **O4 χ²ᵣ normalization:** correlated-noise expectation with ν_eff (recommended), or
  today's nominal degrees of freedom with an explanation.
- **O5 MCMC adjustment:** horizontal (recommended, exact in the quadratic regime) or
  vertical (simpler, one scalar, compromised when the inflations differ).
- **O6 GLS:** never, or a later opt-in for exact processed lineshapes once misfit
  diagnostics can gate it? Recommended: not now.
- **O7 Empirical scatter:** add a PeakFit report of duplicate-plane or flat-region
  scatter as a cross-check? Only with a concrete workflow, such as CEST series with
  declared reference planes.
- **O8 Is a new ADR needed?** This would supersede part of ADR 0001's "Consequences"
  and MCMC decision D3. Recommended: ADR 0002, written when O1–O5 are settled.

## 10. Separate question: local or per-plane noise level

**This document does not change σ.** Cov = σ²R separates the noise level σ from the
correlation R, so the two are independent decisions.

**What the evidence says about σ:**
- On the CEST example, per-plane noise is 1.03–1.06 × the global estimate.
- χ²ᵣ ≈ 2.25 against its correlated-noise expectation points to misfit, to noise
  near peaks larger than σ (t1 noise), or to other unmodelled contributions. Without
  replicates, the data do not separate them.

**Whether to estimate σ per plane or locally** belongs to its own assessment:
- t1 noise is neither white nor stationary, so it would also need a local R;
- it would need its own validation against duplicate planes.

Nothing in §5 presumes the answer.

## References

- A. C. Aitken (1935). On least squares and linear combination of observations.
  *Proceedings of the Royal Society of Edinburgh* 55, 42–48.
- E. Bartholdi and R. R. Ernst (1973). Fourier spectroscopy and the causality
  principle. *Journal of Magnetic Resonance* 11, 9–19.
- G. V. Bayley and J. M. Hammersley (1946). The "effective" number of independent
  observations in an autocorrelated time series. *Journal of the Royal Statistical
  Society Supplement* 8, 184–197.
- S. Bourg and J.-M. Nuzillard (1998). Influence of noise on peak integrals obtained
  by direct summation. *Journal of Magnetic Resonance* 134, 184–188.
- G. L. Bretthorst (1990). Bayesian analysis. I. Parameter estimation using quadrature
  NMR models. *Journal of Magnetic Resonance* 88, 533–551.
- R. E. Chandler and S. Bate (2007). Inference for clustered data using the
  independence loglikelihood. *Biometrika* 94, 167–183.
- A. Ebel, W. Dreher and D. Leibfritz (2006). Effects of zero-filling and apodization
  on spectral integrals in discrete Fourier-transform spectroscopy of noisy data.
  *Journal of Magnetic Resonance* 182, 330–338.
- P. J. Huber (1967). The behavior of maximum likelihood estimates under nonstandard
  conditions. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics
  and Probability* 1, 221–233.
- K. Landheer and C. Juchem (2021). Are Cramér–Rao lower bounds an accurate estimate
  for standard deviations in in vivo magnetic resonance spectroscopy? *NMR in
  Biomedicine* 34, e4521.
- M. Ribatet, D. Cooley and A. C. Davison (2012). Bayesian inference from composite
  likelihoods, with an application to spatial extremes. *Statistica Sinica* 22,
  813–845.
- F. E. Satterthwaite (1946). An approximate distribution of estimates of variance
  components. *Biometrics Bulletin* 2, 110–114.
- H. White (1980). A heteroskedasticity-consistent covariance matrix estimator and a
  direct test for heteroskedasticity. *Econometrica* 48, 817–838.
