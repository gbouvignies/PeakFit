# MCMC formulation

**Status:** specification. Code follows only after this is agreed:
- the sampling problem and likelihood (PR 22);
- the sampler and its diagnostics (PR 23);
- the stored record (PR 24).

Today's `peakfit mcmc` ([mcmc.md](mcmc.md)) predates it, and §8 lists where they
differ.

This document fixes what PeakFit's MCMC samples, what it reports and how that is
validated. The aim is credible intervals for the published quantities, above all the
amplitudes that make up intensity profiles. Those intervals should include the
lineshape uncertainty the analytical errors leave out, and nothing should depend on
an arbitrary convention.

## 1. Model

For one peak cluster:

- **Y**: corrected data, points × S planes;
- θ: the nonlinear parameters the final fit varied, within its bounds, including
  cluster-scoped phases;
- Φ(θ): shapes, peaks × points;
- **A**: amplitudes, peaks × S;
- σ: the recorded statistical noise (`FinalFitOutcome.noise`).

**Y** = Φ(θ)ᵀ **A** + **E**, where **E** has independent N(0, σ²) entries. The
cross-talk corrections and every parameter the fit held fixed are frozen at their
terminal values, and the lineshape model is the fit's own.

For one plane s, with p peaks, write G(θ) = ΦΦᵀ and â_s(θ) = G⁻¹Φ y_s. Then

χ²(θ, **A**) = Σ_s ‖y_s − Φᵀa_s‖² / σ² = χ²_min(θ) + Σ_s (a_s − â_s)ᵀ (G/σ²) (a_s − â_s),

where χ²_min(θ) is the squared norm of the VarPro residual divided by σ². That
residual is the real part of the cluster data, projected onto the orthogonal
complement of the shapes by the same rank-aware `factorize_shapes` the fit uses.

## 2. Three candidate formulations

**(a) Profile likelihood.** L_p(θ) = exp(−½ χ²_min(θ)). Amplitudes are replaced by
their conditional optimum. This is what the current code samples.

**(b) Flat-prior marginal.** With π(**A** | θ) ∝ 1, integrating each plane's
Gaussian in a_s gives (2πσ²)^(p/2) det G^(−1/2), so

L_m(θ) = ∫ L dA ∝ L_p(θ) · det G(θ)^(−S/2).

**(c) Conditional Jeffreys prior on the amplitudes.** The Fisher information of a_s
given θ is G/σ², so the Jeffreys prior for all S planes is
π(**A** | θ) ∝ det G(θ)^(S/2). The same integral now cancels the determinant:

∫ L(θ, **A**) π(**A** | θ) d**A** = (2πσ²)^(pS/2) · L_p(θ).

**So (a) is exactly the integrated likelihood under (c).** The joint posterior then
factorizes:

- θ ~ π(θ) L_p(θ);
- given θ, the planes are independent, and **a**_s | θ ~ N(â_s(θ), σ² G(θ)⁻¹).

This is the integrated-likelihood approach of Berger, Liseo and Wolpert (1999). The
amplitude prior depends on θ. That is a stated reference-prior choice, not an
"uninformative" one.

## 3. Why not the flat-prior marginal

1. **It depends on an arbitrary convention.** If the shapes are rescaled by a
   θ-dependent factor, for example height-normalized versus area-normalized,
   det G and with it the θ posterior change. L_p does not change: χ²_min depends only
   on the span of the shapes.
2. **Its bias does not shrink with more planes.** Each plane adds p amplitudes, so
   the det G^(−S/2) term grows ∝ S, like the information. Relative to the posterior
   width its effect grows, roughly as √S / SNR. Pseudo-3D series often have tens to
   hundreds of planes.
3. **It rewards unidentifiable configurations.** As shapes become collinear,
   det G → 0 and L_m diverges. Coalescing peaks are exactly where an
   overlapping-peak tool must behave best.

### Evidence

`tools/validate_mcmc_formulation.py` simulates one Lorentzian with an unknown
linewidth (true value 6 points), S planes with equal amplitudes, known σ = 1 and a
uniform prior on 3–10 points. It uses 1,000 noise realizations per row, seed
20261104. The results are in
[`validation/mcmc-formulation.json`](validation/mcmc-formulation.json). Each cell
gives the bias of the posterior mean in posterior standard deviations, then the
coverage of the central 68% and 95% intervals. Coverage standard errors are 0.015
(68%) and 0.007 (95%).

| S | Peak SNR | Profile (a/c) | Flat, height-normalized | Flat, area-normalized |
|---:|---:|---|---|---|
| 1 | 5 | +0.33 · 0.686 · 0.955 | +0.24 · 0.698 · 0.954 | +0.41 · 0.662 · 0.949 |
| 1 | 20 | +0.10 · 0.664 · 0.936 | +0.08 · 0.665 · 0.936 | +0.12 · 0.664 · 0.935 |
| 10 | 5 | +0.14 · 0.672 · 0.955 | −0.15 · 0.674 · 0.944 | +0.43 · 0.642 · 0.931 |
| 10 | 20 | +0.04 · 0.689 · 0.963 | −0.03 · 0.678 · 0.962 | +0.11 · 0.693 · 0.963 |
| 50 | 5 | +0.06 · 0.689 · 0.940 | −0.60 · 0.583 · 0.896 | +0.70 · 0.575 · 0.893 |
| 50 | 20 | +0.02 · 0.675 · 0.950 | −0.14 · 0.666 · 0.944 | +0.19 · 0.659 · 0.949 |

**Reading the table:**
- With one plane, all three behave alike. The residual bias they share there
  reflects the nonlinearity and the prior's bounded support at low SNR, not the
  formulation.
- With 50 planes at SNR 5, the flat marginal biases the linewidth by about 0.6–0.7
  posterior standard deviations, in opposite directions for the two
  normalizations. Its 68% coverage falls to about 0.58, six to seven standard
  errors below nominal.
- The profile stays within about two standard errors of nominal coverage at both
  levels.

The tool also checks the derivation numerically:
- On a θ grid, the Jeffreys-weighted amplitude integral divided by L_p is constant
  to a relative spread of 5 × 10⁻¹⁴, and equals the predicted √(2π)·σ.
- The profile log posterior is the same for height- and area-normalized shapes to
  7 × 10⁻¹³.

```bash
uv run python tools/validate_mcmc_formulation.py --output /tmp/mcmc-formulation.json
```

## 4. What the sampler targets

- **Objective:** log π(θ | Y) = −½ χ²_min(θ) + log π(θ). The residual is exactly
  the fit's VarPro residual, so the parent fit's χ² at θ̂ is reproduced by
  construction.
- **θ prior:** uniform within the final fit's bounds, so the bounds are the prior's
  support. The fraction of samples piled up near each bound is reported.
- **Fixed parameters** stay fixed, and cluster-scoped phases are sampled whenever
  the fit varied them.
- **Noise:** the result is conditional on the recorded σ. It is not rescaled by
  χ²_red; χ²_red is reported alongside, as for the fit.
- **Corrections and model:** results are conditional on the frozen neighbour
  corrections and the chosen lineshapes.
- **Label switching:** if the position bounds of equivalent peaks overlap, peaks can
  exchange identities between samples. This is detected and warned about, not
  silently relabelled.

## 5. Amplitude uncertainties

- **Draws:** for each kept θ sample k, draw **a**ₖ,s ~ N(â_s(θₖ), σ² G(θₖ)⁻¹) for
  every plane, using the covariance `factorize_shapes` already provides.
- **Moments:** use the law of total variance. The mean is the average of â(θₖ), and
  the covariance is the mean of the conditional covariances plus the covariance of
  â(θₖ) across samples. This is exact and has a lower Monte Carlo error than moments
  of the draws.
- **Intervals and ratios:** quantiles and per-sample ratios (for example, a plane's
  amplitude relative to a reference plane) come from the draws.
- **Labelling:** MCMC amplitude intervals *include lineshape uncertainty*. The
  published analytical errors are *conditional on the fitted shapes* (see
  **Amplitude correlation** in the glossary). MCMC widths should therefore not be
  smaller than the analytical ones.

## 6. Rank deficiency

The θ density stays finite: L_p uses the rank-aware projection, with the same rank
tolerance as the fit.

Where Φ(θ) is rank-deficient, the conditional amplitude distribution is improper
along its null space. The peaks involved are reported as "amplitude not
identifiable in X% of samples". Their amplitude summaries use only the identifiable
samples, and that fraction is stated. Nothing is silently regularized, and no NaN
becomes a number.

## 7. Diagnostics for an ensemble sampler

The sampler stays emcee's affine-invariant ensemble (Goodman and Weare 2010;
Foreman-Mackey et al. 2013).

- **No R̂ across walkers.** Ensemble walkers are coupled, not independent chains,
  so R̂ computed across them is not meaningful. The report reads "R̂: not assessed"
  (decision D2).
- **Autocorrelation:** the integrated autocorrelation time τ of each parameter,
  from `emcee.autocorr.integrated_time` with its default tolerance. If the estimate
  is unreliable, that is reported rather than suppressed.
- **Convergence:**
  - The kept length must be at least 50 τ_max.
  - A shorter run is labelled "chain too short — provisional", and no final
    intervals are given.
- **Burn-in:** a recorded multiple of τ_max, or an explicit user value. Either way,
  it must leave at least 50 τ_max, and it is discarded exactly once. Statistics use
  no thinning.
- **Effective sample size:** ESS ≈ walkers · N / τ, labelled approximate. The Monte
  Carlo standard errors of means and quantiles are reported with it.
- **Acceptance:** reported for the whole ensemble and per walker. Stuck walkers are
  flagged with a labelled heuristic.
- **Correlations:** a correlation that cannot be computed is reported as undefined,
  never as 0.
- **Large clusters:** the stretch move mixes poorly in high dimensions, which τ
  reveals. Changing samplers is not planned until τ shows a need.

## 8. Changes from today's implementation

| Today | This specification |
|---|---|
| Profile likelihood sampled, but amplitudes are the conditional optimum â(θ), so intensity intervals omit σ²G⁻¹ and are too narrow | Same θ target, now justified by the conditional Jeffreys prior; amplitudes drawn from N(â, σ²G⁻¹) |
| Split R̂ and summed per-walker ESS across coupled walkers | No R̂ across walkers; ESS from τ; 50 τ rule |
| `get_autocorr_time(tol=0)` hides an unreliable τ; a burn-in longer than the chain silently keeps the last 10% | τ reliability reported; burn-in validated against the 50 τ rule |
| NaN correlations replaced by 0 | Undefined correlations reported as undefined |
| Chains stored in HDF5 after burn-in, with burn-in recorded again | A record bound to its parent run (PR 24, D4); burn-in discarded once |

Today's code already samples fitted phases; that fix is among the unreleased changes.

## 9. Validation plan

| Test | What it checks | PR |
|---|---|---|
| V1 | log L(θ̂) equals −½χ² as recorded, to rtol 10⁻¹⁰, for a single peak, a phase-fitted cluster and a multi-plane cluster | 22 |
| V2 | At fixed θ, conditional draws match the closed-form mean and covariance, within a tolerance justified by the number of draws | 22 |
| V3 | One peak, S = 1: quadrature over A with the Jeffreys prior equals L_p up to a constant across θ, in code; this document's tool already shows it | 22 |
| V4 | The driver on correlated and truncated Gaussians with known moments: means and standard deviations within Monte Carlo error; identical chains for the same seed | 23 |
| V5 | Linear regime: θ covariance ≈ VarPro's `identifiable_covariance`; amplitude covariance ≈ an independently computed joint inverse Fisher matrix, and larger than the conditional errors | 23 |
| V6 | Slow marker: 68% and 95% coverage over about 200 synthetic realizations of one peak and of two overlapping peaks, within binomial bounds | 23 |
| V7 | A duplicated peak: amplitudes reported as not identifiable, and the θ density still finite | 22 |
| V8 | A parameter near a bound: pile-up reported, and no walker starts on the bound | 22, 23 |
| V9 | Phase-fitted clusters include their cluster-scoped parameters | 22 |

## 10. Decisions

Merging this specification records these decisions (`decisions.md`, 2026-11-04):

- **D1 Formulation:** the profile likelihood, justified by the conditional
  Jeffreys prior, with conditional amplitude draws. The flat-prior marginal is
  rejected for the reasons in §3.
- **D2 R̂:** one ensemble, with no R̂ across walkers. The alternative was several
  independent seeded ensembles with rank-normalized R̂ (Vehtari et al. 2021), at
  several times the cost. It remains possible later without changing the target.
- **D3 Noise:** results are conditional on the recorded σ, with no χ² rescaling.
- **D4 Storage:** store θ chains and log-probabilities losslessly, plus amplitude
  summaries. Amplitude draws are regenerated from the recorded seed when the
  observations are available. Storing every draw can reach hundreds of MB: for
  example, 32 walkers × 5,000 steps × 5 peaks × 50 planes ≈ 320 MB in float64.

## References

- J. O. Berger, B. Liseo and R. L. Wolpert (1999). Integrated likelihood methods for
  eliminating nuisance parameters. *Statistical Science* 14(1), 1–28.
- J. Goodman and J. Weare (2010). Ensemble samplers with affine invariance.
  *Communications in Applied Mathematics and Computational Science* 5(1), 65–80.
- D. Foreman-Mackey, D. W. Hogg, D. Lang and J. Goodman (2013). emcee: the MCMC
  hammer. *Publications of the Astronomical Society of the Pacific* 125, 306–312.
- A. Vehtari, A. Gelman, D. Simpson, B. Carpenter and P.-C. Bürkner (2021).
  Rank-normalization, folding, and localization: an improved R̂ for assessing
  convergence of MCMC. *Bayesian Analysis* 16(2), 667–718.
