# MCMC formulation

**Status:** specification. Code follows only after this is agreed:
- the sampling problem and likelihood (PR 22);
- the sampler and its diagnostics (PR 23);
- the stored record (PR 24).

Today's `peakfit mcmc` ([mcmc.md](mcmc.md)) predates it, and §9 lists where they
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

**Scope:** this study supports the choice of formulation for one isolated peak with
independent amplitudes. It says nothing about coverage for overlapping peaks, where
near-degenerate shapes (§6), multimodality and label switching (§7) dominate. That
case is validated separately (V6, §10), and passing it is an acceptance criterion
for the implementation.

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

## 5. Amplitude uncertainties

- **Draws:** for each kept θ sample k, draw **a**ₖ,s ~ N(â_s(θₖ), σ² G(θₖ)⁻¹) for
  every plane, using the factorization `factorize_shapes` already provides. Draws
  are made only where §6 allows them.
  - To bound the cost, at most 4,000 kept samples, evenly spaced, are evaluated,
    with one draw each. Their Monte Carlo errors use the smaller of that count and
    the sampler's smallest ESS.
  - The evaluated samples are balanced across walkers, evenly spaced in time within
    each walker, so no walker is under-represented.
  - Whether a single peak is ever undetermined is still judged on every kept sample,
    which the sampler records. A summed amplitude is judged at the evaluated samples
    only, and is reported as determined at all of them; judging it everywhere would
    cost the full evaluation the cap avoids.
- **Moments:** use the law of total variance. The mean is the average of â(θₖ), and
  the covariance is the mean of the conditional covariances plus the covariance of
  â(θₖ) across samples. This is exact and has a lower Monte Carlo error than moments
  of the draws.
- **Intervals and ratios:** quantiles and per-sample ratios (for example, a plane's
  amplitude relative to a reference plane) come from the draws.
- **What the intervals mean:** MCMC amplitude intervals *include lineshape
  uncertainty*, averaged over the θ posterior. The published analytical errors are
  *conditional on the fitted shapes*, evaluated at θ̂ only (see **Amplitude
  correlation** in the glossary).
- **Comparing the two:** MCMC intervals are usually wider than the analytical
  errors, but they need not be.
  - The total variance adds Var_θ(â) to E_θ[σ²G(θ)⁻¹]. That average conditional
    covariance can be *smaller* than σ²G(θ̂)⁻¹ when posterior mass lies where shapes
    are better separated or narrower than at θ̂.
  - In the linear, high-SNR regime the MCMC amplitude covariance approaches the
    amplitude block of the joint inverse Fisher matrix, which is never smaller than
    the conditional covariance at θ̂. That inequality is a property of that regime,
    not a general rule.
  - A narrower MCMC interval is reported as computed, never flagged as an error or
    widened.

## 6. Rank deficiency and near-degeneracy

Overlapping peaks are central to PeakFit, so three situations are kept apart. Each
is judged on the singular values s₁ ≥ … ≥ s_p of the cluster's shape matrix at a
sampled θ. Both thresholds are the ones `factorize_shapes` already applies to the
fit.

| Situation | Criterion at a sample θ | Meaning |
|---|---|---|
| Numerical rank deficiency | s_j ≤ ε·n·s₁ (n = largest shape-matrix dimension) | Shapes coincide to machine precision. In practice this happens only structurally, for duplicated peaks or constraints that make shapes identical. |
| Unreliable conditioning | full rank, but s_p ≤ √ε·s₁ | The conditional covariance exists but cannot be computed reliably; the fit already withholds its covariance here. |
| Practical non-identifiability | well conditioned, but member amplitudes strongly correlated | The data determine a combination of amplitudes (typically their sum) much better than how it splits between peaks. This is a statistical property, not a numerical one. |

**How the likelihood is defined in every case.** L_p(θ) = exp(−½ ‖(I − P(θ)) y‖² /
σ²), where P(θ) projects onto the numerical span of the shapes. It depends only on
that span, so it is defined at any rank.

The derivation in §2 holds on any region of θ where the numerical rank r is
constant, with G restricted to an orthonormal basis of the span and p replaced by
r. The constant (2πσ²)^(rS/2) then does not depend on θ.

The rank changes only on sets of negligible posterior probability, because the
first threshold is at machine precision. The one exception is structural
degeneracy, which holds at every θ, so its rank is constant and the derivation is
exact. The sampler therefore targets the same L_p as the fit, and nothing is
regularized.

**Samples are never dropped or reweighted.** Summarizing amplitudes only over
"identifiable" samples would condition the posterior on θ lying where the peaks
are distinct, biasing every reported quantity towards separated peaks. Instead:

1. **The θ posterior** is always summarized from every kept sample.
2. **Withheld per-peak intervals.** A peak's amplitude interval is withheld when, in
   any kept sample, its conditional distribution is not computable. Two cases count:
   - **Null directions.** These are the right singular vectors below the rank
     cutoff, plus those a reduced SVD omits when the cluster has fewer points than
     peaks. A loading above √ε on them makes the amplitude improper.
   - **Weak directions.** These are retained, with s_i ≤ √ε·s₁, but not reliably
     computable. Their contribution to peak k's variance is σ²·(v_ki / s_i)², so a
     tiny loading can still dominate. The amplitude is withheld when they carry more
     than 10⁻⁴ of its variance.

   Draws omit weak directions. For an amplitude that is kept, this understates its
   standard deviation by at most 0.005%.
   - The report states "amplitude not determined in X% of samples".
   - No interval is computed from the remaining samples.
   - The other peaks' intervals are unaffected.
   - When every weak direction is orthogonal to the summed amplitude of the peaks
     involved, as for duplicated peaks, that sum is reported with its interval.
3. **Practical non-identifiability.** Per-peak intervals are the correct marginals,
   so they are reported, but never alone. Any pair of member amplitudes with
   posterior |ρ| ≥ 0.9 in a plane is flagged "amplitudes not separately determined".
   - The 0.9 is the same threshold as the amplitude-correlation highlight, and it is
     a labelled review heuristic.
   - The summed amplitude of each correlated group is reported with its interval,
     computed from the same draws.
   - The flag stops wide, anticorrelated per-peak intervals from being read as
     independent measurements.
   - The peaks' positions and linewidths keep their intervals, with the caution
     "not separately resolved". A strong amplitude correlation alone does not show
     them to be unidentifiable. Withholding is reserved for exchanges (§7) and
     other demonstrated identifiability failures.
4. **No silent fallback.** No NaN becomes a number, and no singular covariance is
   pseudo-inverted into false precision.

## 7. Diagnostics for an ensemble sampler

The sampler stays emcee's affine-invariant ensemble (Goodman and Weare 2010;
Foreman-Mackey et al. 2013). Walkers start in a small ball around the fit's
optimum. A result can therefore describe the mode containing θ̂ well and still miss
a distant one, and no diagnostic below can rule that out. Every report says so.

- **No R̂ across walkers.** Ensemble walkers are coupled, not independent chains,
  so R̂ computed across them is not meaningful. The report reads "R̂: not assessed"
  (decision D2).
- **Autocorrelation:** the integrated autocorrelation time τ of each parameter,
  from `emcee.autocorr.integrated_time` with its default tolerance. If the estimate
  is unreliable, that is reported rather than suppressed.
- **Minimum length:** the kept length must be at least 50 τ_max. This is the
  minimum for τ, and the error estimates built on it, to be reliable. It does not
  prove convergence or complete exploration.
- **Burn-in:** a recorded multiple of τ_max, or an explicit user value. Either way
  it must leave at least 50 τ_max, and it is discarded exactly once. Statistics use
  no thinning.
- **Effective sample size:** ESS ≈ walkers · N / τ, labelled approximate. The Monte
  Carlo standard errors of means and quantiles are reported with it.
- **Stationarity:** the kept samples are split into first and second halves in
  time, pooling all walkers. Each parameter's median is compared between halves,
  and a difference larger than three combined Monte Carlo standard errors fails the
  check.
  - The error comes from the second half. The first half's own spread would include
    any drift, and so hide it.
  - A drifting ensemble is caught this way without treating walkers as independent
    chains.
- **Stuck walkers:** acceptance is reported for the whole ensemble and per walker.
  A walker accepting under a tenth of the ensemble's median rate is flagged. This
  is a labelled heuristic.
- **Multimodality and label switching:**
  - For each pair of peaks whose position bounds overlap, the report gives the
    fraction of samples in which their order changes along the axis on which they
    are best separated at θ̂.
  - Any exchange withholds both peaks' per-peak position and amplitude intervals,
    as in §6. The pair's summed amplitude remains available.
  - Bound pile-up (§4) is reported per parameter.
- **Correlations:** a correlation that cannot be computed is reported as undefined,
  never as 0.
- **One status, with reasons.** A run is **provisional** if any of these hold:
  - the kept length is under 50 τ_max;
  - τ is unreliable;
  - the stationarity check fails;
  - a walker is stuck;
  - peaks exchange.

  Provisional results list their reasons and give no final intervals. A run that
  passes is labelled "passed the convergence checks", never "converged".
- **Large clusters:** the stretch move mixes poorly in high dimensions, which τ
  reveals. Changing samplers is not planned until τ shows a need.

## 8. Reproducibility and storage (D4)

Each run writes one immutable record, `<fit run>/mcmc/<record id>.npz`
([outputs.md](outputs.md#mcmc-records)). Amplitude draws are not stored. They can be
regenerated exactly from:

- **The parent:** run identity and the SHA-256 of its `scientific.npz`, which fix
  the lineshape model, all parameter values (fixed ones included), bounds, frozen
  corrections, cluster membership, fitting grid and σ. The record also repeats the
  terminal correction revision and σ, and refuses to be written if the parent
  changed while it was sampled;
- **The observations:** the verified source spectrum (SHA-256), or the retained
  observations of a `--detailed-evidence` export;
- **The chain:** the whole chain, burn-in included, stored losslessly in float64
  with the burn-in index, the log-probabilities and per-sample identifiability;
- **The random streams:** one `SeedSequence` entropy value. Cluster *i* (in cluster
  order) uses `SeedSequence(entropy, spawn_key=(i,))`, which spawns a sampler child
  (walker starts: PCG64; emcee moves: MT19937) and an amplitude-draw child (PCG64),
  so regeneration does not replay the sampler. The record stores these bit
  generators and the selection rule: one draw per evaluated θ sample, at most 4,000
  kept samples, balanced across walkers and evenly spaced in time within each;
- **The procedure:** formulation version **1.0** of this specification, which
  changes whenever sampling, diagnosis or summaries change what a run produces,
  and the PeakFit, NumPy and emcee versions.

Regeneration is bit-identical with the same software versions (V11). NumPy does not
promise identical Generator streams across versions, so a different version gives
statistically equivalent draws. Linear-algebra libraries may also differ at
round-off.

**Summaries do not need the source.** The record stores, per cluster:
- θ quantiles and means;
- per peak and plane, the amplitude mean and standard deviation (§5) and the 2.5,
  16, 50, 84 and 97.5% quantiles from the draws made at sampling time;
- the summed-amplitude intervals and every §6 and §7 flag;
- all diagnostics, the status and its reasons.
- the parent cluster's χ²_red, reported alongside (§4).

These remain available when the source spectrum, or the parent itself, is missing.
Only new derived quantities, such as ratios to a reference plane that was not
summarized, need regeneration and therefore the observations.

## 9. Changes from today's implementation

| Today | This specification |
|---|---|
| Profile likelihood sampled, but amplitudes are the conditional optimum â(θ), so intensity intervals omit σ²G⁻¹ and are too narrow | Same θ target, now justified by the conditional Jeffreys prior; amplitudes drawn from N(â, σ²G⁻¹) |
| Split R̂ and summed per-walker ESS across coupled walkers | No R̂ across walkers; ESS from τ; 50 τ minimum, stationarity, stuck-walker and label-switching checks; a provisional status |
| Amplitude summaries whatever the conditioning of the shapes | Per-peak intervals withheld or flagged by the §6 policy, with summed amplitudes reported |
| `get_autocorr_time(tol=0)` hides an unreliable τ; a burn-in longer than the chain silently keeps the last 10% | τ reliability reported; burn-in validated against the 50 τ rule |
| NaN correlations replaced by 0 | Undefined correlations reported as undefined |
| Chains stored in HDF5 after burn-in, with burn-in recorded again | One record per run bound to its parent (PR 24, D4); the whole chain stored, burn-in discarded once when summarizing |

Today's code already samples fitted phases; that fix is among the unreleased changes.

## 10. Validation plan

Each implementation PR passes its tests before merging. **V6 is the acceptance
criterion for trusting MCMC intervals on overlapping peaks.** The single-peak study
in §3 is no substitute for it.

| Test | What it checks | PR |
|---|---|---|
| V1 | log L(θ̂) equals −½χ² as recorded, to rtol 10⁻¹⁰, for a single peak, a phase-fitted cluster and a multi-plane cluster | 22 |
| V2 | At fixed θ, conditional draws match the closed-form mean and covariance, within a tolerance justified by the number of draws | 22 |
| V3 | One peak, S = 1: quadrature over A with the Jeffreys prior equals L_p up to a constant across θ, in code; this document's tool already shows it | 22 |
| V4 | The driver on correlated and truncated Gaussians with known moments: means and standard deviations within Monte Carlo error; identical chains for the same seed; a deliberately too-short run and a drifting target are reported provisional | 23 |
| V5 | Linear regime: θ covariance ≈ VarPro's `identifiable_covariance`; amplitude covariance ≈ the amplitude block of an independently computed joint inverse Fisher matrix. No interval-width inequality is tested outside this regime (§5). | 23 |
| V6 | A recorded study ([validation/mcmc-coverage.md](validation/mcmc-coverage.md), `tools/validate_mcmc_coverage.py`), too slow for CI; acceptance for overlapping peaks. About 200 synthetic realizations each of one peak and of two peaks at three separations: well resolved, partially overlapping and near-coalescent. Where intervals are reported, 68% and 95% coverage of positions, linewidths, per-peak amplitudes and summed amplitudes stays within binomial bounds. Where they are withheld or flagged (§6, §7), that is checked rather than counted. Near-coalescent cases must produce the flags, not confident per-peak intervals. | 23 |
| V7 | A duplicated peak: per-peak amplitude intervals withheld, the summed amplitude reported, and the θ density finite | 22 |
| V8 | A parameter near a bound: pile-up reported, and no walker starts on the bound | 22, 23 |
| V9 | Phase-fitted clusters include their cluster-scoped parameters | 22 |
| V10 | Two equivalent peaks with overlapping position bounds: the exchange fraction is reported, and both peaks' intervals are withheld | 23 |
| V11 | Regeneration from a stored record reproduces the draws bit for bit with the same software, and the stored summaries read back without the source spectrum | 24 |

## 11. Decisions

Merging this specification records these decisions (`decisions.md`, 2026-11-04):

- **D1 Formulation:** the profile likelihood, justified by the conditional Jeffreys
  prior, with conditional amplitude draws. The flat-prior marginal is rejected for
  the reasons in §3. Degenerate and near-degenerate shapes follow §6: samples are
  never dropped, and per-peak intervals are withheld or flagged, with summed
  amplitudes reported instead.
- **D2 Convergence:** one ensemble, with no R̂ across walkers. Fifty τ_max is a
  minimum reliability criterion, not proof of convergence. The stationarity,
  stuck-walker and label-switching checks of §7 make a run provisional rather than
  trustworthy-looking. The alternative was several independent seeded ensembles
  with rank-normalized R̂ (Vehtari et al. 2021), at several times the cost. It
  remains possible later without changing the target.
- **D3 Noise:** results are conditional on the recorded σ, with no χ² rescaling.
- **D4 Storage:** store θ chains and log-probabilities losslessly, plus complete
  summaries that need no source spectrum. Amplitude draws are regenerated from the
  inputs listed in §8. Storing every draw can reach hundreds of MB: for example,
  32 walkers × 5,000 steps × 5 peaks × 50 planes ≈ 320 MB in float64.

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
