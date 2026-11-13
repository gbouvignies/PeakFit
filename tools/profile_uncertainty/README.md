# Evidence for docs/profile-uncertainty.md

These are the exploratory studies behind
[`docs/profile-uncertainty.md`](../../docs/profile-uncertainty.md): the choice of
estimator, the uncertainty of intensity profiles, and their use downstream.
- **Seeds:** every study uses a fixed seed.
- **Records:** their outputs are in
  [`docs/validation/profile-uncertainty/`](../../docs/validation/profile-uncertainty/).
- **Status:** they are evidence, not tests. The implementation's validation tools
  (T3–T7 in the specification) supersede them.

```bash
uv run python tools/profile_uncertainty/run_all.py              # every study
uv run python tools/profile_uncertainty/run_all.py cest_refit   # one study
```

## Studies

| Script | Specification | What it shows |
|---|---|---|
| `estimators.py` | §2.1 | OLS, shrinkage and GLS on one processed axis: exact model, misfit and wrong R |
| `cest_conditioning.py` | §2.1 | Size and conditioning of R on the example's clusters |
| `cest_first_order.py` | §2.4, §3, §6a | First-order GLS/OLS SDs; marginal/conditional σ; correlated modes of Σ_rel; amplitude correlation with a cluster partner |
| `linear_prediction.py` | §2.2 | LP on the ¹⁵N axis, detected from the time-domain noise power; lag correlations |
| `cest_refit.py` | §2.3, §5.2 | GLS and shrinkage refits: whitened χ²ᵣ, parameter shifts and residual excess |
| `cest_replicates.py` | §2.3, §5 | CEST baseline planes as replicates: real scatter and calibration of each estimator |
| `formula_check.py` | §3.1 | The covariance formula over all parameters against the full-Jacobian sandwich |
| `curvature_term.py` | §3.1 | The factors' C_θ against the published lineshape covariance, which carries a residual term; effect on marginal and profile σ |
| `ratios.py` | §4 | Ratio errors over 5 planes: conditional, marginal, full covariance |
| `downstream_decay.py` | §4 | A decay fitted with a free scale: which per-plane σ is calibrated |
| `global_cpmg.py` | §6a | A ChemEx-like global CPMG fit of an overlapping pair |

## Shared modules

- `processed_lines.py`: synthetic processed lines and their derivatives. It reuses
  the processing chain of `tools/validate_correlated_noise.py`.
- `cest_example.py`: the bundled CEST example, clustered and fitted as PeakFit does
  it.

## Limits

- **One dataset:** the real-data studies use only the bundled CEST example, whose
  ¹⁵N axis was linearly predicted.
- **Synthetic scope:** the synthetic studies use one processed axis.
- **First order:** the CEST first-order results are evaluated at the OLS optimum.
