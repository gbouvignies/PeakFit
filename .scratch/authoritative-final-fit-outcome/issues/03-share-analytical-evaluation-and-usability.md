# 03 — Share analytical evaluation and usability

**What to build:** One numerical operation that evaluates analytical amplitudes,
model values, residuals, and numerical usability consistently for optimizer
passes and final outcomes.

**Blocked by:** 01 — Characterize competing fit truth.

**Status:** ready-for-agent

- [ ] Consolidate the existing QR-based analytical amplitude work without
      changing solver mathematics, stopping behavior, rank-deficiency fallback,
      or tolerances.
- [ ] Return either a coherent usable evaluation or an explicit unusable reason.
- [ ] Keep optimizer convergence status separate from numerical usability and
      derive the three classifications: converged, usable non-converged, and
      unusable.
- [ ] Classify an outcome as unusable when required nonlinear values,
      amplitudes, model values, residuals, cost, or chi-squared are non-finite
      or cannot be evaluated.
- [ ] Use the shared operation or its single numerical primitive in VARPRO and
      basin hopping.
- [ ] Prove deliberately stale injected amplitude parameters do not become
      amplitude authority.
- [ ] Preserve the strict unequal point/series contract and evaluated shapes.
- [ ] Exercise rank-deficient design-matrix behavior with existing warnings,
      fallbacks, and tolerances.
- [ ] Preserve characterized VARPRO and fixed-seed basin-hopping results within
      existing tolerances.
- [ ] Prove a termination message cannot override convergence status while a
      finite non-converged result may still be classified as usable.
- [ ] Verify with
      `uv run pytest -q -p no:cacheprovider -k "analytical_model_evaluation or numerical_usability or varpro or basin_hopping"`.
