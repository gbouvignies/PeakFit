# 03 — Share analytical evaluation and usability

**What to build:** One numerical operation that evaluates analytical amplitudes,
model values, residuals, and numerical usability consistently for optimizer
passes and final outcomes.

**Blocked by:** 01 — Characterize competing fit truth.

**Status:** resolved

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

## Comments

### 2026-07-26 — Shared analytical evaluation and usability complete

- Added a typed analytical evaluation that re-solves amplitudes with the
  existing QR-based operation and returns model values, raw and normalized
  residuals, amplitude uncertainty inputs, and one typed statistics record.
- Added independent three-state classification for converged, usable
  non-converged, and unusable optimizer outcomes. Identity, amplitude count,
  shapes, nonlinear parameters, optimizer residual and cost, analytical
  amplitudes, model values, residuals, chi-squared, and derived values must be
  compatible and finite.
- Pipeline parameter merging and cross-talk contribution now admit only usable
  cluster outcomes. Unusable results remain present with their optimizer result,
  cluster identity, and explicit reason.
- Basin hopping now uses the shared terminal analytical evaluation and no longer
  promotes convergence from termination-message text. VARPRO retains its
  characterized QR projection mathematics and agrees with the shared final
  evaluation within existing tolerances.
- Durable result reconstruction, CLI review, summaries, persistence, writers,
  simulation, final correction freezing/revisions, and `FinalFitOutcome` remain
  deferred to their assigned tickets.
- Validation passed: focused ticket-03 selection (`20 passed`); combined
  ticket-01/02 characterization and ticket-03 tests (`36 passed, 6 xfailed`);
  full headless suite (`163 passed, 6 xfailed`); Ruff lint and format checks;
  `ty check --error-on-warning`; import contracts (`2 kept, 0 broken`);
  `git diff --check`; Graphify update; and separate Standards/Spec reviews with
  no remaining hard or spec findings.
