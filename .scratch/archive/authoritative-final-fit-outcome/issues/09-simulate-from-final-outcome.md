# 09 — Simulate from the final outcome

**What to build:** Optional simulated-spectrum output that uses authoritative
outcome amplitudes and a verified matching final model snapshot without solving
new amplitudes.

**Blocked by:** 06 — Migrate CLI and RunSummary.

**Status:** resolved

- [ ] Retain or construct the minimum final model snapshot needed to evaluate a
      simulated spectrum on its full output grid.
- [ ] Verify the snapshot matches the outcome's run identities, nonlinear
      parameters, positive finite noise, and final correction revision.
- [ ] Use the outcome's authoritative analytical amplitudes; do not re-solve
      amplitudes during simulation.
- [ ] Reject stale or mismatched snapshots with errors identifying the failed
      invariant.
- [ ] Do not fabricate simulated models for unusable cluster outcomes.
- [ ] Preserve full-grid lineshape mathematics, output orientation, and the
      strict point/series contract.
- [ ] Assert simulated output is unaffected by later mutation of continuation
      state.
- [ ] Verify with
      `uv run pytest -q -p no:cacheprovider -k "simulated_spectrum or simulation"`.

## Comments

- Implemented the completed-run simulation projection from `FinalFitOutcome`
  plus a verified, copied final model snapshot. Usable outcomes supply retained
  analytical amplitudes; unusable outcomes supply no simulated model, and an
  all-unusable run writes no simulated spectrum.
- Added deterministic coverage for outcome classifications, nonconsecutive
  identities, reverse snapshot order, multidimensional grids, series and
  grid-index validation, no amplitude solve, no legacy result reconstruction,
  source immutability, and finite-input equivalence with the former projection.
- Validation passes for the focused simulation suite, ticket 01–08 consumer
  coverage, Ruff, formatting, ty, import contracts, and `git diff --check`.
  The repository-wide safety harness still has two unrelated ticket-08 CSV
  failures: nonnumeric `"unavailable"` standard errors and phase rows assigned
  to a peak rather than `cluster_<id>`. They are outside ticket 09 scope and
  were left unchanged.
