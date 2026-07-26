# 09 — Simulate from the final outcome

**What to build:** Optional simulated-spectrum output that uses authoritative
outcome amplitudes and a verified matching final model snapshot without solving
new amplitudes.

**Blocked by:** 06 — Migrate CLI and RunSummary.

**Status:** ready-for-agent

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
