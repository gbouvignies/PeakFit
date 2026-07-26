# 02 — Establish stable cluster-result identity

**What to build:** First-class run-local cluster identity throughout task
creation, execution, result collection, and pipeline completion.

**Blocked by:** 01 — Characterize competing fit truth.

**Status:** resolved

- [ ] Validate that pipeline input contains unique run-local cluster
      identifiers.
- [ ] Carry cluster identity as a first-class task and result value rather than
      relying on list position or open metadata.
- [ ] Produce identical associations under ordered and reverse/unordered
      execution with nonconsecutive identifiers.
- [ ] Reject duplicate input identifiers and duplicate returned identifiers
      with errors naming the duplicates.
- [ ] Keep display or serialization order deterministic by sorting on
      `cluster_id`, never by completion order.
- [ ] Preserve current behavior for a returned non-converged result versus an
      optimizer exception that aborts the run.
- [ ] Convert the ticket-01 ordered/unordered association expected failures into
      passing contract tests.
- [ ] Verify with
      `uv run pytest -q -p no:cacheprovider -k "cluster_identity or unordered_execution"`.

## Comments

### 2026-07-26 — Stable cluster identity complete

- Added first-class run-local `cluster_id` to submitted optimizer tasks and
  `FitResult`; pipeline association no longer reads task position or open
  metadata.
- Pipeline entry rejects duplicate cluster identifiers. Every pass rejects
  duplicate, missing, and unexpected result identifiers with errors naming the
  offending IDs.
- Final pass results and continuation-state clusters use ascending
  `cluster_id` presentation order, independent of input, submission, or
  completion order. Live progress remains an operational completion-event
  stream so long-running unordered execution continues to report promptly.
- Preserved returned non-converged results and optimizer exception propagation.
  Usability, correction scheduling, immutable outcomes, and consumer migration
  remain assigned to later tickets.
- Validation passed:
  focused identity tests (`7 passed`); full headless suite
  (`147 passed, 6 xfailed`); Ruff lint and format checks;
  `ty --error-on-warning`; import contracts (`2 kept, 0 broken`);
  `git diff --check`; and `graphify update .`.
