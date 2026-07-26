# 02 — Establish stable cluster-result identity

**What to build:** First-class run-local cluster identity throughout task
creation, execution, result collection, and pipeline completion.

**Blocked by:** 01 — Characterize competing fit truth.

**Status:** ready-for-agent

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
