# 10 — Remove competing truth and validate

**What to build:** One end-to-end PeakFit result path with obsolete
reconstruction branches removed and the representative real-data workflow
protecting cross-consumer agreement.

**Blocked by:** 08 — Migrate human and tabular writers; 09 — Simulate from the
final outcome.

**Status:** ready-for-agent

- [ ] Delete synthetic convergence and post-fit numerical reconstruction from
      the bounded completed-result flow.
- [ ] Delete temporary compatibility properties and redundant fit-run copies of
      final optimizer results, overall convergence, and summary.
- [ ] Delete all positional cluster/statistics association in final consumers.
- [ ] Verify no writer or CLI consumer reconstructs completed fit truth from
      mutable continuation state.
- [ ] Run the representative real-data CLI workflow with its existing inputs,
      unexplained golden values, and tolerances unchanged.
- [ ] Add identity-based agreement assertions among CLI review, `RunSummary`,
      JSON statistics/provenance, Markdown status, README summary, CSV
      estimates, and optional simulated output.
- [ ] Assert overall convergence, all three classification counts, distribution
      population sizes, and terminal evaluation counts derive from the same
      final cluster outcomes.
- [ ] Assert persisted parameters, amplitudes, residual-derived statistics, and
      uncertainty scaling match the authoritative outcome.
- [ ] Update lasting architecture, testing, output-schema, migration, and domain
      vocabulary documentation after the implementation decision is approved.
- [ ] Refresh Graphify after production changes and verify scoped paths from
      terminal optimizer results to every final consumer.
- [ ] Do not create or change an ADR unless the maintainer separately approves
      one after implementation review.
- [ ] Run
      `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest -q -p no:cacheprovider`.
- [ ] Run `uv run ruff check .`.
- [ ] Run `uv run ruff format --check .`.
- [ ] Run `uv run ty check --error-on-warning`.
- [ ] Run `uv run lint-imports`.
- [ ] Run `uv build`.
