# 10 — Remove competing truth and validate

**What to build:** One end-to-end PeakFit result path with obsolete
reconstruction branches removed and the representative real-data workflow
protecting cross-consumer agreement.

**Blocked by:** 08 — Migrate human and tabular writers; 09 — Simulate from the
final outcome.

**Status:** resolved

- [x] Delete synthetic convergence and post-fit numerical reconstruction from
      the bounded completed-result flow.
- [x] Delete temporary compatibility properties and redundant fit-run copies of
      final optimizer results, overall convergence, and summary.
- [x] Delete all positional cluster/statistics association in final consumers.
- [x] Verify no writer or CLI consumer reconstructs completed fit truth from
      mutable continuation state.
- [x] Run the representative real-data CLI workflow with its existing inputs,
      unexplained golden values, and tolerances unchanged.
- [x] Add identity-based agreement assertions among CLI review, `RunSummary`,
      JSON statistics/provenance, Markdown status, README summary, CSV
      estimates, and optional simulated output.
- [x] Assert overall convergence, all three classification counts, distribution
      population sizes, and terminal evaluation counts derive from the same
      final cluster outcomes.
- [x] Assert persisted parameters, amplitudes, residual-derived statistics, and
      uncertainty scaling match the authoritative outcome.
- [x] Update lasting architecture, testing, output-schema, migration, and domain
      vocabulary documentation after the implementation decision is approved.
- [x] Refresh Graphify after production changes and verify scoped paths from
      terminal optimizer results to every final consumer.
- [x] Do not create or change an ADR unless the maintainer separately approves
      one after implementation review.
- [x] Run
      `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest -q -p no:cacheprovider`.
- [x] Run `uv run ruff check .`.
- [x] Run `uv run ruff format --check .`.
- [x] Run `uv run ty check --error-on-warning`.
- [x] Run `uv run lint-imports`.
- [x] Run `uv build`.

## Comments

### 2026-07-26 — completed-result authority cleanup

- Removed `fit.results`, `fit.result_models`, the `build_fit_results(FittingState)`
  reconstruction path, synthetic convergence/provenance fields, and the legacy
  CSV/Markdown adapters that accepted reconstructed writer models.
- Moved the remaining non-scientific `RunMetadata` and metadata capture into
  `fit.output_metadata`. `FitRun.state` is removed; the explicit
  `continuation_state` is used only for continuation-state persistence.
- Split internal `PipelineCompletion` (raw terminal optimizer attempts used once
  by finalization) from returned `PipelineResult` (final outcome, explicitly
  named continuation state, and frozen simulation geometry). Completed pipeline
  consumers can no longer access raw results or evaluations.
- Retained `engine.fitting.simulation.simulate_data` because it is a low-level
  non-completed helper. Completed optional simulation remains exclusively
  `simulate_final_outcome(FinalFitOutcome, FinalModelSnapshot, data)` and uses
  stored analytical amplitudes without a new solve.
- Updated point/series tests to exercise the shared analytical evaluation,
  removed obsolete reconstructed-result and positional Markdown fixtures, and
  added an explicit no-`.state` compatibility test. The mixed-classification
  output test mutates continuation state before writing to prove completed
  artifacts remain outcome-derived.
- Inspected a representative finite CLI fit: JSON 4.0.0, `clusters.csv`, and
  `intensities.csv` agree on stable nonconsecutive `cluster_id`, correction
  revision, and terminal optimizer provenance. Inspected a mixed run: JSON,
  CSV, Markdown, and README agree on converged / usable-non-converged /
  unusable classifications; the unusable row has no fabricated numerical data.
- The representative real CLI regression now also asserts JSON, status CSV,
  parameter/intensity tables, Markdown, and README agree by `cluster_id`.
  Outcome-based Markdown coverage retains the 40-row cap and omission notices.
- Validation passed: focused ticket-10 suite (71 tests), full serial headless
  suite, representative real-data output tests, Ruff, formatting, ty,
  import-linter, `git diff --check`, `prek --all-files`, and `uv build`.
  `graphify update .` completed; its known warning concerns three non-source
  zero-node artifacts and the final rebuild reported 2,741 nodes and 6,475 edges.
