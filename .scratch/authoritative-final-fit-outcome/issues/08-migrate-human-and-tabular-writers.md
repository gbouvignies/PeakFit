# 08 — Migrate human and tabular writers

**What to build:** CSV, Markdown, README, and output orchestration that consume
the same writer-facing projection as JSON without positional joins or
scientific recomputation.

**Blocked by:** 06 — Migrate CLI and RunSummary; 07 — Project the final outcome
to JSON.

**Status:** resolved

- [x] Make CSV, Markdown, README, and output planning consume projections of the
      final outcome.
- [x] Associate every cluster-specific status, statistic, and estimate through
      explicit `cluster_id`.
- [x] Remove Markdown's positional join between cluster estimates and
      statistics.
- [x] Preserve outcome amplitudes, residual-derived statistics, uncertainty
      scaling, and optimizer provenance without recalculation.
- [x] Assert exact identity and classification agreement among CLI review,
      `RunSummary`, JSON, Markdown, and README for a run containing all three
      classifications.
- [x] Report unusable outcomes without fabricated fit-quality statistics.
- [x] Assert CSV estimates correspond to the same cluster outcomes used by the
      other consumers.
- [x] Preserve file planning, atomic writes, checksums, and presentation layout
      except where JSON 4.0.0 or explicit identity requires a documented break.
- [x] Verify with
      `uv run pytest -q -p no:cacheprovider -k "markdown or readme or csv or output_layout"`.

## Comments

### 2026-07-26 — human and tabular writers migrated to final outcomes

- `write_fit_outputs` now accepts `FinalFitOutcome` directly. Completed CSV,
  Markdown, and README output no longer invokes `build_fit_results` or reads
  mutable continuation state. JSON remains its independent 4.0.0 projection.
- `tables/clusters.csv` is the explicit one-row-per-cluster table: it carries
  stable `cluster_id`, classification, usability, correction revision, actual
  terminal optimizer provenance, and usable-only statistics. The established
  normalized parameter, intensity, and shift tables retain their roles; they
  contain only usable numerical rows and include the same outcome status fields.
- Markdown presents final outcomes in ascending `cluster_id` order, labels all
  three classifications, shows actual terminal messages, and uses `N/A` for
  unavailable scientific values. README reuses ticket-06 `RunSummary` counts
  and usable-only distribution population semantics.
- Added deterministic all-converged, mixed, all-unusable, nonconsecutive-ID,
  ordering, terminal-provenance, successful-sounding-message, empty-unusable-
  numerical-field, cross-consumer, and no-legacy-reconstruction coverage.
- Simulation remains deliberately deferred to ticket 09.

### 2026-07-26 — CSV safety-harness contract clarification

- `parameters.csv` represents an absent final nonlinear standard error as the
  explicit string `unavailable`; consumers must accept that value and must not
  fabricate a numerical uncertainty.
- Shared phase parameters retain their stable identity through matching
  `cluster_<id>.F3.phase` and `cluster_id`. The normalized `peak_name` column
  remains a concrete peak label, not a synthetic `cluster_<id>` pseudo-peak.
