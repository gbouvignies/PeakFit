# 08 — Migrate human and tabular writers

**What to build:** CSV, Markdown, README, and output orchestration that consume
the same writer-facing projection as JSON without positional joins or
scientific recomputation.

**Blocked by:** 06 — Migrate CLI and RunSummary; 07 — Project the final outcome
to JSON.

**Status:** ready-for-agent

- [ ] Make CSV, Markdown, README, and output planning consume projections of the
      final outcome.
- [ ] Associate every cluster-specific status, statistic, and estimate through
      explicit `cluster_id`.
- [ ] Remove Markdown's positional join between cluster estimates and
      statistics.
- [ ] Preserve outcome amplitudes, residual-derived statistics, uncertainty
      scaling, and optimizer provenance without recalculation.
- [ ] Assert exact identity and classification agreement among CLI review,
      `RunSummary`, JSON, Markdown, and README for a run containing all three
      classifications.
- [ ] Report unusable outcomes without fabricated fit-quality statistics.
- [ ] Assert CSV estimates correspond to the same cluster outcomes used by the
      other consumers.
- [ ] Preserve file planning, atomic writes, checksums, and presentation layout
      except where JSON 4.0.0 or explicit identity requires a documented break.
- [ ] Verify with
      `uv run pytest -q -p no:cacheprovider -k "markdown or readme or csv or output_layout"`.
