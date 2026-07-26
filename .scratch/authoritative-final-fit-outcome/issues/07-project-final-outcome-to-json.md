# 07 — Project the final outcome to JSON

**What to build:** A pure writer-facing projection and JSON 4.0.0 schema whose
per-cluster classification, applicable statistics, and optimizer provenance
come directly from the final fit outcome.

**Blocked by:** 05 — Assemble the immutable final fit outcome.

**Status:** ready-for-agent

- [ ] Make result construction accept the final outcome as its only source of
      fitted estimates, statistics, and provenance.
- [ ] Keep result construction limited to deterministic projection plus output
      metadata and plane values.
- [ ] Prove the projection does not evaluate lineshapes, solve amplitudes,
      calculate residuals or statistics, infer identity, convergence, or
      usability.
- [ ] Bump the development JSON schema from `3.0.0` to `4.0.0`.
- [ ] Nest each cluster's classification, optional statistics, and optimizer
      provenance with its explicit `cluster_id`; remove the parallel top-level
      per-cluster statistics list.
- [ ] Persist optimizer kind, convergence status, usability status, termination
      message or code, function-evaluation count, iteration count where
      available, final cost or chi-squared, and correction revision.
- [ ] Persist useful trustworthy VARPRO and basin-hopping details without
      manufacturing unavailable or artificial common values.
- [ ] Represent unusable outcomes without fabricated amplitudes, statistics, or
      scientific model values.
- [ ] Reject `3.0.0` with a clear error naming both encountered `3.0.0` and
      supported `4.0.0`.
- [ ] Keep mutable continuation-state persistence separate and do not claim it
      is completed fit truth.
- [ ] Convert the ticket-01 JSON provenance expected failures into passing
      projection and schema tests.
- [ ] Verify with
      `uv run pytest -q -p no:cacheprovider -k "fit_summary_schema or json_statistics or optimizer_provenance"`.
