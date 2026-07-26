# 06 — Migrate CLI and RunSummary

**What to build:** A fit-run and CLI flow whose convergence, summary, review
decisions, and terminal display are projections of the final fit outcome.

**Blocked by:** 05 — Assemble the immutable final fit outcome.

**Status:** resolved

- [x] Make the fit-run container carry exactly the authoritative outcome plus
      separately named mutable continuation state and operational context.
- [x] Derive overall convergence from the three-state per-cluster
      classifications.
- [x] Report total, converged, usable non-converged, and unusable cluster counts
      in `RunSummary`.
- [x] Build reduced-chi-squared distributions from converged and usable
      non-converged outcomes while excluding unusable outcomes.
- [x] Report the population size used for every summary distribution.
- [x] Derive cluster review status, reason, bound checks, peak names, and
      reduced chi-squared from the corresponding cluster outcome.
- [x] Ensure mixed classifications produce identical identities and statuses in
      headless and interactive workflows.
- [x] Ensure ordered versus reverse completion cannot change summary or review.
- [x] Introduce only temporary derived compatibility properties needed for
      later consumer migration; mark them for deletion in ticket 10.
- [x] Convert the ticket-01 CLI and summary expected failures into passing
      cross-consumer tests.
- [x] Keep Rich formatting, presentation thresholds, and user-facing wording
      otherwise unchanged.
- [x] Verify with
      `uv run pytest -q -p no:cacheprovider -k "run_summary or review_clusters or outcome_classification"`.

## Comments

### 2026-07-26 — CLI and RunSummary migrated to the final outcome

- `FitRun` now carries `outcome`, explicit `continuation_state`, and operational
  context. Its temporary `.state` compatibility alias is documented for
  deletion in ticket 10; existing state persistence therefore remains deferred.
- `RunSummary.from_outcome()` counts converged, usable non-converged, and
  unusable outcomes independently. Its reduced-chi-squared distribution covers
  only usable outcomes and records `redchi_population_size`; when no outcome is
  usable, its aggregate values are `None` rather than invented numeric values.
- Cluster review is ordered by `cluster_id` and reads classification, peak
  names, immutable final parameter bounds, frozen analytical reduced chi-
  squared values, unusability reason, and actual terminal optimizer message
  directly from the outcome. A successful-sounding termination message cannot
  promote a usable non-converged outcome.
- Rich now presents converged, usable-not-converged, and unusable counts and
  review rows distinctly. JSON, Markdown, CSV, README outcome projection, state
  persistence, and simulation remain intentionally deferred to tickets 07–09;
  the legacy README receives only an `N/A` compatibility fallback so an
  all-unusable CLI run can finish.
- Added deterministic projection tests for all-converged, mixed, and
  all-unusable runs, nonconsecutive IDs, presentation order, usable-only
  distributions, absent unusable numerical fields, and terminal provenance.
- Remaining strict xfails: the two durable JSON/Markdown provenance and
  classification contracts owned by tickets 07–08.
