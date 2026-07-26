# 06 — Migrate CLI and RunSummary

**What to build:** A fit-run and CLI flow whose convergence, summary, review
decisions, and terminal display are projections of the final fit outcome.

**Blocked by:** 05 — Assemble the immutable final fit outcome.

**Status:** ready-for-agent

- [ ] Make the fit-run container carry exactly the authoritative outcome plus
      separately named mutable continuation state and operational context.
- [ ] Derive overall convergence from the three-state per-cluster
      classifications.
- [ ] Report total, converged, usable non-converged, and unusable cluster counts
      in `RunSummary`.
- [ ] Build reduced-chi-squared distributions from converged and usable
      non-converged outcomes while excluding unusable outcomes.
- [ ] Report the population size used for every summary distribution.
- [ ] Derive cluster review status, reason, bound checks, peak names, and
      reduced chi-squared from the corresponding cluster outcome.
- [ ] Ensure mixed classifications produce identical identities and statuses in
      headless and interactive workflows.
- [ ] Ensure ordered versus reverse completion cannot change summary or review.
- [ ] Introduce only temporary derived compatibility properties needed for
      later consumer migration; mark them for deletion in ticket 10.
- [ ] Convert the ticket-01 CLI and summary expected failures into passing
      cross-consumer tests.
- [ ] Keep Rich formatting, presentation thresholds, and user-facing wording
      otherwise unchanged.
- [ ] Verify with
      `uv run pytest -q -p no:cacheprovider -k "run_summary or review_clusters or outcome_classification"`.
