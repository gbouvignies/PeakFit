# 05 — Assemble the immutable final fit outcome

**What to build:** A deep finalization module that validates pipeline completion
and constructs one deeply immutable completed fit outcome.

**Blocked by:** 04 — Finalize cross-talk before the terminal fit.

**Status:** ready-for-agent

- [ ] Provide the single
      `finalize_fit(pipeline_completion) -> FinalFitOutcome` construction seam.
- [ ] Require the exact expected cluster-identity set and reject missing,
      duplicate, and unexpected results with errors naming the identifiers.
- [ ] Reject stale correction revisions, mismatched noise, and mismatched
      terminal nonlinear parameters for usable outcomes.
- [ ] Reject missing, nonpositive, or non-finite noise.
- [ ] Construct immutable converged and usable non-converged outcomes containing
      identity, estimates, shared evaluation, statistics, scaled amplitude
      uncertainty, and trustworthy terminal optimizer provenance.
- [ ] Construct immutable unusable outcomes containing identity, explicit
      reason, correction revision, and trustworthy provenance without invented
      amplitudes, model values, or statistics.
- [ ] Preserve convergence status independently from usability status and
      expose the derived three-state classification.
- [ ] Remove basin-hopping message-based convergence promotion.
- [ ] Derive overall convergence and global fit-quality statistics, excluding
      unusable outcomes from scientific aggregation.
- [ ] Preserve existing count, degree-of-freedom, information-criterion, and
      uncertainty-scaling formulas.
- [ ] Preserve optimizer kind, convergence status, usability status,
      termination message or code, function-evaluation count, iteration count
      where available, final cost or chi-squared, and correction revision.
- [ ] Keep useful trustworthy optimizer-specific metadata without inventing an
      artificial common field set.
- [ ] Leave unavailable provenance absent; never synthesize convergence, zero,
      or persistence messages.
- [ ] Reject an empty cluster set explicitly.
- [ ] Copy all nested values and expose no mutable collection, parameter object,
      cluster, mapping, or writable array by reference.
- [ ] Keep mutable continuation state separately named and unreachable through
      the outcome.
- [ ] Keep paths, checksums, timestamps, schemas, and writer configuration
      outside finalization.
- [ ] Verify with
      `uv run pytest -q -p no:cacheprovider -k "final_fit_outcome or final_cluster_outcome"`.
