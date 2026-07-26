# 01 — Characterize competing fit truth

**What to build:** Executable evidence that demonstrates the current disagreement
between real terminal optimizer results and reconstructed durable results, while
also defining the desired cross-consumer agreement without changing production
behavior.

**Blocked by:** None — can start immediately.

**Status:** resolved

- [ ] Add a deterministic three-cluster fixture with nonconsecutive run-local
      identifiers: one converged, one finite non-converged, and one non-finite
      or non-evaluable optimizer result.
- [ ] Demonstrate the current difference between real success, message, and
      function-evaluation count and the synthetic values produced for durable
      statistics.
- [ ] Demonstrate the current difference between CLI review/summary and JSON or
      Markdown status.
- [ ] Capture the current successful-cluster-only summary distributions and
      their lack of explicit population counts.
- [ ] Use ordered and reverse-completion executors to expose the current
      positional association mechanism.
- [ ] Add a correction-observation fixture showing that the current final
      correction update occurs after the retained terminal optimizer result.
- [ ] Characterize the current `refine_iterations + 1` default pass count for
      zero, one, and multiple values plus an explicit multi-step schedule.
- [ ] Characterize current VARPRO amplitude, residual, and provenance values.
- [ ] Characterize fixed-seed basin-hopping amplitude, residual, common
      provenance, and optimizer-specific termination values.
- [ ] Characterize deterministic analytical amplitude re-solving before and
      after deliberately modifying injected amplitude parameters.
- [ ] Demonstrate that current parameter merging and correction updates do not
      use one explicit shared numerical-usability classification.
- [ ] Add desired-contract tests as strict expected failures or an equivalent
      executable diagnostic so the repository remains green before production
      changes.
- [ ] Do not change unexplained golden values, numerical tolerances, output
      schema, or production behavior.
- [ ] Verify with
      `uv run pytest -q -p no:cacheprovider -k "competing_fit_truth or terminal_optimizer_provenance"`.

## Comments

### 2026-07-26 — Characterization complete

- Added `tests/test_competing_fit_truth.py` with small unequal point/series
  fixtures and nonconsecutive cluster identifiers.
- Fourteen passing tests capture current behavior across CLI review,
  `RunSummary`, reconstructed output statistics, JSON, Markdown, ordered and
  reverse executor completion, correction timing, pass counts, VARPRO, basin
  hopping, and independent analytical amplitude re-solving.
- Six strict expected failures define the approved future agreement for
  cross-consumer classification, optimizer provenance, summary populations,
  frozen terminal corrections, positive pass-count validation, and exact
  optimizer-pass counts. Running these with `--runxfail` produces exactly six
  failures.
- Repository reality differs from the future contract in two notable ways:
  current `FitResult` has no numerical-usability classification, so even a
  non-finite residual result is merged before correction; basin hopping returns
  residuals and optimizer metadata but does not add amplitude parameters when
  they were absent from the input `Parameters`.
- Validation:
  `uv run pytest -q -p no:cacheprovider tests/test_competing_fit_truth.py`
  (`14 passed, 6 xfailed`);
  `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest -q -p no:cacheprovider`
  (`141 passed, 6 xfailed`);
  Ruff lint and format checks, `ty check --error-on-warning`, and
  `lint-imports` all passed.
