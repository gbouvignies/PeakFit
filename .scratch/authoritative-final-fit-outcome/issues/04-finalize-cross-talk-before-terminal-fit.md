# 04 — Finalize cross-talk before the terminal fit

**What to build:** A finite pass schedule in which only numerically usable
results influence corrections and every final cluster result references the
frozen terminal correction revision.

**Blocked by:** 02 — Establish stable cluster-result identity; 03 — Share
analytical evaluation and usability.

**Status:** resolved

- [x] Define `refine_iterations = N` as exactly `N` optimizer passes and require
      `N >= 1`.
- [x] Perform exactly `N - 1` correction updates and no update after the
      terminal pass.
- [x] Preserve explicit step iteration counts as optimizer-pass counts and apply
      corrections only between passes in the flattened schedule.
- [x] Give each pass an immutable correction snapshot or isolated copy and a
      monotonically increasing `correction_revision`.
- [x] Stamp every task and returned result with the revision it used.
- [x] Merge parameters and update corrections from converged and usable
      non-converged results only.
- [x] Prevent unusable results from changing parameters or corrections.
- [x] Require terminal results to reference the frozen final revision.
- [x] Do not introduce a correction digest.
- [x] Cover rejected zero passes, one pass with zero updates, and multiple
      passes with exactly `N - 1` updates.
- [x] Prove mutation of a source correction array cannot change a snapshot
      already supplied to an optimizer.
- [x] Preserve fitted mathematics apart from the approved pass-count and
      correction-order semantics.
- [x] Verify with
      `uv run pytest -q -p no:cacheprovider -k "cross_talk or correction_revision or refine_iterations"`.

## Completion note — 2026-07-26

The pipeline now freezes a read-only correction snapshot for every optimizer
pass, stamps tasks and results with its revision, and updates corrections only
when another flattened pass remains. `refine_iterations` is now an exact pass
count; callers using the former `N + 1` behavior should increment their value
by one. Focused scheduling tests, ticket 01–03 characterization tests, and the
full validation suite passed before review.
