# 04 — Finalize cross-talk before the terminal fit

**What to build:** A finite pass schedule in which only numerically usable
results influence corrections and every final cluster result references the
frozen terminal correction revision.

**Blocked by:** 02 — Establish stable cluster-result identity; 03 — Share
analytical evaluation and usability.

**Status:** ready-for-agent

- [ ] Define `refine_iterations = N` as exactly `N` optimizer passes and require
      `N >= 1`.
- [ ] Perform exactly `N - 1` correction updates and no update after the
      terminal pass.
- [ ] Preserve explicit step iteration counts as optimizer-pass counts and apply
      corrections only between passes in the flattened schedule.
- [ ] Give each pass an immutable correction snapshot or isolated copy and a
      monotonically increasing `correction_revision`.
- [ ] Stamp every task and returned result with the revision it used.
- [ ] Merge parameters and update corrections from converged and usable
      non-converged results only.
- [ ] Prevent unusable results from changing parameters or corrections.
- [ ] Require terminal results to reference the frozen final revision.
- [ ] Do not introduce a correction digest.
- [ ] Cover rejected zero passes, one pass with zero updates, and multiple
      passes with exactly `N - 1` updates.
- [ ] Prove mutation of a source correction array cannot change a snapshot
      already supplied to an optimizer.
- [ ] Preserve fitted mathematics apart from the approved pass-count and
      correction-order semantics.
- [ ] Verify with
      `uv run pytest -q -p no:cacheprovider -k "cross_talk or correction_revision or refine_iterations"`.
