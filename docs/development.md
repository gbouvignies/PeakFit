# Development Guide

This guide records the current contributor workflow and coding conventions. Keep
it short; implementation details should live in code and tests.

## Commands

Use `uv` for project commands.

```bash
uv sync --all-extras
uv run ruff check src tests
uv run ruff format --check src tests
uv run ty check --error-on-warning
uv run lint-imports
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest -q
uv run prek run --all-files
uv build
```

Use `prek`, not `pre-commit`, for hooks.

## Testing And Scientific Fixtures

Run the complete suite headlessly with `QT_QPA_PLATFORM=offscreen` and
`MPLBACKEND=Agg`, as shown above. This is the supported environment for
automated tests that may import GUI or plotting dependencies.

Tests protect both numerical/scientific behavior and refactoring compatibility
baselines. Treat golden files, regression data, tolerances, and reference values
as evidence with an explicit provenance and intended claim. Do not silently
regenerate them, weaken their assertions, or change them without explaining and
reviewing the scientific or contract reason.

## Architecture Rules

- Keep numerical code independent from Typer, Rich, Qt, and filesystem side
  effects unless that boundary is intentionally changed.
- Prefer direct functions and explicit data flow over managers, service layers,
  registries, factories, or adapters.
- Excellent visual design, readability, intuitiveness and consistency are explicit
  product requirements. A dedicated GUI UX and visual polish milestone remains
  required after establishment of the core whole-dataset workflows; functional
  changes should support that direction.
- Keep `engine` focused on numerical/domain primitives.
- Keep `fit` focused on loading, validation, orchestration, and fit-output
  assembly.
- Keep `auto_pick` isolated as an experimental workflow.
- Keep one shared scientific authority: `scientific.npz`, CSV tables under
  `tables/`, and a short report; JSON is explicit interoperability.

The import-linter contracts in `pyproject.toml` enforce the current package
layers.

## CLI And Terminal Output

PeakFit runs can be long. Terminal output should leave a useful scrollback:

- Show a compact input/config/output summary before work starts.
- Keep routine success terse.
- Make parse errors, failed clusters, boundary hits, and high reduced
  chi-squared easy to find.
- Prefer actionable messages over decorative panels.
- Do not expose internal architecture names in user-facing output.

Shared helpers under `src/peakfit/ui/` are useful only when they reduce
duplication and improve consistency. They may be merged or removed during
simplification.

## Typing

- Do not add `from __future__ import annotations`; Python 3.14 already defers
  annotations.
- Use modern syntax: `X | Y`, `list[str]`, `dict[str, float]`.
- For NumPy-style public inputs, prefer `numpy.typing.ArrayLike` and normalize
  immediately with `np.asarray(...)`.
- Return `numpy.typing.NDArray[...]` for vectorized numerical functions.
- Use overloads only when the public API guarantees scalar-in/scalar-out
  behavior.
- Enforce shape requirements at runtime; do not encode fragile shape types.

## Documentation

Documentation should be consolidated with code changes. Prefer one accurate page
over several overlapping pages. Delete stale notes when they stop matching the
current design.


## GUI responsiveness

The desktop application keeps the Qt event loop free of scientific and disk work:

- Qt objects, paths and widgets are created only on the GUI thread. Input
  validation (`gui.spectrum.read_dataset`), contour geometry and model-plane
  projection run in a `ViewerComputation` lane: one serial background worker per
  owner, with latest-pending requests and revision checks so stale deliveries are
  dropped.
- Fitting runs in one disposable spawned process (`gui.fit_process.FitProcess`).
  Cancellation terminates and reaps the child off the GUI thread; only records from
  our own child cross the pipe, and cancellation or failure never publishes a
  partial result.
- Saving, opening and source verification of `.peakfit` analyses run in a
  disposable archive process (`gui.project_process`), with cancellation in the
  status bar.
- Result export writes off-thread behind a modal progress dialog, then publishes
  a staged directory by rename.
- Status and warning text uses fixed-height fields so progress never reflows the
  plots. Large tables are replaced by clearing rows, using interactive column
  sizing and sizing once after population; per-row resizing froze the GUI for
  more than 30 s on a 664-row parameter table.

Measured latencies are local evidence, not portable guarantees; native
measurements and screenshots are recorded under `docs/validation/`.

## Whole-dataset desktop fitting and review

The GUI runs `run_pipeline_iter` in its existing disposable-process bridge, with a
reviewed two-pass default and threshold independent of display contours. Completed
review uses `CompletedFit`, retains original input/provenance snapshots, and evaluates
all usable model contributions without an amplitude solve or spatial cutoff. Shared
process/export helpers now use the scope-neutral `FitProcess` and `export_fit_result`
names. CLI defaults and canonical output schemas are unchanged.

Verification: full offscreen Qt/Agg suite **386 passed**; focused GUI pipeline/export
checks also passed after the final provenance adjustment. Ruff, formatting, type
checking, import-linter, commit hooks and wheel/sdist build passed. The existing
uv-build version-bound warning remains. `graphify update .` refreshed AST relationships.

The native Cocoa acceptance script drove the bundled 131-plane / 166-peak dataset
through settings review, actual two-pass execution/corrections, full-model overlay,
residuals, plane input, crosshair/selection, cancellation, protected close, edits and
Undo. All 121 clusters were numerically usable/converged; three uncertainty warnings
remained explicit. Scientific values, repeated plane identities and stale delivery
are protected by focused tests. See [native evidence and limitations](validation/gui-dataset-fit.md)
for before/after screenshots, exact native sizes and measured latency.

The unattended native GUI suite encountered focus-dependent failures; the first
Tab-focus failure was reproduced verbatim on unmodified main. It is not reported as
a passing native suite. The dedicated native acceptance run establishes Qt focus
explicitly for synthetic keyboard input and passed; normal application focus code
was not changed to accommodate the test environment.


## GUI v2 analysis persistence validation

`uv run python tools/validate_gui_persistence.py` fits the bundled 131-plane NMRPipe
dataset, saves, repeats Save, uses Save As, reopens verified numerical evidence,
inspects model/residual, edits positions, and performs a fresh real fit. It captures
native Qt light/dark and numerical-only views, JSON timings/payload sizes and
per-operation heartbeat gaps. Run on a desktop with Cocoa/Qt permissions; offscreen
captures are not native acceptance evidence. Results and limitations are recorded
in [validation](validation/gui-v2-persistence.md).

`tests/test_gui_project.py` protects source relocation/absence/mutation, same-size
same-mtime changes, checksum corruption, version refusal, interruption/disk-full,
immutable reuse, lossless numerical round trips, unsupported models, working
transition validation, protected/full/summary retention and trusted child-process
transport. Reopen tests prohibit preparation/optimization and compare reconstructed
models with retained analytical arrays. Existing editor tests protect the changed
analysis-loss guard and scientific Currentness; CLI outputs remain unchanged.


### PR #70: history lifetime, resident memory and compression

`uv run python tools/benchmark_gui_history.py --output /tmp/history.json --reopen`
runs twelve native process-based fits, samples current main-process RSS after
settling and GC, then checks normal-close cleanup and verified reopening. It also
records how many complete results are resident. Use a fresh process for each source
revision; the baseline was measured against the approved pre-review code at
`88d2fa2` via an isolated source copy. `--fits 1 --compression` measures gzip levels
independently so compression buffers do not contaminate repeated-fit RSS.
The saved reports, native captures and declared limitations are in the persistence
validation document. They show one active result and disk references for history;
RSS is variable and child-process peak memory is not claimed to be measured.

Tests cover the full ten-run save/reopen/eleventh-fit/demotion cycle and every
promised compact scientific field, byte-budget pruning, on-demand selection,
normal close while a preservation writer is in flight, non-coalesced cleanup,
recovery adoption/Save as and ownership transfer to numerical-only review.
Session cleanup is queued after writers and is never canceled by executor shutdown.
The internal gzip wrapper is lossless and bounded; typed-array scientific authority
is recommended for separate review rather than held back by compatibility concerns.
