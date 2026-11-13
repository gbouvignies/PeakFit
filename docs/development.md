# Development Guide

This guide records the current contributor workflow. Keep it short;
implementation details should live in code and tests.

## Commands

Use `uv` for project commands.

```bash
uv sync
uv run ruff check src tests
uv run ruff format --check src tests
uv run ty check --error-on-warning
uv run lint-imports
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest -q -n auto
uv run prek run --all-files
uv build
```

Use `prek`, not `pre-commit`, for hooks.

## Testing And Scientific Fixtures

Run the complete suite headlessly with `QT_QPA_PLATFORM=offscreen` and
`MPLBACKEND=Agg`, as shown above. This is the supported environment for
automated tests that may import GUI or plotting dependencies. Run the full suite
with `-n auto`, as CI does: about 70 s instead of about 300 s serially.

NMRPipe is needed only to regenerate fixtures (`tools/make_nmrpipe_noise_fixtures.py`,
`tools/profile_uncertainty/make_lp_fixtures.py`). Its own init script configures
it for csh, so `which nmrPipe` in bash or zsh can find nothing even when it is
installed. Check with `csh -c 'echo $NMRBASE'`. The fixture tools take
`--nmrpipe <install dir>` and default to `$NMRBASE`.

Shared synthetic inputs and completed fits live in `tests/builders.py`, which
imports no Qt; `tests/gui_helpers.wait_until` runs the Qt event loop until a
condition holds, and GUI tests take the session `qapp` fixture from
`tests/conftest.py`. Test modules do not import each other, and GUI tests wait
for a condition before asserting, not for a fixed delay.

Tests protect both numerical/scientific behavior and refactoring compatibility
baselines. Treat golden files, regression data, tolerances, and reference values
as evidence with an explicit provenance and intended claim. Do not silently
regenerate them, weaken their assertions, or change them without explaining and
reviewing the scientific or contract reason.

### Golden fit reference

`tests/integration/test_golden_regression.py` fits the bundled example once and
compares it with `tests/data/golden/example-fit.json` (provenance, fit arguments,
statistics, every cluster's classification, chi-squared and nonlinear parameters)
and `example-fit-amplitudes.npz` (amplitudes, their conditional standard errors and
their profile σ, the published `intensity_err`). Identifiable clusters must match
within 1e-3 of each reference standard error, standard errors to 1e-6 and
chi-squared to 1e-8 (relative). Clusters
carrying the uncertainty warning are compared through classification, warning and
chi-squared only, because the data do not determine their individual values. The
module docstring records the measurements behind these tolerances.
The reference reads only `scientific.npz` and records only numerical fit
arguments; the shared fixture adds the export formats that the export-projection
tests in `test_safety_harness.py` inspect.

A failure there is a numerical change. If it is intended, explain the scientific
reason in the PR and `docs/decisions.md`, then regenerate the reference with
`uv run python tools/write_golden_reference.py --source-commit <commit>`.

### Changes that must not alter numerical results

The golden tolerances admit rounding between platforms. A refactor that claims no
numerical change must also reproduce the fit exactly in one environment: run
`uv run python tools/scientific_digest.py` on the base commit and on the change
and compare the digests. It fits the example and hashes `scientific.npz` without
its run ID, timestamps, output directory or fit timings.

## Architecture Rules

The review rules for architecture, GUI, typing, CLI output and documentation are
in `CODING_STANDARDS.md`.

The import-linter contracts in `pyproject.toml` enforce the current package
layers, keep the siblings `gui`/`ui` and `fit`/`mcmc`/`plot` independent of each
other, and keep Qt, Typer, Rich and matplotlib out of `engine`, `fit`, `io`,
`mcmc` and `shared`, and Qt out of `plot`. Imports used only for type checking
count too; the few mutual type references between sibling modules are listed in
the acyclic contract's `ignore_imports`, which fails if one of them disappears.

## GUI responsiveness

The desktop application keeps the Qt event loop free of scientific and disk work:

- Qt objects, paths and widgets are created only on the GUI thread. Input
  validation (`gui.spectrum.read_dataset`), contour geometry, model-plane
  projection and the Profiles view's tiles run in a `ViewerComputation` lane: one serial background worker per
  owner, with latest-pending requests and revision checks so stale deliveries are
  dropped. `close()` waits for a delivery already emitting and no delivery starts
  after it, so an owner may be deleted as soon as its lanes are closed.
- Fits and analysis save, open and source verification each run in one disposable
  spawned process through `gui.child_process.run_child`: a `ready` handshake, the
  encoded payload, progress messages, then one result or error record, with the
  child always reaped. Only records from our own child cross the pipe. A fit's
  cancellation terminates the child; archive operations cancel cooperatively, with
  cancellation in the status bar. Cancellation or failure never publishes a
  partial result.
- Historical runs from a saved analysis (`StoredRun.load`) are validated and
  loaded on the requesting view's background worker, not in a separate process.
- Result export writes off-thread behind a modal progress dialog, then publishes
  a staged directory by rename.
- Every job boundary (background lane, fit process, archive process, export) logs
  a failure's full traceback to the `peakfit.gui` logger, which prints to the
  terminal that started PeakFit; worker processes send theirs back with the
  failure (`gui.failures`). Dialogs show `ValueError`/`OSError` messages as raised,
  and ask for a report for anything else.
- The editor renders fitting state on its transitions, never by polling:
  `DatasetFitting.state_changed` fires when a job or result changes, and the
  workspace redraws from that state (`EditorWorkspace.render_fitting`). One busy
  state, `DatasetFitting.busy`, locks every scientific edit; a timer runs only
  while a fit is busy, to show elapsed time.
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
