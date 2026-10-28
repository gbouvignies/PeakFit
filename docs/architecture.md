# PeakFit Architecture Guide

This is a short map of the current repository, not an immutable contract. `AGENTS.md`
is the source of truth for AI-assisted work and explicitly allows architecture changes
when they simplify the project while preserving scientific correctness and useful
workflows.

## Purpose

PeakFit fits lineshape models to pseudo-ND NMR spectra through a Python package and CLI.
The important domains are spectra and peak inputs, clustering, lineshapes, optimization,
fit orchestration, uncertainty analysis, plotting, and output generation.

## Current Package Map

- `src/peakfit/cli/` - Typer command-line entrypoints and terminal presentation.
- `src/peakfit/gui/` - PySide6/PyQtGraph desktop window, reusable spectrum viewer,
  and contour geometry helpers. Uses the existing `Spectra` model and input readers;
  working peak records and undo history are Qt-independent. Imported placeholder
  shapes are discarded; editable GUI coordinates are separate from completed results.
  Local scope preparation constructs fresh engine-safe peaks from snapshots, reuses
  engine clustering and pipeline parameter preparation, and displays the exact mask
  in a contextual Inspect/Fit header above persistent experimental inspection plots.
  Explicit numerical details temporarily replace those plots in the same column;
  header state and popup contour controls do not reflow the main spectrum.
  One serial background lane handles numerical
  preparation and contour extraction; Qt applies only current results.
  Exact lightweight scope geometry remains available even when numerical data
  exceeds preparation limits; only eligible selected-group arrays are materialized.
  Full contours remain visible during background replacement, without a vertex cutoff.
  `gui.local_fit.execute_local_fit()` is Qt-independent: it submits the reviewed
  cluster/parameters through `fit_single_cluster_task()`, classifies its analytical
  evaluation, and uses the pipeline's shared `complete_fit()` finalization boundary.
  It never repeats clustering or effective-parameter preparation. Frozen outcomes
  retain signed amplitudes, sampled model/residual values, diagnostics, original
  inputs, stable GUI identities, and copied model geometry for future visualization.
  One disposable spawned fit process runs independently of the preview/contour lane;
  the bridge serializes selected-cluster inputs and freezes received values off Qt.
  There is no fit queue. Cancellation terminates/reaps the child. Scientific edits
  pause until execution stops; navigation remains available. One unsaved result is
  retained separately and protected against replacement/loss. No automatic peak
  picking, result contour comparison, or fitted-coordinate application is involved.
- `src/peakfit/ui/` - Rich terminal helpers used by the current CLI.
- `src/peakfit/fit/` - validation, data loading, fitting workflow, result assembly, and output coordination.
- `src/peakfit/auto_pick/` - experimental automatic peak-picking workflow used only when no peak list is provided.
- `src/peakfit/mcmc/` - uncertainty sampling, diagnostics, and MCMC-facing workflows.
- `src/peakfit/plot/` - plotting and the spectrum viewer.
- `src/peakfit/io/` - parsing, serialization, and file-format handling.
- `src/peakfit/engine/` - domain models, numerical algorithms, lineshapes, and fitting math.
- `src/peakfit/shared/` - small cross-cutting helpers.

## Current Flow

1. `cli` parses user intent and builds configuration.
2. `fit` validates inputs, loads spectra/peaks, prepares clusters, and runs direct fit pipeline functions.
   When no peak list is provided, `fit` delegates to the experimental `auto_pick` package instead of owning that logic.
3. `engine` performs numerical work: clustering, lineshapes, optimizers, residuals, and statistics helpers.
4. `fit.final_outcome` freezes completed scientific truth by run-local
   `cluster_id` against the terminal correction revision, while
   `fit.output_metadata` captures non-scientific run metadata; `io` reads inputs
   and projects the outcome into structured artifacts.
5. `mcmc` and `plot` run post-fit workflows from canonical fit outputs.
6. `ui` renders terminal output where the CLI needs it.

## Boundaries

- The GUI is a presentation layer above the scientific code. Retained Qt contour
  paths serve interactive navigation; the original spectrum samples remain unchanged.

- `engine` should stay free of Typer, Rich, Qt, and filesystem side effects.
- `fit`, `mcmc`, and `plot` may orchestrate workflows, but should call direct
  functions rather than adding manager/service layers.
- `io.schemas` is the JSON validation/read contract. `FinalFitOutcome` is the
  sole completed scientific result; `FittingState` is continuation-only and
  must not be used to reconstruct completed output.
- `auto_pick` is intentionally separate from `fit` because it is experimental.

## Invariants To Preserve Or Change Explicitly

- Numerical and scientific behavior should remain correct and tested.
- Expensive fit work should fail early on invalid inputs.
- Input parsing errors should be clear and actionable.
- Long-running jobs should show understandable progress and failures.
- Pure numerical code should not accidentally depend on terminal UI, Typer, Qt, or filesystem side effects.
- Peak-cluster data has the strict shape `(n_points, n_series)`; point, series,
  observation, and amplitude counts derive from that one representation.
- Public behavior changes should be documented with migration notes when needed.

## Simplification Notes

The current vertical-slice shape is a starting point. It is acceptable to merge packages,
rename modules, delete thin wrappers, or move responsibilities when that makes the code
easier to understand and test. Prefer direct data flow and domain names over generic
services, managers, factories, adapters, or registries.

Completed-model projection uses `fit.simulation.project_final_outcome`: paired
fractional grid coordinates and optional original plane indices produce local or
combined contributions from the verified final snapshot and retained amplitudes.
The existing full-grid simulation delegates to this same path. Shape evaluation
is chunked; callers bound their output region. No optimization, amplitude solve,
or parameter-default preparation takes place during visualization.
