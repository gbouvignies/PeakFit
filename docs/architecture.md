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
  preparation, contour extraction and display projection; Qt applies only current
  results. `gui.fit_display` evaluates bounded planes/native traces through the
  scientific simulation API; `gui.comparison` owns Qt graphics and bounded caches.
  Experimental contours have queue priority; obsolete projections cancel between
  chunks. Graphics never connect model/experiment from different planes.
  Exact lightweight scope geometry remains available even when numerical data
  exceeds preparation limits; only eligible selected-group arrays are materialized.
  Full contour geometry has no vertex cutoff. Existing experimental contours remain
  during threshold replacement; comparison hides them during a plane change.
  `gui.local_fit.execute_local_fit()` is Qt-independent: it submits the reviewed
  cluster/parameters through `fit_single_cluster_task()`, classifies its analytical
  evaluation, and uses the pipeline's shared `complete_fit()` finalization boundary.
  It never repeats clustering or effective-parameter preparation. Frozen outcomes
  retain signed amplitudes, sampled model/residual values, diagnostics, original
  inputs, stable GUI identities, and copied model geometry for deterministic graphical comparison.
  One disposable spawned fit process runs independently of the preview/contour lane;
  the bridge serializes selected-cluster inputs and freezes received values off Qt.
  There is no fit queue. Cancellation terminates/reaps the child. Scientific edits
  pause until execution stops; navigation remains available. One unsaved result is
  retained separately and protected against replacement/loss. No automatic peak
  picking or persistence is involved. `gui.fit_positions` validates final chemical
  shifts against retained shape axes and the engine-to-working-ID mapping. Explicit
  application uses one `PeakEdits.update_positions()` history entry, with all edits
  validated before commit. Scientific inputs must still match. The original outcome
  and export state remain owned by the retained result; applying positions makes
  comparison graphics stale without discarding that result.
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


Local GUI fit export reuses `io.writers.orchestrator.write_fit_outputs` directly
from the retained `LocalFitResult.outcome`; `gui.fit_export` stages and validates
the canonical artifacts and adds the scope/identity companion documented in
[outputs](outputs.md#gui-local-fit-export). `gui.export_dialog` owns only directory
selection and a short-lived background export thread. The viewer records export
status by completed run identity independently of working peak edits. No persisted
GUI session or numerical continuation state is constructed.

The linear projection boundary in `engine.algorithms.linear_algebra` supplies
one rank-aware model space and amplitude pseudoinverse to optimization and
analytical output. A covariance failure does not invalidate an otherwise finite
model: frozen evaluations can carry unavailable uncertainties plus an explicit
warning. Numerical usability, optimizer termination and peak identifiability
remain distinct. See [the scientific overlap validation](validation/overlapping-peaks.md).

The optional CLI linewidth comparison lives in `fit/linewidth_comparison.py`.
It composes two `run_fit` calls on independently copied peaks/clusters with
shared read-only experimental spectra. Eligible alternative starts are expressed
as ordinary per-peak constraints, using the existing resolver and first-pass
preparation; the optimizer and pipeline remain unchanged. Each run retains its
own `FinalFitOutcome`, corrections, continuation state and normal output tree.
Comparison projections use only immutable final outcomes and stable cluster and
parameter identities. Comparison provenance and execution failures live beside
the candidates, outside the canonical scientific JSON schema. This orchestration
has no GUI dependency and may be reused by a future candidate-management UI.
