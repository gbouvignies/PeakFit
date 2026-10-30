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
  over persistent spectrum/slice plots. `gui.workspace.SpectrumWorkspace` owns
  splitter geometry, side-slice visibility and the collapsed-by-default lower
  panel. It receives existing widgets and owns no scientific state. The viewer
  keeps preparation/result state independent of which details are visible;
  background deliveries never open panels. `gui.fit_preview` presents local
  scope/results below the plots; `gui.editing` requests a whole-peak commit on Enter
  or focus leaving the field group. Tab between fields retains the draft; Escape
  restores committed values. The viewer also commits before selection, hiding
  details, fitting and peak export, retaining invalid drafts with inline errors. The
  `pending_edits_changed` notification updates editor controls and unsaved-work
  indicators only; `edits_changed` is reserved for actual committed peak changes
  and invalidates scientific dependents. No-op commit/history/drag operations do
  not advance the scientific revision. One coordinated
  comparison selector identifies either complete-dataset or isolated local scope. The reserved
  footer exposes fit state and cancellation while details are closed.
  One serial background lane handles numerical
  preparation, contour extraction and display projection; Qt applies only current
  results. `gui.fit_display` evaluates bounded planes/native traces through the
  scientific simulation API; `gui.comparison` owns Qt graphics and bounded caches.
  Complete models evaluate every native coordinate and every usable final cluster;
  their residual is experimental minus total model outside as well as inside masks.
  Local display retains its distinct bounded prediction and fitted-sample residual.
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
  `gui.fit_process.FitProcess` serializes reviewed inputs and receives completed
  values off Qt. `gui.dataset_dialog` prepares complete problems asynchronously
  and presents effective settings and initial bounds. `gui.dataset_fit` runs
  `run_pipeline_iter` with the reviewed configuration, including cross-cluster
  updates, and binds its `CompletedFit` to the original input snapshot. It does not
  implement another fitting scheduler. The GUI defaults to two passes; CLI defaults
  are unchanged. A private trusted-child pickle transport preserves mapping proxies
  and read-only NumPy buffers; it is not a saved-run format or an external reader.
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
fractional grid coordinates and optional fitted-series column indices produce local or
combined contributions from the verified final snapshot and retained amplitudes.
The existing full-grid simulation delegates to this same path. Shape evaluation
is chunked; callers bound their output region. No optimization, amplitude solve,
or parameter-default preparation takes place during visualization.

`fit.completed.CompletedFit` retains the outcome, the existing model snapshot,
and owned read-only `FinalClusterSamples` at pipeline completion. Exact paired
sample coordinates, original observations and terminal neighbor-correction values
remain distinct from model geometry and analytical outputs. `PipelineResult`
exposes this record and preserves its `final_outcome`/`simulation_snapshot`
accessors; `run_fit()` also supplies it through `FitRun.completed_fit`. Existing
FitRun construction and writers remain supported. Completed parameter/amplitude
arrays are referenced, never duplicated into the input record. `FittingState`
remains independently mutable. Capturing this numerical record is not a persisted
session or an MCMC readiness check.

`project_final_residual` calculates observations minus the sum of usable completed
models, including outside fitting masks. This display residual can differ from
the frozen, neighbor-corrected optimizer residuals; their existing statistics must
not be reinterpreted as whole-grid display statistics. Projection planes index
fitted-series columns; source-plane identities after exclusions require an explicit
mapping in the future persisted-run interface. See [GUI design](gui-design.md)
for that interface's scientific requirements and staged delivery.


GUI fit export reuses `io.writers.orchestrator.write_fit_outputs` directly
from the retained local or complete dataset outcome; `gui.fit_export` stages and validates
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


## GUI v2: open → inspect → edit → fit → compare → export

`gui.editor.window` is the separately launched production shell (`peakfit-editor`).
The existing `gui.app` / `gui.viewer` remains the fitting application until v2
reaches core fitting parity. There is no runtime dependency on `experiments/`.

The workspace has concrete responsibilities: `editor.loading` validates
one complete reader handoff off Qt; `editor.canvas` owns retained signed geometry,
native slices and gestures; `editor.rail` owns a flat assignment list and the precise
whole-group editor; `editor.workspace` commits scientific edits and composes those
views. `editor.window` owns file dialogs, atomic replacement, menus and appearance.
`editor.fitting` directly coordinates the shared `DatasetFitDialog`,
`prepare_dataset_fit`, disposable `FitProcess`, `DatasetCompletion`, and canonical
exporter. It does not duplicate fitting algorithms or introduce a job framework.
These are direct widgets/functions, with no panel/job/state framework. Shared
`gui.spectrum`, `gui.peaks`, `gui.editing` and `gui.background` retain reader,
coordinate, editing, CSV and bounded computation behavior. Qt stays out of the
engine. Data/plane arrays are read-only after validation; changing plane values
creates replacement metadata rather than modifying an accepted source in place.

Preview extracts one native row/column and updates only transient highlights,
curves and sampled-value text. It never extracts contours or rebuilds all markers.
One Qt-turn coalescer installs the latest pointer request. Plane changes remove
old-plane geometry immediately and use the existing serial worker with revision
checks. Numerical geometry is bounded to eight planes/32 MiB; only two sign paths
are resident in the Qt scene. Theme changes restyle those paths. Dataset-wide trace
limits are computed once during loading, independently of display navigation.

`PeakEdits` holds immutable stable-ID/unfolded-coordinate records, named undo
history and a monotonic manual revision. Committed changes, Undo and Redo advance
the revision; no-ops, drafts, hover, selection, plane/viewport/display changes and
export do not. Peak import is one undoable collection change. Deleting an assignment
never merges or reidentifies coincident/duplicate names. The flat list updates rows
in place, avoiding tree/model resets during ordinary coordinate editing.

The editor retains peak-file origins by stable working ID; effective `Dataset.sources`
is derived from the current IDs on every committed/history refresh. Replace/Append
and Undo/Redo therefore restore peaks and effective provenance together, without a
second source-history cursor. Origins of undone IDs are retained for Redo but are
not advertised as active inputs. Non-peak source records, including the current
plane mapping, survive peak Undo. Canvas and workspace use the same dataset record.

Plane-value import/replacement advances the scientific revision and sets a
session-only metadata flag. The window/header and close/replacement guard include
that flag alongside peak dirtiness/drafts. Peak CSV export clears only peak
export dirtiness; it neither clears the metadata flag nor authorizes its loss.
Export from the loss guard keeps the window open when plane metadata remains;
only explicit Discard permits loss. No project persistence or plane-value Undo
is introduced by this contract.

### Atomic fitting completion

`gui.fit_positions.dataset_completion` prepares one immutable `DatasetCompletion`
value from a `DatasetFitResult` plus exact current `FitInputSnapshot` and revision.
The v2 fit launcher now uses this contract for every successful publication. It validates
source/settings/identity/axis consistency before returning anything publishable,
preserves the original immutable outcome and submitted seeds, and constructs the
complete post-fit working tuple using final chemical shifts by stable ID. Both
axes change together; missing, unusable, non-finite or unmappable pairs retain their
original values with per-ID warnings. Usable non-convergence/uncertainty warnings
remain evidence and do not prohibit valid coordinate adoption.

The workspace publishes **this single completion value** and resets the
`PeakEdits` history in the same Qt delivery, retaining its prior exported-state
baseline and `revision=completion.manual_revision`. Currentness
compares exact inputs to the recorded post-fit tuple and the recorded manual
revision, rather than comparing optimized coordinates to original seeds. Returning
values through Undo cannot recertify a run. Source/settings/plane replacements
also change the snapshot/revision. Completion preparation never rewrites accepted
bounds/masks/parameters; the next fit must call `prepare_dataset_fit` afresh from
optimized working positions. Failed/canceled jobs never call completion publication.
The UI delivery must preserve selection, viewport and panel choices and move a
selected-peak pin once, retaining a free inspection pin. `test_gui_editor_fitting.py` protects the real process loop, partial/all-unusable
completion, invalid publication, selected/free pins, rapid stale plane requests,
native total-model slices, draft/currentness behavior and repeated refits.

The canvas uses the existing `dataset_model_plane` projector: every usable cluster
contributes at every native point. Completed corrected optimizer residuals remain
immutable. Overlay and signed residual geometry share that native plane; hover
extracts only a native row/column and never launches projection. Run/plane identity
and monotonically increasing request revisions reject stale deliveries. Model
planes have a separate 128 MiB/eight-entry bound; contour geometry remains bounded
to 32 MiB/eight entries. Experimental/Overlay share a locked intensity scale;
Residual has a separately retained signed scale.

One full accepted scientific result stays resident. Completion validation runs in
the existing worker and is published atomically; no private result serialization,
temporary recovery directory or history storage is introduced. Failure/cancellation
preserves the prior accepted result. A successful refit replaces it. Canonical
exports include both exact submitted and adopted working coordinates plus
currentness at export, and preserve the working peak state only when its exact
stable-ID tuple matches the exported completion.

Experimental contour geometry is cached independently of run/mode. An uncached
comparison request schedules current-plane experimental geometry before the model
in the existing computation lane, or immediately reuses cached experimental paths.
Those paths remain explicitly labeled experimental/pending until the matching
model/residual is ready. Both stages use the same request revision, so old-plane
and old-run deliveries cannot replace a newer view. Geometry extraction and model
projection remain outside Qt; fallback paths share the existing bounded cache.

The canonical JSON writer validates the document before writing it. The GUI
exporter checks artifacts and atomically publishes the staged directory; it does
not redundantly decode the full JSON again on a same-process worker, which held
the GIL long enough to stall Qt. Reader round trips remain scientific regressions. The existing export dialog
submits its writer from a queued Qt timer after entering the modal event loop,
so worker serialization does not compete with the initial modal transition.
The progress object is explicitly deleted after completion.
See [fit validation](validation/gui-v2-fitting.md) for measured bounds/limitations.

Protection: `test_gui_editor.py` checks real-data geometry, exact samples,
editing/cancel/Undo, actual Qt drag/pan routing, keyboard access and asynchronous
replacement; `test_gui_completion_transition.py` checks the atomic scientific
handoff independently of widgets. Reproduction and measured limits are in
[GUI v2 editor validation](validation/gui-v2-editor.md).
