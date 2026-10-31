# PeakFit Architecture Guide

This is a short map of the current repository, not an immutable contract. `AGENTS.md`
is the source of truth for AI-assisted work.

## Purpose

PeakFit fits lineshape models to pseudo-ND NMR spectra through a Python package and CLI.
The important domains are spectra and peak inputs, clustering, lineshapes, optimization,
fit orchestration, uncertainty analysis, plotting, and output generation.

## Current Package Map

- `src/peakfit/cli/` - Typer command-line entrypoints and terminal presentation.
- `src/peakfit/gui/` - PySide6/PyQtGraph desktop application, launched only by
  `peakfit gui` (`cli.commands.gui` imports Qt lazily, inside the command). Uses
  the existing `Spectra` model and input readers; working peak records and undo
  history (`gui.peaks`) are Qt-independent, and editable coordinates are separate
  from completed results. `gui.editor` holds the application shell and its views
  (see "GUI v2" below). Shared modules: `gui.spectrum` validates complete dataset
  handoffs and contour geometry; `gui.editing` provides the peak inspector and
  markers; `gui.background.ViewerComputation` is the serial latest-pending worker
  lane. `gui.theme` holds the light/dark tokens, palette, stylesheet and UI font;
  every GUI colour, including plot pens, comes from it. `gui.icons` draws the
  tinted tool icons and `gui.formatting` formats displayed numbers (display
  precision from axis digital resolution, value ± σ, axis powers of ten) without
  ever rounding stored values. `gui.fitting.prepare_dataset_fit` builds engine-safe peaks from snapshots
  and reuses engine clustering and pipeline parameter preparation;
  `gui.dataset_dialog` reviews effective settings and initial bounds
  asynchronously. `gui.dataset_fit` runs `run_pipeline_iter` with the reviewed
  configuration, including cross-cluster updates, and binds its `CompletedFit` to
  the original input snapshot; it does not implement another fitting scheduler.
  The GUI defaults to two passes; CLI defaults are unchanged.
  `gui.fit_process.FitProcess` runs one fit in a disposable spawned child and
  receives completed values off Qt. A private trusted-child pickle transport
  preserves mapping proxies and read-only NumPy buffers; it is not a saved-run
  format or an external reader. There is no fit queue; cancellation terminates and
  reaps the child. `gui.fit_display` evaluates the complete model on native planes
  through the scientific simulation API. `gui.fit_positions.dataset_completion`
  adopts usable fitted coordinates by stable peak ID. No automatic peak picking is
  involved.
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
4. `engine.scientific.final_outcome` freezes completed scientific truth by run-local
   `cluster_id` against the terminal correction revision, while
   `io.output_metadata` captures non-scientific run metadata; `io` reads inputs
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

Completed-model projection uses `engine.scientific.simulation.project_final_outcome`: paired
fractional grid coordinates and optional fitted-series column indices produce local or
combined contributions from the verified final snapshot and retained amplitudes.
The existing full-grid simulation delegates to this same path. Shape evaluation
is chunked; callers bound their output region. No optimization, amplitude solve,
or parameter-default preparation takes place during visualization.

`engine.scientific.completed.CompletedFit` retains the outcome, the existing model snapshot,
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


CLI and GUI scientific export both call `fit.export.write_completed_outputs`
with the same retained `CompletedFit`, coordinate context, metadata and output
options. It writes the lossless `scientific.npz` authority and calls shared
CSV/report projections (JSON is explicit legacy interoperability). Optional simulation is a shared float64 NPY export.
`gui.fit_export` only adapts retained inputs and stages publication; it has no
scientific writer or separate scientific schema. `gui.export_dialog` owns directory
selection and a short-lived background export thread. `engine.scientific.coordinates` supplies
one chemical-shift resolution rule to automatic dataset completion and exports.
A stable-ID table records exact initial/fitted coordinates, parameter links and
actual recorded adoption; export-time currentness is a separate context field.
CLI represents absent working-coordinate adoption explicitly. Intensity CSVs
retain both fitting-column and original source-plane identity. See [outputs](outputs.md).

The linear projection boundary in `engine.algorithms.linear_algebra` supplies
one rank-aware model space and amplitude pseudoinverse to optimization and
analytical output. A covariance failure does not invalidate an otherwise finite
model: frozen evaluations can carry unavailable uncertainties plus an explicit
warning. Numerical usability, optimizer termination and peak identifiability
remain distinct. See [the scientific overlap validation](validation/overlapping-peaks.md).


## GUI v2: open → inspect → edit → fit → compare → save → reopen

`gui.editor.window.launch` is the only application entry point, called by
`peakfit gui` with an optional spectrum (plus peak list, plane values and explicit
peak-axis mapping) or a saved `.peakfit` analysis. There is no runtime dependency
on `experiments/`.

The workspace has concrete responsibilities: `gui.spectrum.read_dataset` validates
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
Save analysis now preserves those plane values along with committed peaks and
accepted results. The loss guard offers Save analysis, Discard changes and Cancel;
CSV export alone does not preserve the analysis. Plane-value Undo is unchanged.

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
a worker and is published atomically after accepted evidence is durable.
History persistence and temporary ownership are described below. Failure/cancellation
preserves the prior accepted result. A successful refit replaces it. Projects
retain submitted/adopted working coordinates, manual revision and Currentness;
scientific exports retain the fitted model, exact original sample mapping and a
validated initial/fitted/adopted coordinate table. Export-time Current/Outdated
status accompanies that evidence without changing the immutable numerical outcome.

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


### External-source analysis persistence

`engine.scientific` owns immutable published results, sample identities and pure
model projection. `io.scientific` owns bounded typed-array serialization;
`io.readers.continuation` restores the recorded geometry/likelihood inputs for
both interfaces and MCMC. `gui.project_results` only adds editor metadata and
validates the automatic working-coordinate transition. `gui.project` owns atomic
manifest publication and verified external source links. Projects never own
experimental or simulated spectra.

Default exports retain published values, exact terminal corrections/mappings and
a small evaluator reference; detailed evidence adds observations/historical model
samples. Residuals are regenerated in the original operation order and checked
against recorded statistics. Bit-identical replay in one environment does not
replace evaluator tolerances. There is no solver/amplitude recovery on reopen.
Standard Save/Save as retain compact fitted authority: parameters, amplitudes,
uncertainties/covariance, diagnostics/provenance, exact terminal corrections,
observation dtype/digests, paired coordinates and plane identities. The small
component-basis reference validates evaluators, not a precomputed spectrum.
There are no extracted observations, sampled predictions or dense residual arrays
in projects. Missing sources permit numerical review. Exact observation recovery
and on-demand model/residual calculation share `io.readers.continuation`, including
its digest, evaluator-reference and terminal-statistics checks; no visualization
path solves amplitudes or refits parameters. Scientific export policies are unchanged.

`editor.persistence` supplies file actions, immutable capture and Qt publication;
`editor.numerical` supplies numerical review and lazy history. Requested sample
inspection materializes temporary arrays on its existing worker. History records
stay compact until explicitly selected. `gui.project_process` isolates archive
operations in a disposable spawned process; trusted IPC is not a disk format.
The experimental cube never crosses the save pipe. Routine Save reuses immutable
runs without source reads/hashing; a sibling PID lock protects writers. Payloads
are flushed before manifest replacement and obsolete runs are pruned afterward.
No embedded-spectrum serializer, portable mode or legacy project adapter remains.

`editor.fitting` keeps one active `DatasetCompletion`. Historical full evidence is
represented by `StoredRun` (project root + checksummed immutable entry), with no
implicit array-loading property. Reopen checks payload integrity but materializes
only the active record. `editor.numerical` lists metadata immediately and loads one
explicitly selected history record in a disposable process, releasing it on
selection change or inspector closure. History-list construction never decodes all
full runs. A summary-only eviction materializes the entire compact scientific
contract from the canonical record before publication can remove any full payload.
A reopened Full journal entry with `parameters=None` is therefore never merely
relabeled Summary only. Cancellation/failure leaves the last manifest intact.

`session_evidence.SessionEvidence` has one concrete role: durably spool accepted
outcomes before they replace resident data. It owns a single `peakfit-session-*`
temporary directory containing an ordinary `Recovery.peakfit`, not another format.
Writes run in a disposable process; the owner returns disk references and small
journals. Normal closure queues cleanup after in-flight publication, and successful
Save first rebinds every reference to the published project, then cleans scratch
storage. A transition to numerical-only review transfers ownership to the window,
so stopping spectral widgets cannot delete retained numerical history.

A checkpoint may precede final Qt adoption. Its writer preserves all input-run
payload leases until adoption, so a late cancellation cannot invalidate disk
references still held by the old workspace. Adoption retires obsolete leases on
the worker; cleanup is FIFO and cannot be coalesced or canceled at shutdown.

A hard interruption leaves the last atomic snapshot readable by the ordinary
project reader. **Recover interrupted session…** offers only PeakFit-owned
snapshots with dead POSIX owner PIDs, adopts the selected directory and requires
Save as to a permanent location. Discard/normal closure removes the adopted scratch
directory. Automatic discovery is conservative on non-POSIX systems; the documented
path can still be opened explicitly. Recovery checkpoints are linked projects:
source-free numerical review remains available, but spectral work needs relinking.
Recovery preserves accepted scientific state,
not every later unsaved coordinate draft. No general persistence service or shared
asset registry is introduced.

Ten full runs and protected evidence remain on disk; older entries retain exact
coordinates, fitted tables/errors/bounds, submitted parameters/settings and concise
diagnostics/statistics, explicitly without sample/profile capabilities. The 1 GiB
historical budget excludes active/protected evidence. New writes use project
schema 5.0.0, always referencing external sources; earlier project formats are
explicitly rejected.
Scientific schema/defaults remain unchanged. Explicit JSON remains a derived
optional scientific export, shared by both interfaces. See
[external-project validation](validation/external-projects.md) for measured sizes,
timings, publication memory and limitations.
