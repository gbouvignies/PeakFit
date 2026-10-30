# Fit, compare and export with GUI v2

The v2 workspace supports opening, inspecting, editing, whole-dataset fitting,
comparison and scientific export. Start it with:

```bash
uv run peakfit-editor
```

The existing `uv run peakfit gui` remains available while platform and feature
parity are evaluated. There is no change to CLI fits,
scientific output schemas or numerical algorithms.

For the bundled example:

```bash
uv run peakfit-editor examples/02-advanced-fitting/data/pseudo3d.ft2 \
  --peaks examples/02-advanced-fitting/data/pseudo3d.list \
  --peak-axis-order indirect-direct \
  --plane-values examples/02-advanced-fitting/data/b1_offsets.txt
```

Use **Open dataset…** for 2D/pseudo-3D NMRPipe spectra, optionally with a peak list
and plane-values file. Choose the peak-axis order explicitly from your file's
metadata. The reader's first/second coordinates map to the two spectral axes;
`indirect-direct` places the first coordinate vertically and the second
horizontally. No mapping is inferred from assignment names. A validated preview
shows actual axes, dimensions and example assignments before replacement. Cancel
or failed validation retains the current dataset. Loading can be canceled while
its private file read finishes.

The spectrum's top and right traces are native row/column cross-sections. Move
over the plot to preview them. **Pin slices**, Space on the canvas, or a click on
blank spectrum freezes the sample. Selecting a peak pins once; later blank clicks
and arrow-key sample steps retain selection. Resume preview without losing the
selected peak or its coordinate draft. Plane controls preserve acquisition order,
including repeated or unordered plane values. Missing values are labeled honestly;
imported values use the file's units, since a plain values file provides no unit
metadata.

The **Peaks** rail combines searching, selection and exact editing. Hover never
selects or overwrites a draft. Selection alone never opens a closed rail. Search
matches assignments and stable IDs. A filtered-out selection remains in the
inspector and can be revealed in the list. Coincident markers open a chooser with
coordinates and IDs, rather than silently choosing one assignment.

Use **Add peak** or canvas A, then click an exact ppm location. Select a marker,
then drag it to move; a ghost shows the draft while the committed marker stays.
Release commits one change; Escape or deactivation cancels the gesture. Working
positions are continuous unfolded float64 ppm, even when an indirect position
aliases into the display window. Only the inspection crosshair snaps to samples.

Name and coordinates form one editing group. Enter or leaving the whole group
commits; Tab within it retains the draft; Escape discards. Invalid text stays
visible with an error, while spectrum and plane navigation remain available.
Coordinates show eight decimal places when unfocused and round-trip precision
when focused/copied. Untouched fields retain their original float64 value.
Dot-decimal/scientific notation is accepted; a comma is also accepted when it is
the local decimal separator. Thousands separators and mixed punctuation are not.

Delete acts on the selected peak; native modifier selection in the list permits
one-step multi-delete. Undo/Redo restores exact records. In a clean coordinate
field, Cmd/Ctrl+Z invokes peak Undo; within a draft, ordinary text Undo remains
available. Undo labels identify the edit.

**File → Export working peaks…** writes canonical PeakFit CSV atomically, using
round-trip dot-decimal coordinates. Replacement/close offers export for unexported
peak edits. Importing or replacing plane values marks the session Modified, even
when no peaks changed. Peak CSV cannot preserve that plane mapping: exporting peaks
leaves it Modified, and close/replacement requires explicit **Discard session
changes** or Cancel. Keep the values file and select it again when reopening;
plane-value imports are session-only and are not undone by peak Undo.

Peak imports retain their source fingerprints by stable peak ID. Replace drops
superseded peak sources from the effective inputs; Append retains sources of all
current peaks. Undo/Redo restores the effective sources of the restored IDs,
including across later coordinate edits. Previous sources remain available only
for undoable IDs, rather than appearing as active scientific provenance.

## Fit → compare → export

Choose **Fit dataset** in the header or Analysis menu (Cmd/Ctrl+Return).
The first action opens a concise scientific-settings review. It prepares every
working peak and original plane, independently of selection, viewport, search and
display thresholds. The default is the existing **two-pass** whole-dataset
pipeline, including the neighbor correction update between passes. Resolved models,
statistical noise, threshold, cluster/sample counts and expanded actual starts,
bounds, units and freedom are reviewable before **Run dataset fit**.

Subsequent fits reuse those reviewed settings directly and prepare fresh masks,
clusters and bounds from the current optimized working coordinates. **Settings…**
reopens the review; Cancel leaves the current settings/result unchanged. Numerical
execution uses the existing disposable process outside Qt. The result line shows
pass/cluster/stage and elapsed time, with visible **Cancel**. Navigation, selection,
preview and pinning remain available; scientific edits and input replacement pause.
Cancellation, process failure and validation failure retain the previous result
and working coordinates. No partial result is adopted.

Successful completion automatically adopts each usable valid fitted coordinate
pair by stable peak ID. Usable non-converged fits and uncertainty warnings retain
their positions and warnings. Missing/unusable pairs keep their submitted positions
with explicit reasons. There is no Apply operation. Selection, viewport, plane,
rail visibility and free sample pins survive; a pin following a selected peak moves
to its optimized sample once. Pre-fit Undo is reset to avoid silently restoring
seeds under a Current label; the run retains exact submitted seeds independently.
Optimized working coordinates remain exportable as ordinary peak CSV.

A usable completed run opens **Overlay**, showing the sum of every usable cluster
at native coordinates and both aligned slices. Selection never isolates the model.
**Residual** shows `(native experiment − total usable model) / σ_stat`, including
model tails outside fitting masks. Default signed contours are ±3, ±5, ±8, ±13 and
±21 σ_stat; Display can scale these levels by changing the first residual contour.
Absence of contours means below the visible threshold. The horizontal
and vertical residual traces share a locked ±8 σ_stat range. Experimental/Overlay
share their original intensity range. **Display → Fit current traces** changes gain
explicitly and retains it through hover and plane navigation; clipping is labeled.
A newly accepted model can expand the intensity scale once, with an explicit notice.

Model planes compute asynchronously. Old-plane model/residual graphics disappear
immediately. While Overlay or Residual is pending, the 2D view shows the current
plane’s experimental contours with an explicit pending label. Cached experimental
geometry is reused; otherwise it is prepared off Qt before model projection.
No old plane or run is used as a fallback. Native measured inspection remains available;
residual slices wait for a matching model. Hover reads cached native rows/columns
and starts no model or contour jobs. Display caches are bounded to 128 MiB/eight
model planes and 32 MiB/eight contour geometries, with stale deliveries rejected.

The result line separates Current/Outdated, usable/total clusters, convergence,
completeness and uncertainty warnings. **Review** reveals concise selected-cluster
diagnostics in the rail's existing inspector scroll area. **Next cluster needing
review** explicitly navigates non-convergence, unusable results and uncertainty
warnings. Recorded optimizer statistics refer to corrected fitting samples and
are not recomputed from the displayed whole-spectrum residual.

Drafts retain the accepted comparison. A committed manual scientific change marks
the run **Outdated** and switches to Experimental. Undo alone cannot recertify it.
Refit to establish a new Current result. An all-unusable run remains inspectable
and exportable, with comparisons disabled and “No usable model”.

**Export results…**, also under File, uses the canonical scientific JSON/CSV writers
in a new `dataset-fit-<run_id>` directory. The provenance companion preserves exact
submitted seeds, stable identities, source fingerprints, original plane order,
actual correction/pass counts, post-fit coordinates, adoption warnings and
currentness at export. Outdated export requires an explicit acknowledgment and
exports that retained run's evidence. Raw spectra are never rewritten.

There is one resident immutable completed result. Cancellation and failure preserve
it. A successful refit replaces it; export results you need to keep before refitting.
Routine repeated fits introduce no discard/export prompt. There are no private
session pickle files or write-only recovery storage. Analysis save/reopening and
historical browsing are outside this milestone.

A successful canonical result export also preserves its exact post-fit working
peak tuple. When those are still the working coordinates, closing does not ask
for a duplicate peak export. Later manual edits and imported plane metadata keep
their existing close protection. Exporting an outdated run never marks different
working peaks as preserved. Failed or canceled exports leave the state unchanged.
MCMC and full cluster refinement remain out of scope.

**Display…** changes signed contour threshold, spacing/count, labels and stroke
weight. Trace amplitude limits are common, signed and locked across hover and
planes. **Fit current traces** explicitly changes gain once; a clipped trace is
identified in the sampled-value status. **Full spectrum** changes only frequency
ranges. Appearance follows System or explicit Light/Dark under View.

Keyboard access includes Cmd/Ctrl+O, Cmd/Ctrl+F, Cmd/Ctrl+Z/Shift+Z,
Cmd/Ctrl+0, Ctrl+Page Up/Down, native list arrows/Enter and canvas arrows/Space/A.
View also provides exact sample coordinates and frequency ranges. Help lists the
shortcuts. Delete and Space in text fields retain their ordinary text behavior.

See [the approved design](PeakFit_GUI_Design_Specification_v2.0.md) and
[editor validation](validation/gui-v2-editor.md) and
[fit/comparison validation, screenshots and limitations](validation/gui-v2-fitting.md).
