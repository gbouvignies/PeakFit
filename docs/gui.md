# Desktop spectrum viewer

Launch the desktop application with a 2D or pseudo-3D NMRPipe spectrum:

```bash
uv run peakfit gui examples/02-advanced-fitting/data/pseudo3d.ft2 \
  --peaks examples/02-advanced-fitting/data/pseudo3d.list \
  --z-values examples/02-advanced-fitting/data/b1_offsets.txt
```

Both options are optional. `uv run peakfit gui` opens an empty application window.
Use **File → Open spectrum** to open or replace the current spectrum, and the same
menu to load a peak list or plane values afterward. A missing peak list never
starts automatic peak picking. Other dimensionalities are rejected with an error.

![Spectrum viewer with the bundled experimental dataset](images/spectrum-viewer.png)

Native macOS capture; [A small dark window](images/spectrum-viewer-dark.png)
is also available.

## Navigation

- Drag the spectrum to pan; scroll to zoom about the pointer. **View → Reset view** or
  **Ctrl+0** restores the full spectral range.
- Click to place the crosshair, or drag either crosshair line. It snaps to sampled
  points so the two orthogonal traces and raw plane profile inspect the same data.
- Use the plane slider/spin box or **Ctrl+Page Up / Ctrl+Page Down**. Plane numbers start at
  1; the adjacent value is the experimental coordinate supplied by `--z-values`,
  or the original zero-based plane index when no values file is supplied.
- **Contours · 5.0× ▾** in the spectrum title shows the current threshold.
  Click it for precise threshold, spacing and level-count inputs and their sliders. The threshold slider is
  logarithmic (0.1–1000 × the first plane's estimated noise), with fine relative
  adjustments at low thresholds. Slider arrow keys make small changes; Page Up /
  Page Down make larger changes when a slider has focus. Blue solid contours are
  positive; red dashed contours are negative. Changes are briefly coalesced.
- Imported peaks appear as crosses. Hover to see a name and displayed coordinates;
  enable **View → Peak labels** for assignments. Indirect positions fold into the
  sampled window using PeakFit's coordinate conversions. Direct positions do not wrap.

Both right-hand traces follow the visible spectral ranges during zooming and
panning: the upper trace follows the horizontal (direct) axis, and the second
follows the vertical (indirect) axis. Both retain descending ppm orientation and
automatically scale intensity to the visible region. Navigate in the main spectrum;
the traces are inspection panels, with no independent pan/zoom. Both inspect the
crosshair position on the selected plane. The bottom panel shows
raw sampled intensity at the crosshair across all planes, not fitted amplitudes.
The larger point identifies the current plane. Points are deliberately not joined:
experimental coordinates may repeat or arrive out of order. Values have no assumed
physical units; their interpretation depends on the experiment.

Plot backgrounds and labels follow Qt's system palette, with contrasting contour,
trace, and crosshair colors for light and dark appearances. The horizontal splitter
lets you allocate space between the spectrum and inspection panels.

## Working peak coordinates

Click a marker to select its assignment. Coincident markers open a chooser. The
compact inspector edits the name and unfolded ppm coordinates together with Apply
or Enter; Escape restores the committed fields. Valid pending fields also apply
before selecting another peak, placing a new peak, dragging, or exporting. Invalid
fields remain visible with an actionable message. Delete removes the selected peak;
Edit → Undo/Redo restores working records without affecting spectrum navigation.
Peak-history Undo/Redo and peak deletion are unavailable while inspector text is
pending. Apply the text or press Escape to restore the committed fields first;
an inline hint explains this. Native text-field undo remains available.
Choose **Add peak…** (or A with spectrum focus), then click once to place a peak.
The same button becomes **Cancel placement**; click it or press Escape to cancel.
The spectrum tooltip describes the active placement gesture.
Inspector fields pause during placement and become editable again when it finishes.
Spectrum tooltips explain selection, navigation, and editing the selected peak.
Select a peak before dragging it: an open-hand cursor advertises movement. The ring and label preview the new position while the original cross
remains as a reference. Release commits one undoable move; Escape restores it.
Dragging an unselected marker pans. Peak coordinates retain fractional precision
and dragging preserves the original unfolded chemical-shift representation.

Imported shapes are discarded, and working peaks are not fitted results.

## Export and unsaved edits

**File → Export peaks…** (Command+S on macOS, Ctrl+S elsewhere) writes a canonical
PeakFit CSV (`name,F1_ppm,F2_ppm`) that can be loaded by the GUI or existing fitting
commands. Coordinates retain unfolded values and round-trip float precision;
quoted names, duplicate assignments, and names such as `001` and `NA` are preserved.
Exports contain names and coordinates, not internal identities, undo history,
plane values, or fitting configuration. Imported files are never changed implicitly.

The window's modified indicator and status bar identify unsaved edits, including
typed inspector fields. Opening another spectrum, replacing a peak list, and closing
the window offer Export/Discard/Cancel (the native dialog labels Export as Save).
Canceling export or encountering a write error keeps the working list. Undo/redo
continues across exports; returning to the saved coordinates clears the indicator.
Native text-field undo and deletion remain local to text; A and peak deletion
shortcuts require spectrum focus. Selection, zoom, crosshairs, and plane navigation
never enter peak history. Internal identities survive rename, move, and undo, but
are newly allocated on import; they are not scientific parameter identifiers.

![Manual peak editing on the bundled experimental spectrum](images/peak-editing.png)

[Drag preview with the original marker retained](images/peak-editing-drag.png).

![Highlighted one-shot placement and cancel control](images/peak-placement.png)

## Compact contour controls

The spectrum's existing title band contains one compact **Contours · 5.0× ▾**
control. Its value is the current threshold in multiples of first-plane noise.
Click once to edit the precise numerical threshold, logarithmic threshold slider,
spacing and level count in an anchored popup. The numeric input receives focus;
Enter commits a typed value, Tab moves between controls, and Escape or clicking
outside closes the popup. The button has a fixed width, so changing values never
moves it or the plots. It remains outside the spectral data area.

There is no separate toolbar, interaction banner, or permanent threshold editor.
Reset is available through **View → Reset view** and **Ctrl+0**. Plane navigation
remains below the spectrum. Interaction instructions are in spectrum tooltips;
contour progress/errors temporarily use the existing coordinate footer. Threshold
uses first-plane noise for its scale, not the independently estimated statistical
fitting noise. The numerical contour worker and debouncing are unchanged.

![Contour settings open from the spectrum title](images/contour-controls.png)

[Contour settings in a small dark window](images/contour-controls-dark.png).

During preparation, threshold changes still update the exact group and sampled mask.
Once submitted, a fit retains its original scientific inputs: contour changes during
execution or result inspection affect display only. Spacing and level count always
affect display only.

## Local fitting scope preview

Select an existing peak, then choose **Fit** beside **Inspect** in the
right-hand workspace. A compact group header replaces the peak editor; the F3
trace, F2 trace, and raw intensity across planes stay in their original positions
through preparation, execution, and results. **Inspect** restores the editor.
**Details** explicitly replaces the plots with scrollable numerical inspection;
**Close details** or returning to Inspect restores the same plots and navigation.
The header reserves the same height in every state, and details never resize the
main spectrum. Valid pending peak fields apply before
preparation, following the existing editor convention; invalid fields keep the
editor visible. Preparation does not execute optimization or change peak coordinates.

PeakFit constructs fresh scientific peaks from a snapshot of **all working peaks**,
resolves the existing lineshape models from NMRPipe processing metadata, estimates
noise from **all planes**, and uses the regular clustering algorithm. Safe internal
names and an explicit ID/assignment mapping keep duplicate assignments distinct.
The selected peak identifies its connected cluster, including every other working
peak in that cluster. The region combines thresholded signal from any plane with
small neighborhoods around working peaks; it is not a rectangular crop.

Amber cells show the **actual sampled fitting mask**, aligned with the spectral
samples, and rings highlight all included peaks. The preview lists group members,
plane/sample counts, resolved axis models, noise and clustering threshold, and the
nonlinear starting values, bounds, units, and fixed/variable flags from the same
parameter preparation function used by the pipeline. Positions remain unfolded;
direct-axis coordinates outside the sampled extent are rejected instead of wrapped.
The clustering threshold is the **absolute intensity of the lowest displayed
contour**, shared with the viewer's Threshold control (whose scale uses first-plane
noise). The optimizer's statistical noise is estimated independently from all planes;
changing the contour threshold does not change that estimate. No model or parameter
editing controls are exposed yet. Programmatic local preparation
supports only global and axis/parameter-wide wildcard rules (`*`, `*.*.lw`,
`*.F2.cs`, and the same pattern forms in fit-step fix/vary rules). It rejects
`parameters.peaks`, name-targeted patterns, and `parameters.from_file` with an
actionable error: assignment names cannot safely target the generated scientific
IDs, especially with duplicate assignments. Use the CLI for those configurations;
CLI constraint handling is unchanged.

The local fit is **one isolated pass across every experimental plane**.
Positions and lineshape parameters are shared across planes; signed amplitudes are
solved independently per peak and plane, with no amplitude starting controls or
assumed response law. Models inferred from processing metadata need scientific
review; a successful preparation does not validate their suitability for the data.
Signals from other clusters are **not** subtracted, and the complete iterative
cross-cluster correction procedure is not included. Check surrounding peaks before
using a local fit; a missing peak or significant external signal needs attention.

Panning, zooming, and plane navigation remain available. Preparing or dismissing a
preview does not change committed peaks, undo history, source files, or spectrum
samples. Selecting another member keeps the same prepared group without recomputing it.
During scope review, selecting an outside peak or clearing selection dismisses the preview and returns
to Inspect. Peak edits, peak-list replacement, and plane-coordinate changes also
dismiss it. Snapshots retain working IDs/assignments, spectrum identity and shape,
spectral metadata, plane coordinates, the effective absolute clustering threshold,
and fitting configuration. They are preparation inputs, not completed results
or a persisted session.

Changing **Threshold** while the Fit workspace is open updates membership, mask,
member rings, and effective parameters. Rapid slider changes are coalesced after a
short pause (150 ms); the previous scope stays visible at reduced opacity, marked
as updating until the new exact scope is ready. Spacing, number of levels, and displayed plane affect
rendering only and never redefine the cluster. The all-plane noise estimate is
reused for the unchanged dataset. Returning to Inspect cancels any pending update.

Preparation and numerical contour extraction run outside the GUI event loop in
one serial background lane. Each kind of request retains only its latest pending
version, and obsolete results are discarded. Working records and spectral metadata
are captured before submission. Only the selected cluster's data is materialized;
all-plane segmentation, boundary merging, and sampled-point order are unchanged.
Returning to Inspect, changing selection before a scope is ready, replacing the
spectrum, or closing the window invalidates the corresponding work without waiting.
Qt paths, widgets, and graphics are constructed only on the GUI thread.

![Local fitting scope with the bundled pseudo-3D spectrum](images/fit-scope-preview.png)

## Scope and performance

Manual editing changes one shared peak list across all experimental planes.
Local optimization and numerical result inspection are available. Completed local fits can be compared graphically; session persistence is not yet implemented. Existing
Matplotlib viewers and static plots remain available unchanged.

The viewer keeps the `Spectra` object in memory. Panning and zooming reuse retained
Qt graphics; raw traces and plane navigation remain available while work is pending.

Low-threshold contours remain fully available. Existing contours stay visible while
their replacement is computed, and the complete new set is installed together.
There is no contour-vertex cutoff. Progress uses a reserved one-line status area;
status changes do not move the spectrum or its controls.

Numerical preparation is separate from scope visualization. If a selected group's
data/correction arrays would exceed 128 MiB, its exact segmentation mask, all member
markers, model metadata, and sample counts still appear. No dense point-by-plane
arrays are allocated, and the parameter table remains unavailable. A compact status
explains that numerical preparation is unavailable; **Details**
provides the full explanation, required memory, and advice to use the CLI or a
higher threshold for numerical preparation. No region is truncated. Internally,
`PreparedLocalFit.scope` always contains geometry and membership; `cluster` and
`parameters` are explicitly absent when only the lightweight scope is available.

Fixed-height header and status areas keep the plots stable during progress,
warnings, and scope updates. Details open inline only on request.
The [native comparison](development.md#gui-responsiveness) covers the original
viewer and this workspace down to 0.1× noise.

Running preparation/contour work is not forcefully interrupted; it finishes in the
background and its obsolete result is ignored. Optimization uses a separate,
terminable process described below. Shutdown does not wait in the Qt event loop.
The first all-plane noise estimate still has substantial temporary memory use.

## Execute a reviewed local fit

After reviewing the prepared group, choose **Run local fit**. It executes the
exact sampled data, shape objects, starting values, bounds, and parameter freedom
shown in the preview. It does not repeat clustering or parameter preparation.
A scope-only preparation cannot run; the disabled button and **Details** explain
why. Configurations with no variable nonlinear parameters are also rejected.

One dedicated spawned process executes the existing VarPro single-cluster engine,
followed by the existing analytical evaluation and completed-outcome finalization.
Only the selected cluster and input metadata are transmitted, not the full spectrum.
The existing preview/contour worker remains independent. Status reports startup,
running time, completion/failure, and cancellation; it does not invent percentages.
**Cancel fit** terminates and reaps the numerical process, then restores editing.
No partial result is retained. Exiting during execution asks before canceling it.
Canceling any close, peak-discard/export, or result-discard confirmation keeps the
fit running. Shutdown begins only after every confirmation is accepted; results
that finish during an earlier dialog are checked by the final result-discard guard.

While fitting, zoom, pan, crosshair inspection, plane navigation, and ordinary
contour controls remain available. Switching Inspect/Fit and selecting another
peak retain the active fit. Scientific edits, Undo/Redo, peak-list/plane-coordinate
replacement, and opening another spectrum pause until the fit stops. The Fit
header and spectrum tooltip explain this restriction. Contour changes affect rendering only once a fit
has been submitted; its original fitting threshold and mask remain fixed.

The completed Fit workspace distinguishes **converged**, **usable but not converged**,
and **unusable** outcomes. Its compact header shows numerical usability and reduced
χ², labeled as an all-plane statistic over fitted samples. The termination message, evaluation
counts, χ², observation/parameter counts, and degrees of freedom are in **Details**.
Convergence describes optimizer termination and is not styled as scientific success.
The read-only table in **Details** switches between starting/fitted nonlinear
parameters (including bounds and fixed/variable flags) and independently recovered signed amplitudes for the displayed
experimental plane. Use the main plane controls to inspect other amplitudes;
repeated/unordered plane values retain their original plane identities.
**Details** retains model assumptions, statistical noise, the submitted threshold,
warnings, optimizer settings, software/run identity, timings, and child peak memory.
Use the graphical comparison described below to assess fit quality.

Working peaks, their undo history, and spectrum samples never change on completion.
The result separately owns the original input snapshot and stable GUI identities,
effective starting parameters, sampled data and exact grid indices, frozen nonlinear
values, amplitudes, model/residual arrays, analytical statistics, diagnostics, and
copied model geometry. Graphical comparison never repeats fitting.
Only one completed result is retained. A new fit requires confirmation before
replacing it; a canceled or failed attempt keeps the previous result, accessible
through **Retained result**. **Prepare again** starts another review.

Choose **File → Export fit results…** and select a parent directory. PeakFit writes
`local-fit-<run_id>/` containing canonical JSON 4.0.0, CSV tables and a local-fit
provenance record ([format details](outputs.md#gui-local-fit-export)). The footer
and confirmation show the destination. The completed fit and graphical comparison
remain available. Working coordinates, pending inspector fields, undo history and
peak-list saved status are unchanged. Exporting working peaks remains a separate action.

Closing, replacing the spectrum, or starting a replacement fit offers
**Export… / Discard / Cancel** for an unexported completed result. Canceling the
directory picker or a failed write keeps the result unexported. Already exported
results do not prompt again. Every new completed result has its own export state;
a failed/canceled replacement attempt retains the old result and its export status.
No shutdown/cancellation begins until all close decisions are accepted, including
when a fit finishes while an earlier dialog is open.

Later scientific edits mark the retained result outdated and remove its
current-spectrum scope overlays. Numerical inspection and export still use its
original scientific inputs, which the export companion records explicitly.
Obsolete background deliveries never replace it. Export is scientific output, not
an editable GUI session; reopening exported results in the GUI is not available.

### Apply fitted positions

After reviewing a numerically usable result, choose **Apply fitted positions** in
its Fit workspace. The confirmation shows the number of changed peaks and a compact
preview of indirect/direct ppm changes; **Show Details…** lists the whole group.
Cancel leaves working peaks untouched. No coordinates change automatically on fit
completion or scientific export.

Apply copies only the retained final chemical shifts, including fixed coordinates,
using engine identifiers mapped to stable working peak IDs. Duplicate assignments
remain distinct. Coordinate order follows the spectral axes, preserving unfolded
indirect shifts and existing aliasing. Missing, ambiguous, nonfinite, out-of-bounds,
or invalid direct-axis coordinates reject the entire operation. No parameters are
recalculated; linewidths and amplitudes are never applied to the working list.

One Apply is one **Undo peak edit** step; **Redo peak edit** restores the entire
group. Other peaks, assignments, ordering and selection stay unchanged. If every
position already matches, PeakFit explains that no edit was made and adds no undo
entry. Applying marks the working peak list modified; use **Export peaks…** to save
it independently of scientific results.

Apply requires unchanged scientific inputs and is unavailable during fitting or
with pending inspector edits. After applying, the header says **Outdated · comparison
hidden**: working inputs changed, so model/residual graphics and fitting-sample
overlays are hidden. The original fit, provenance, fitted quantities, and successful
export status remain intact. An unexported result can still be exported from
**File → Export fit results…**. Undo/Redo restores working coordinates without
reclassifying the retained fit as a new fit; prepare and run again for a current
comparison.

![Native confirmation for applying three fitted positions](images/apply-fitted-positions.png)

![Local fit running natively on macOS](images/local-fit-running.png)

## Comparing a completed local fit

A usable completed fit opens **Overlay** in the existing spectrum title band.
**Experiment** restores the experimental view; **Residual** shows only the actual
fitted-sample errors, with faint experimental contours for orientation. Changing
mode preserves the spectrum viewport and the three inspection plots.

Purple contours show the **local group contribution**, at the same absolute
positive/negative contour levels as the experiment. They do not simulate other
groups. Solid (negative: dashed) fragments lie in sampled native cells; dotted
(negative: dash-dot) fragments are predictions outside that support. The short
legend and selector tooltip explain this distinction. Open **Contours** and check
**Show exact fitting samples** for the original amber mask; it is off by default.
Changing display thresholds never changes the retained fit's inputs.

Smooth model contours evaluate the retained scientific lineshapes and signed
amplitudes on a bounded fractional grid. Bounds start at the sampled region plus
six fitted linewidths and double until visible tails no longer cross the boundary,
or the loaded spectral edge is reached. **Edge clipped** warns when visible model
tails still reach that edge. There is no wrapping of graphical paths across aliases.
Narrow lines receive additional near-center samples. These contours are display
predictions, not additional observations or an uncertainty estimate.

F3/F2 traces follow the same crosshair and native ppm coordinates as experiment,
with common intensity units and scales. Purple points/solid lines identify fitted
cells; dotted lines identify predictions. The evaluated model continues underneath
the solid fragments so the styles meet without a missing boundary segment. The scientific model is explicitly
evaluated along the native rows/columns, including tails outside the 2D grid.
Undefined residual samples stay absent; lines never bridge fitting-mask gaps.

The plane-profile selector distinguishes **Crosshair intensity** (experimental
circles and calculated total-group triangles) from each peak's **Amplitude**.
Amplitudes have their own signed scale and replace the intensity points. Profiles
keep the original plane order and use unconnected points, including repeated or
unordered plane values. Clicking a point navigates to its original plane.

**Residual** uses experimental minus local-model intensities at exact fitted
samples, divided by the retained statistical noise. The raster has a fixed
blue/neutral/red scale saturated at −3/+3 σ; neutral cells remain visible, and
unsampled cells are transparent. There is no residual interpolation. The traces
and coordinate profile show residual points in σ units; an unsampled crosshair
explicitly reports **No fitted sample at crosshair**. Neighboring unmodeled signals
may remain. Residual structure matters even when the optimizer converged.

Scientific edits remove comparison graphics immediately. Retained outdated results
remain numerically inspectable, but cannot restore current-spectrum model graphics.
Plane changes hide mismatched graphics until the current experimental/model pair
is ready. Existing editing, cancellation and unsaved-result guards still apply.

![Experimental and fitted local-group contours and traces](images/local-fit-overlay.png)

![Exact fitted-sample residuals](images/local-fit-residual.png)

[Compact dark residual view](images/local-fit-residual-dark.png) ·
[Compact dark overlay](images/local-fit-overlay-dark.png) ·
[Separate signed peak amplitude](images/local-fit-amplitude.png).
