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

Native macOS capture; [dark appearance](images/spectrum-viewer-dark.png) is also available.

## Navigation

- Drag the spectrum to pan; scroll to zoom about the pointer. **Reset view** or
  **Ctrl+0** restores the full spectral range.
- Click to place the crosshair, or drag either crosshair line. It snaps to sampled
  points so the two orthogonal traces and raw plane profile inspect the same data.
- Use the plane slider/spin box or **Ctrl+Page Up / Ctrl+Page Down**. Plane numbers start at
  1; the adjacent value is the experimental coordinate supplied by `--z-values`,
  or the original zero-based plane index when no values file is supplied.
- Use the sliders for visual adjustment of threshold, geometric spacing, and level
  count; adjacent spin boxes allow precise entries. The threshold slider is
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
The instruction strip above the spectrum highlights **PLACING PEAK**, and the
same button becomes **Cancel placement**; click it or press Escape to cancel.
Inspector fields pause during placement and become editable again when it finishes.
The strip otherwise explains selection, navigation, or editing the selected peak.
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

## Local fitting scope preview

Select an existing peak, then choose **Fit group…** beside **Inspect** in the
right-hand workspace. The preview replaces the editor and inspection plots;
**Back to Inspect** restores them. Valid pending peak fields apply before
preparation, following the existing editor convention; invalid fields keep the
editor visible. No optimization, fit results, or coordinate acceptance is available
in this step.

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
The scientific clustering threshold defaults to 5 times the all-plane noise and is
independent of the display threshold, which still uses first-plane noise. No model
or parameter editing controls are exposed yet. Programmatic local preparation
supports only global and axis/parameter-wide wildcard rules (`*`, `*.*.lw`,
`*.F2.cs`, and the same pattern forms in fit-step fix/vary rules). It rejects
`parameters.peaks`, name-targeted patterns, and `parameters.from_file` with an
actionable error: assignment names cannot safely target the generated scientific
IDs, especially with duplicate assignments. Use the CLI for those configurations;
CLI constraint handling is unchanged.

The proposed fit is **one isolated local pass across every experimental plane**.
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
Selecting an outside peak or clearing selection dismisses the preview and returns
to Inspect. Peak edits, peak-list replacement, and plane-coordinate changes also
dismiss it. Snapshots retain working IDs/assignments, spectrum identity and shape,
spectral metadata, plane coordinates, and fitting configuration. They are preparation
inputs, not completed results or a persisted session.

Preparation is synchronous and performed only on explicit entry, not on navigation;
it does not rebuild experimental contours. Large datasets can briefly pause during
noise estimation/segmentation. There are no background workers in this step.

![Local fitting scope with the bundled pseudo-3D spectrum](images/fit-scope-preview.png)

![Three overlapping peaks and their exact sampled fitting region](images/fit-scope-closeup.png)

## Scope and performance

Manual editing changes one shared peak list across all experimental planes.
Optimization, completed model/residual display, and session persistence are not yet
implemented. Existing Matplotlib viewers and static plots remain available unchanged.

The viewer keeps the `Spectra` object in memory. Contour extraction happens on the
GUI thread when a plane or contour setting changes; zooming, panning, and crosshair
updates reuse graphical objects. There is no plane cache or background renderer.
Very large or densely contoured spectra can therefore pause during extraction;
reduce the number of levels or increase the threshold if needed.
