# Inspect and edit with GUI v2

Milestone 1 is a usable spectrum and peak editor. Start it with:

```bash
uv run peakfit-editor
```

The existing `uv run peakfit gui` remains the fitting application until the v2
workspace reaches whole-dataset fitting parity. There is no change to CLI fits,
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
peak edits. Exporting peaks is not saving an analysis: projects, fits, MCMC,
refinement and result persistence arrive in subsequent vertical slices. Overlay
and Residual remain visible with explanations of their unavailability.

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
[production validation, screenshots and limitations](validation/gui-v2-editor.md).
