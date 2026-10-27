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
or Enter; Escape restores the committed fields. Delete removes the selected peak;
Edit → Undo/Redo restores working records without affecting spectrum navigation.
Imported shapes are discarded, and working peaks are not fitted results.

## Scope and performance

This milestone provides read-only exploration. Peak editing, fitting, completed
model/residual display, and session persistence are not yet implemented. Existing
Matplotlib viewers and static plots remain available unchanged.

The viewer keeps the `Spectra` object in memory. Contour extraction happens on the
GUI thread when a plane or contour setting changes; zooming, panning, and crosshair
updates reuse graphical objects. There is no plane cache or background renderer.
Very large or densely contoured spectra can therefore pause during extraction;
reduce the number of levels or increase the threshold if needed.
