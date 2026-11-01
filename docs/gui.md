# Desktop application

`peakfit gui` is the PeakFit desktop application. It opens 2D and pseudo-3D NMRPipe
datasets, edits working peaks, fits the complete dataset, compares the model with
the experiment, exports scientific results and saves analyses that can be reopened.

## Launch

```bash
uv run peakfit gui
```

opens an empty window. To open a dataset at startup:

```bash
uv run peakfit gui examples/02-advanced-fitting/data/pseudo3d.ft2 \
  --peaks examples/02-advanced-fitting/data/pseudo3d.list \
  --peak-axis-order indirect-direct \
  --z-values examples/02-advanced-fitting/data/b1_offsets.txt
```

To reopen a saved analysis, pass its `.peakfit` directory:

```bash
uv run peakfit gui Analysis.peakfit
```

`--peaks` requires `--peak-axis-order`: `indirect-direct` maps the list's first
coordinate to the vertical (indirect) axis and the second to the horizontal (direct)
axis; `direct-indirect` swaps them. Choose it from the peak list's metadata; no
mapping is guessed. `--peaks`, `--z-values` and `--peak-axis-order` require a
spectrum and cannot be combined with a `.peakfit` analysis, which records its own
inputs. These combinations are rejected before the window opens. Files are then
validated off the UI thread; a startup loading failure is reported in a dialog,
and the window stays open so you can choose **Open dataset…** or **Open analysis…**.

## Open, inspect and edit

Use **Open dataset…** for 2D/pseudo-3D NMRPipe spectra, optionally with a peak list
and plane-values file. Choose the peak-axis order explicitly from your file's
metadata. The reader's first/second coordinates map to the two spectral axes;
`indirect-direct` places the first coordinate vertically and the second
horizontally. No mapping is inferred from assignment names. A validated preview
shows actual axes, dimensions and example assignments before replacement. Cancel
or failed validation retains the current dataset. Loading can be canceled while
its private file read finishes.

The spectrum's top and right traces are native row/column cross-sections. Move
over the plot to preview them. The **Pin traces** tool, Space on the canvas, or a click on
blank spectrum freezes the sample. Selecting a peak pins once; later blank clicks
and arrow-key sample steps retain selection. Resume preview without losing the
selected peak or its coordinate draft. Plane controls preserve acquisition order,
including repeated or unordered plane values. Missing values are labeled honestly;
imported values use the file's units, since a plain values file provides no unit
metadata.

The workspace has three columns: the **navigator** on the left, the spectrum in
the centre and the **inspector** on the right. Toolbar buttons at either end hide
or show the side columns; their widths are adjustable. The navigator's **Peaks**
tab lists working peaks with search; after a fit its **Clusters** tab lists the
active run's clusters. The inspector shows the selected peak (position editor,
amplitude profile, fitted values and cluster review) or, with nothing selected,
the analysis: the fit setup and the active run. Hover never selects or overwrites
a draft. Selection alone never opens a hidden column. Search matches assignments
and stable IDs. A filtered-out selection remains in the inspector and can be
revealed in the list. Coincident markers open a chooser with
coordinates and IDs, rather than silently choosing one assignment.

Use **Add peak** or canvas A, then click an exact ppm location. Select a marker,
then drag it to move; a ghost shows the draft while the committed marker stays.
Release commits one change; Escape or deactivation cancels the gesture. Working
positions are continuous unfolded float64 ppm, even when an indirect position
aliases into the display window. Only the inspection crosshair snaps to samples.

Name and coordinates form one editing group. Enter or leaving the whole group
commits; Tab within it retains the draft; Escape discards. Invalid text stays
visible with an error, while spectrum and plane navigation remain available.
Coordinates show one decimal place beyond the axis digital resolution when
unfocused (four in ¹H and three in ¹⁵N for the bundled dataset) and round-trip
precision when focused/copied. Untouched fields retain their original float64 value.
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

The **fit setup** is the inspector's analysis page, shown whenever nothing is
selected. It edits the working settings directly: lineshape model (each axis's
resolved model with its source: NMRPipe apodization, the pseudo-Voigt default,
shown as a caution, when no apodization is recognized, or explicit), clustering threshold as a factor of the statistical
noise with its absolute intensity, optimizer passes (default **two**, with the
neighbor correction update between passes; an imported step schedule replaces
the field) and, under **Advanced**, iteration limit and tolerance. Statistical
noise is shown with its source. A committed setting change marks an existing run
Outdated; settings are not part of peak Undo.

After every committed change to peaks or settings the setup prepares the exact
fitting problem off Qt, with the same preparation that Fit submits, and reports
cluster and sample counts. Preparation never depends on selection, viewport,
search or display contours. An obsolete preparation is hidden at once. When
preparation fails, **Fit** is disabled and the setup and its tooltip give the
reason. The member rings are the default sign of a selected peak's cluster.
**Show sampled fitting regions** (off by default) adds a faint tint behind the
contours, one cell per sampled point, with the selected peak's cluster slightly
stronger. The legend names the source: "recorded in run …" while the active run
is Current, otherwise "preview for current settings". The tint never changes the
viewport, comparison mode or model scope.

Choose **Fit** in the toolbar or **Fit dataset** in the Analysis menu
(Cmd/Ctrl+Return). Fit commits a valid draft and submits the prepared problem;
it always covers every working peak and original plane. There is no confirmation
dialog: the first Fit of a session whose setup has not been displayed opens it
instead of running, and the next Fit runs. The settings summary beside **Fit**
(lineshapes · clustering threshold · passes; "Fit settings" in windows narrower
than 1200 px, with the summary in its tooltip) and **Analysis › Fit setup** open
the setup. Numerical execution uses the existing disposable process outside Qt. The run line beside the
analysis name shows pass/cluster/stage and elapsed time, with **Cancel** beside **Fit**;
the settings summary steps aside while a fit runs. Navigation, selection,
preview and pinning remain available; scientific edits and input replacement pause.
Cancellation, process failure and validation failure retain the previous result
and working coordinates. No partial result is adopted.

Successful completion automatically adopts each usable valid fitted coordinate
pair by stable peak ID. Usable non-converged fits and uncertainty warnings retain
their positions and warnings. Missing/unusable pairs keep their submitted positions
with explicit reasons. There is no Apply operation. Selection, viewport, plane,
side-column visibility and free sample pins survive; a pin following a selected peak moves
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

The run line starts with the run and Current/Outdated, then completeness,
convergence and usable/total clusters, so a narrow window shortens only the tail;
its tooltip keeps the full text. Clusters with a numerical warning have their own
amber label (▲ N) that never elides.

The first completed run opens the navigator's **Clusters** tab: one row per
recorded cluster with its members and three measures: χ²ᵣ, the largest |recorded
residual| in σ (|r|max) and the largest |amplitude correlation| between members
("–" for a single peak, "n/a" when the covariance is unavailable). The first
column shows an amber ▲ for a numerical warning (unusable, not converged,
uncertainty unreliable, or a varied parameter at a recorded bound) or else a
subdued ◇ for a highlight; its tooltip names each one. Highlights are descriptive
rules, never warnings: amplitude correlation |ρ| ≥ 0.9; a position shift
(|fitted − initial| in Hz, over the fitted linewidth, on the larger axis) above
0.5; a fitted linewidth outside 0.5–2× the run's median for that axis over usable
peaks. Each highlight states its rule and value, for example "shift 1.03 × LW
(¹⁵N)". **Display** sets the thresholds for this session; they change no results,
warnings or run state. The menu above the list shows all clusters, any warning,
any highlight, or one warning or highlight type, each with its count. Columns sort
on click; the default is χ²ᵣ, highest first. Choosing a
row, or **N**, opens the inspector's **Cluster** page and selects one of its
members (keeping the selected peak when it belongs to the cluster), so the cluster
is the unit of review while a peak stays the unit of editing. Under its header, a
compact amber line names the cluster's numerical warnings (“▲ Uncertainty
unreliable · see Review”); their explanations stay in Review. The cluster page
lists the numbered members, each a link to its peak page; the amplitude profile of
all members; a parameter table of every member's recorded values at display
precision (± σ on hover, recorded bounds in amber); the recorded amplitude
correlation matrix in member numbers, shaded by |ρ|, or why it is unavailable or
not applicable; the cluster's warnings, measures and highlights; and provenance.
On a peak page, **Review cluster of N ›** opens its cluster page; selecting a peak
on the spectrum or in the Peaks list opens its peak page. Whenever the selected peak belongs to the active run, the other members
of its recorded cluster carry a thin accent ring on the spectrum, while the
selected peak keeps its stronger ring and label; this never changes the viewport,
comparison mode or the total-model scope. **N** in the cluster list, or **Select next cluster with a warning** on
the analysis page, moves in list order to the next cluster with a numerical
warning that the current filter shows; it never selects a hidden row. The
inspector's **Cluster review** keeps three families apart: numerical warnings
(with the engine's own text), highlights (with their rule), and measures: χ²ᵣ with its sample count, the largest recorded
residual against the maximum expected from Gaussian noise for that many samples,
and the selected peak's strongest amplitude correlation, conditional on the
fitted lineshapes. Recorded optimizer statistics refer to corrected fitting
samples and are not recomputed from the displayed whole-spectrum residual.
**Provenance** discloses the optimizer, termination message, correction revision
and run identity. **Fitted** lists the peak's recorded parameters with σ and
units; parameters at a recorded bound are marked. **Starting values and bounds**
(collapsed) shows the selected peak's parameters as the next fit would submit
them: start, range, unit and whether each is varied or fixed, from the current
preparation of the working position and fit setup. They are read-only; while
preparation runs or fails the section says so rather than showing old values.

The inspector's **Amplitude profile** shows the selected peak's signed fitted
amplitudes across planes, with the other members of its cluster lighter on the
same vertical scale. It follows the accepted selection only: an invalid draft that
blocks selection leaves the profile unchanged. The inspector header names the
peak, its cluster size, the source run and Current/Outdated state.

Points are the recorded amplitudes, unconnected, against the supplied plane values
in acquisition order; repeated values stay separate observations. Without plane
values, or when the current values differ from those the run recorded, the axis is
the one-based plane number. Clicking a point opens the plane of the nearest
point; points sharing that exact value offer a choice. Error bars are the scaled analytical standard errors written
by **Export results…**: computed with the fitted lineshape parameters held fixed,
multiplied by √χ²ᵣ when the cluster's χ²ᵣ exceeds 1, one σ per peak for every
plane. They can underestimate the uncertainty of overlapping peaks; warned
clusters use hollow markers and dashed bars. When the core reports no finite σ for
a peak, its amplitudes stay visible without error bars and are labeled
**σ unavailable**. When at most two observations sit
beyond a gap wider than half the span, such as a distant reference plane, the
plot frames the remaining range and labels the off-scale points; **Show all
points** shows them. Outdated runs stay visible and labeled; peaks added after a
run and unusable clusters show why there is no profile.

The profile shows **Raw amplitudes** by default. Its quantity chip offers
**Choose reference planes…**: enter one or more plane numbers (for example `1` or
`1-3`); the dialog points out isolated planes but never chooses one. The profile
then shows each peak's ratio A / A_ref to the mean of its own amplitudes on those
planes, with ±1σ propagated to first order using the full gradient, so a plane in a
multi-plane reference is treated as correlated with the mean. A single reference
plane is drawn exactly at 1 without an error bar; reference planes use diamonds.
Normalization is a display: recorded amplitudes, σ, exports and saved results are
unchanged, and the reference lasts for the session.

When a ratio cannot be shown, the profile keeps the raw amplitudes and says so
directly under the plot (“▲ Showing raw amplitudes …”); the quantity chip reads
“÷ plane 1 · not applied” and the axis names what is plotted. A ratio is
**undefined** when the reference mean is zero or not finite, when σ is
unavailable, or when a reference plane is not in the run. It is **withheld as
unreliable**, although computable, when σ_R/|R| exceeds 0.1 or when the engine
marks the cluster's uncertainties unreliable: a degenerate decomposition can
inflate amplitudes and σ by one arbitrary scale that the ratio would cancel,
giving precise-looking values for amplitudes the data do not determine. The
reference choice is kept, so the next well-determined peak is normalized again.

Drafts retain the accepted comparison. A committed manual scientific change marks
the run **Outdated** and switches to Experimental. Undo alone cannot recertify it.
Refit to establish a new Current result. An all-unusable run remains inspectable
and exportable, with comparisons disabled and “No usable model”.

**File → Export results…** uses the shared scientific archive and
CSV/report writers with an optional detailed-evidence checkbox in a new `dataset-fit-<run_id>` directory. The stable-ID
coordinate table and export-state metadata in `scientific.npz` preserve exact
submitted seeds, stable identities, source fingerprints, original plane order,
actual correction/pass counts, post-fit coordinates, adoption warnings and
currentness at export. Outdated export requires an explicit acknowledgment and
exports that retained run's evidence. Raw spectra are never rewritten.

The **active result** is the one completed run shown in the comparison and result
line. Cancellation and failure leave it unchanged. A successful refit makes the new
run active; the previous run is not discarded but stays in the **analysis history**
(**File → Analysis history…**, see below), so routine refits need no discard/export
prompt. Three operations are distinct:

- **Analysis history** retains earlier completed runs within the session, under the
  retention limits described below.
- **Save analysis** writes the working state, the active result and the retained
  history to a `.peakfit` directory that references the external spectrum.
- **Export results…** writes one run's canonical scientific outputs for use outside
  PeakFit; it does not save the analysis.

All three use the same scientific evidence. See [formats and migration](outputs.md).

A successful canonical result export also preserves its exact post-fit working
peak tuple. When those are still the working coordinates, closing does not ask
for a duplicate peak export. Later manual edits and imported plane metadata keep
their existing close protection. Exporting an outdated run never marks different
working peaks as preserved. Failed or canceled exports leave the state unchanged.
MCMC and full cluster refinement remain out of scope.

The **Display** tool changes signed contour threshold, spacing/count, labels and
stroke weight. Contours are graded from light low levels to strong high levels;
measured data is blue (negative plum, dashed), the model amber (negative dotted)
and the residual vermilion/teal, as named in the in-plot legend. Peak markers
keep a plot-coloured halo so they stay visible over contours; the selected and
hovered peaks are labelled with their assignment only, and resting on a marker
shows its exact unfolded (and displayed alias) coordinates in a tooltip. Trace axes show a common power of ten (for example ×10⁶). Trace amplitude limits are common, signed and locked across hover and
planes. **Fit current traces** explicitly changes gain once; a clipped trace is
identified in the sampled-value status. The **Full spectrum** tool changes only frequency
ranges. Appearance follows System or explicit Light/Dark under View.

Keyboard access includes Cmd/Ctrl+O, Cmd/Ctrl+F, Cmd/Ctrl+Z/Shift+Z,
Cmd/Ctrl+0, Ctrl+Page Up/Down, native list arrows/Enter and canvas arrows/Space/A.
View also provides exact sample coordinates and frequency ranges. Help lists the
shortcuts. Delete and Space in text fields retain their ordinary text behavior.

See [the approved design](PeakFit_GUI_Design_Specification_v2.0.md) and
[editor validation](validation/gui-v2-editor.md) and
[fit/comparison validation, screenshots and limitations](validation/gui-v2-fitting.md).

## Save, reopen and continue

Use **Open dataset…** to load a spectrum, optional peak list and plane values.
Confirm the peak-axis mapping. Edit unfolded positions and choose **Fit dataset**;
usable optimized positions are adopted automatically, while initial positions stay
with the immutable run. Edit again to mark it Outdated, then refit normally.

**File → Save analysis** (Ctrl+S / Command+S) creates a `.peakfit` **directory**.
**Open analysis…** uses Ctrl+O / Command+O; **Save analysis as…** uses the standard
platform Save As shortcut. Open dataset moves to Ctrl+Shift+O / Command+Shift+O.
Save As requires a new destination. Keep the whole directory together. Routine
saves show brief status-bar progress without a dialog; expensive archive and source
verification work runs outside Qt, with cancellation available in the status bar.
Closing changed work offers **Save analysis / Discard changes / Cancel**. Exporting
peaks (CSV) or results is separate and does not save the analysis; an unsaved
session's history is lost on Discard.

Projects reference the original NMRPipe spectrum; they never copy it. Working peak
IDs/positions, effective settings, ordered plane values, accepted numerical evidence,
original seeds and the automatic working transition survive reopening without the
imported peak/plane-value files. The selected plane/peak, inspection pin, viewport,
side-column visibility and widths and comparison/display settings also persist. Undo starts afresh.

Reopen verifies spectrum SHA-256, calibration/processing, sample identity and model
evaluation before enabling spectral comparisons or new fits. It never optimizes or
solves amplitudes. Missing, changed, corrupt or unsupported sources/models open
**Numerical review only**, with fitted parameter/error tables, original starting
peaks, signed amplitude profiles, diagnostics/settings and exact sample inspection
and export. **Relink source…** verifies your explicit candidate. An identical moved
file restores spectral views; a different spectrum cannot inherit the old results.
Corrupt project evidence and unsupported project versions are explicit errors.

**File → Analysis history…** inspects retained runs. The latest ten completed runs
retain full evidence; older runs have explicitly labeled Summary-only journal
entries. **Keep full evidence** protects a full run from pruning. Historical evidence
has a 1 GiB budget, excluding the active and protected runs; their costs appear in
the manifest and Save status. Only the active complete result stays in memory;
history loads when selected. Each accepted result has a durable temporary snapshot
before resident replacement. Save and normal closure clean owned scratch storage.
After an interruption, **File → Recover interrupted session…** offers retained POSIX
session snapshots and requires Save as to a permanent project. Recovery is not a
saved analysis or an autosave of later edits. On other systems an explicit Open
analysis can read the recovery path shown in diagnostics. Source-dependent MCMC, full refinement controls, source
embedding and reopening legacy local-fit exports remain outside this workflow.

See [native workflow evidence, measurements and limitations](validation/gui-v2-persistence.md)
and [project format](outputs.md#linked-source-analysis-projects).
