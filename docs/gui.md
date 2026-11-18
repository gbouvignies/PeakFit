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
metadata: **Indirect, then direct** (`indirect-direct`) places the first coordinate
vertically and the second horizontally; **Direct, then indirect** swaps them. No
mapping is inferred from assignment names. A valid dataset opens directly. When
more than half of the peaks lie outside the direct (horizontal) axis window with
the chosen mapping, PeakFit says how many, and how many the other mapping would
leave outside, and asks before opening (or importing peaks); it never picks the
mapping. Indirect positions outside the window are not counted, since they can be
folded peaks. Cancel or failed validation retains the current dataset. Loading can
be canceled while its private file read finishes.

The **row slice** above the 2D plot and the **column slice** to its right are native
row/column cross-sections: lines through the native samples, with no markers and
no values between samples. Each shares its frequency axis exactly with the 2D plot.
Move over the plot to preview them. The **Pin slices** tool, Space on the canvas, or
a click on blank spectrum freezes the sample. Selecting a peak pins once; later blank clicks
and arrow-key sample steps retain selection. Resume preview without losing the
selected peak or its coordinate draft. Choosing a peak in the Peaks list, a cluster
row, a change summary or warning navigation centres it at the current zoom within
the data: the view never extends past the spectrum's extent, and at full spectrum
it does not move; a click
on the spectrum or a member link on a cluster page leaves the view where it is, and
cluster focus keeps its frame. Plane controls preserve acquisition order,
including repeated or unordered plane values. Missing values are labeled honestly;
imported values use the file's units, since a plain values file provides no unit
metadata.

**Names, numbers and controls** are the same everywhere. A spectral axis is named
by its nucleus, as the file's header records it, and its F-dimension: ¹⁵N F2,
¹H F3 (an unlabelled nucleus reads N F2), so two ¹H axes stay distinct; units
follow in parentheses, ¹H F3 (ppm). The plane axis is “Plane value” when plane
values are loaded and “Plane” otherwise. A value with a standard error shows the
σ to two significant digits and the value to the same place, sharing one power of
ten: (3.10 ± 0.13) × 10⁹; a table row or column of one magnitude carries the
× 10ⁿ in its header instead. A missing σ reads “not assessed”. ppm values have
three decimals, Hz one, χ²ᵣ two, residuals in σ one and correlations two; times are
local with their zone. Controls have one look per kind: a bordered button runs a
command, the filled **Fit** button is the main command, a checkbox is an on/off
state, joined segments are one choice, a button ending in ▾ opens a menu, and
accent text navigates.

The workspace has three columns: the **navigator** on the left, the spectrum in
the centre and the **inspector** on the right. Toolbar buttons at either end hide
or show the side columns; their widths are adjustable. The toolbar starts with
PeakFit; the window title names the dataset. PeakFit never enlarges the
window: a side panel opens by taking room from the spectrum (residual lanes, then
slices, down to their minimums) and then from the other side panel. When even the
minimums do not fit, it stays closed and a message says what makes room. When the
window is too narrow for the residual lanes, they give way with a message in every
comparison mode alike; your lane choice is kept and the lanes return once there is
room. A new or
reopened analysis keeps the spectrum, then the inspector, then the navigator.
1160×720 holds both side panels with slices and residual lanes. The navigator's **Peaks**
tab lists working peaks with search; after a fit its **Clusters** tab lists the
active run's clusters. The inspector shows the selected peak (position editor,
amplitude profile, fitted values and cluster review) or, with nothing selected,
the analysis, on its **Run** or **Model** page. Hover never selects or overwrites
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
Coordinates show three decimals in ppm when unfocused, as every ppm value in the
GUI does, and round-trip precision when focused/copied. Untouched fields retain their original float64 value.
Dot-decimal/scientific notation is accepted; a comma is also accepted when it is
the local decimal separator. Thousands separators and mixed punctuation are not.

Delete (Backspace on a Mac keyboard) on the spectrum or in the Peaks list,
**Edit → Delete selected peaks** (its key shown) and the spectrum's and list's
context menus delete the selected peak as one Undo entry, without confirmation;
the peak page has no Delete button. Right-clicking a marker selects it before the
menu opens. Native modifier selection in the list permits one-step multi-delete.
Undo/Redo restores exact records. In a clean coordinate
field, Cmd/Ctrl+Z invokes Undo (peak and settings edits); within a draft, ordinary text Undo remains
available. Undo labels identify the edit.

**File → Export working peaks…** writes canonical PeakFit CSV atomically, using
round-trip dot-decimal coordinates. A peak list is not the analysis: exporting peaks
or results leaves it **Modified**.

**Modified** has one meaning, shown by the toolbar, the window title and the close
prompt alike: the analysis differs from the last one saved or opened, or an edit is
still a draft. The toolbar tooltip names what differs: peaks, fit settings, the
active run, input files, plane values, kept runs, run history or an uncommitted
draft. Importing or replacing plane values marks it Modified even when no peaks
changed; Undo does not undo a plane-value import, and peak CSV cannot carry
the plane mapping. **Save analysis** records it.

Peak imports retain their source fingerprints by stable peak ID. Replace drops
superseded peak sources from the effective inputs; Append retains sources of all
current peaks. Undo/Redo restores the effective sources of the restored IDs,
including across later coordinate edits. Previous sources remain available only
for undoable IDs, rather than appearing as active scientific provenance.

## Fit → compare → export

With nothing selected, the inspector shows the **analysis** on one of two pages,
under the dataset's name and a **Run | Model** switch. **Model** holds the working
fit settings and is shown before the first fit; **Run** holds the active run as
recorded and comes forward when a run completes or an attempt fails. The switch,
the toolbar settings summary (or **Analysis ▸ Model settings**) and a click on the
run summary (or **Analysis ▸ Active run**) choose the page, which then stays until
the next run or failed attempt. Run is unavailable, and says so, until there is a
run, a failed attempt or a fit in progress.

The **Run** page reads as label rows: Run (its number, short ID and Current or
Outdated, the full ID in the tooltip), Model, Converged and Usable as counts of clusters, Fitted
(date and local time with its zone, and duration), Warnings when any cluster has
one, with **Select next cluster with a warning** right under the rows, and Noise
check as a one-line verdict: **Consistent with the recorded processing**,
**Inconclusive on …** or **Not checked** (why, behind its ⓘ), or, for a mismatch,
the axes and that the errors may be miscalibrated in the warning style, with what
was found and what to do behind its ⓘ, in the wording of the run line's tooltip. A recovery problem adds
a row in the warning style. The changes from an earlier run and the unexplained
signal follow (below). The summary names the run and whether it is Current; it
gives cluster counts only when some cluster is unusable or not converged (always
in its tooltip), and the run is named there only, with “Outdated” repeated where an
outdated run's values are shown.

The **Model** page edits the working settings directly. Its **Lineshape model** is
a table with one row per spectral axis, in the engine's axis order: **Axis**,
**Family**, **J (Hz)**, **Fit J** and **Phase**. Family reads **Auto · SP2** when
Automatic resolves to SP2 from that axis's apodization; when no apodization is
recognized the row carries a caution under it (“No recognized apodization. Check
that the pseudo-Voigt default suits these data, or choose a family.”), which does
not block the fit. A single family for every axis is stored once, as
`[fitting].family`; a family that differs from the others is stored on that axis.
A singlet's J cell offers **+ J**: it opens the J field, with “Enter J in Hz; leave
it empty to cancel.” under the axis, and the doublet exists once a positive J in
Hz is entered (an empty J cancels it). Its range, by default
[J/2, 2J], then sits on a line under the row, **J range [low] – [high] Hz**, with ×
to remove the doublet; clearing **Fit J** holds J at its start and disables the
range. **Phase** fits one phase per cluster on that axis and needs SP1, SP2 or No
apodization. Tab reaches family, J, Fit J (which enables the range), the two
bounds, × and Phase in that order, then the next axis; every control's name says which axis it changes. An invalid
entry (a J outside its range, an empty range, a phase on a family without one) is
explained under its axis, outlines the fields concerned and disables **Fit**, with
the same reason in Fit's tooltip, until it is corrected; nothing invalid reaches
the settings. Below the table: **Clustering**, with the statistical noise,
**Estimated** from the fitted planes or **Manual** (choosing Manual starts from the
estimate as shown, and the estimate stays under the value), and the threshold, a
**Factor** of the noise with the absolute level it gives, or an absolute **Level**
with the factor it amounts to (choosing Level starts from the level the factor
gave). The threshold selects the fitting regions, not only how peaks are grouped:
the points above it, in regions around the peaks, are the points fitted, and peaks
that share a region form one cluster. Each value reads on one line with its
source after it as a ▾ menu (“1.192 × 10⁴ estimated ▾”, “5.00 × noise ▾”), and
the derived value on the line under it. **Fit**, with optimizer passes (default **two**, with the neighbor
correction update between passes; an imported step schedule replaces the field)
and the planes **excluded**, by the plane numbers navigation shows (“1, 41–42”;
the configuration's `exclude_planes` counts from 0), with how many are fitted and
the numbering (“128 of 131 fitted · numbered 1–131”);
**Constraints** (collapsed, with a one-line summary): a position window per axis,
how far a peak may move from its start in ppm, and rules on a parameter kind
(linewidth or η) on one axis or all, each setting a start, a range or **Hold**,
applied in order so a later rule overrides an earlier one. A rule waits, with a
hint, until it sets something. An imported axis-wide rule on another parameter
(`*.F2.cs`, `*.*.phase`, `*`) is shown as written, with its start, range and Hold
editable; an imported rule for named peaks is shown as written and must be removed
before a desktop fit. A peak's own constraints are set on its Peak page
(**Parameters**); the section counts the peaks that have them (“2 peaks also have
constraints of their own”), and so does its summary. Then **Optimizer limits** (collapsed:
solver, iteration limit, tolerance). Values use × 10ⁿ, never e notation, and
accept plain or scientific entry (1.5e5, 1.5 × 10⁵); the tolerance reads
1 × 10⁻⁸. A typed value that cannot be fitted (a noise or level that is not
positive, an exclusion that leaves no plane or names a plane that does not exist,
a rule whose start lies outside its range or whose range is empty, a rule that
matches no parameter of the current model) is explained under its setting,
outlines the field, keeps the committed value and disables **Fit** with the same
reason, until it is corrected. The reason starts “Not applied:”, and nothing derived
from the setting in effect (the estimate under a manual noise, the factor or level
a threshold amounts to, the planes fitted) shows beside an entry that is not applied. Once a run exists, every setting whose working value differs from
the one the run used is marked • (an axis on its name, a setting on its label, with
what the run used in the tooltip), and the page says “• differs from run 3”.

Every committed settings change is one undoable edit in the same history as peak
edits (**Undo settings change**), and marks an existing run Outdated; Undo restores
the setting but never makes the run Current again.

**Analysis ▸ Exclude plane N from the fit** (also on the plane strip's context
menu) excludes the plane shown, or includes it again, as one such edit; the last
fitted plane cannot be excluded. An excluded plane stays in plane navigation,
which marks it **not fitted**, and stays measured: Overlay and Residual show its
experimental contours, with “No model on this plane” (or “No residual”) in the
header strip above the plot, never over the contours, and the spectrum and slices
keep their place; the peak
profiles and the Profiles view draw it as a faint dotted line with “plane N
measured, not fitted”, never as a point. The Run page lists the settings each run
was submitted with: its noise (a manual value as entered, an estimate as
computed), threshold and, when it had any, its excluded planes and constraints,
each rule on its own line. Before the first fit the toolbar
says **Not fitted** and Run is unavailable.

After every committed change to peaks or settings the Model page prepares the exact
fitting problem off Qt, with the same preparation that Fit submits, and reports
cluster and sample counts. Preparation never depends on selection, viewport,
search or display contours. An obsolete preparation is hidden at once. When
preparation fails, **Fit** is disabled and the Model page and its tooltip give the
reason. The member rings are the default sign of a selected peak's cluster.
**Show sampled fitting regions** (off by default) adds a faint tint behind the
contours, one cell per sampled point, with the selected peak's cluster slightly
stronger. The legend names the source: "recorded in run …" while the active run
is Current, otherwise "preview for current settings". The tint never changes the
viewport, comparison mode or model scope.

Choose **Fit** in the toolbar or **Fit dataset** in the Analysis menu
(Cmd/Ctrl+Return). Fit commits a valid draft and submits the prepared problem;
it always covers every working peak and original plane. There is no confirmation
dialog: the first Fit of a session whose Model page has not been displayed opens it
instead of running, and the next Fit runs. The settings summary beside **Fit**
(lineshapes · clustering threshold · passes; "Fit settings" in windows narrower
than 1200 px, with the summary in its tooltip) and **Analysis › Model settings** open
the Model page. Numerical execution uses the existing disposable process outside Qt. The run line beside the
analysis name shows pass/cluster/stage and elapsed time, with **Cancel** beside **Fit**;
the settings summary steps aside while a fit runs. Navigation, selection,
preview and pinning remain available; scientific edits and input replacement pause.
Cancellation, process failure and validation failure retain the previous result
and working coordinates. No partial result is adopted.

A failed fit says whether anything was kept: **Failed · no result yet** when there
is no active run, **Failed · previous result retained** otherwise; a canceled fit
reads **Canceled · …** the same way. A failure leads the run line in the error
style with ▲ (“▲ Failed · previous result retained · Run …”), and the Run page
comes forward with a **Last fit attempt** section above the run:
status, the stage that failed (Preparation, Fitting or Completion), the reason and,
when the application logs to a file, the log path. **Copy details** puts the
status, stage, message, dataset, active run, PeakFit, Python and Qt versions,
platform and log path on the clipboard for a report. Every failure's traceback
goes to the terminal that started PeakFit and to the per-user log file
`~/Library/Logs/PeakFit/peakfit.log` (macOS), `%LOCALAPPDATA%\PeakFit\Logs\peakfit.log`
(Windows) or `$XDG_STATE_HOME/peakfit/peakfit.log`, by default
`~/.local/state/peakfit/peakfit.log` (Linux), so a failure can be reported
without a terminal. The section disappears when the next fit starts.

When two peaks' fitted shapes are near-collinear (1 − |cos| ≤ 10⁻⁴ over the
fitting samples), their fitted positions are not adopted: their markers stay at
the working positions, a small □ in the model colour marks each recorded fitted
position, and the peak page
says “Fitted position not adopted: near-collinear fitted shapes with …”. The
recorded fit is kept as it is, and the amplitude profile withholds their
individual amplitudes as not established (below). The next fit starts them from the configured defaults;
moving either peak makes it an ordinary edited peak.

Runs are numbered in the order they completed, **run 1**, **run 2**, …, from the
analysis journal, so a number stays the same after save, reopen and recovery;
failed and canceled attempts get no number. Every caption that names a run uses
its number: the toolbar (“Run 4 · Current · …”), the focus bar, the Profiles title
and caption, historical comparisons, the Model page and the progress of a save.
The Run page's Run and Baseline rows add the short ID, and tooltips give the full
ID. Recorded continuation reasons (“moved since run …”) keep the short ID they were
recorded with.

After a refit, the Run page summarizes **Changes from run N ▾**: what differs
between the active run and a baseline run, read from both recorded runs without
refitting or judging either. The baseline is the run the active one was refitted
from, its parent (for a run with no recorded parent, the newest other run),
unless you choose another in the heading's menu, which lists
every earlier completed run newest first with its time, “parent” and, for a pruned
run, “summary only”. The first row, **Baseline**, names it. Choosing a baseline
changes only this section: the active run, working peaks and settings, Current or
Outdated, Undo and Modified stay as they were, and nothing is saved with the
choice. A new run, or reopening the analysis, compares with the parent again. Use
**File → Analysis history…** to inspect a historical run in detail. Peaks are
matched by stable ID, so renamed peaks still compare.

- **Settings** first: each setting that differs, as “baseline → active”, each
  side read from that run's own recorded settings (the same comparison as the
  Model page's • marks): the per-axis lineshape model, statistical noise,
  threshold, passes, excluded planes, constraints, iteration limit and tolerance.
  Identical settings read “Settings: identical”.

- **Clusters**: each cluster whose numerical warnings or highlights changed,
  that contains a peak moved, added or given other constraints of its own by hand
  in a refit after the baseline and not set back to where the baseline left it
  (“Edited: …”, from the edits each intermediate run recorded, so an edit undone
  before fitting, or a position changed only by fitting, is never listed), or whose
  members were grouped differently. Clusters with newly reported warnings lead and
  are drawn in the warning style, then those whose warnings are no longer
  reported, then edited ones, then highlights and regroupings. Warnings include the
  decomposition marks of the amplitude profile, as in the cluster list:
  **▲ Near-collinear shapes**, **▲ Ill-conditioned shapes** and **▲ Amplitudes not
  identifiable**. With the same
  members, a mark is “now reported” or “no longer reported”. A regrouped cluster
  names the clusters it came from and lists its marks “Now” and “Before
  regrouping”; a mark that disappeared with regrouping is not presented as resolved.
  A cluster made only of peaks added since the baseline is listed as a new
  cluster with its members and current marks, never as a regrouping, and without
  before → after values.
- **Measures** (χ²ᵣ, |r|max, |ρ|) before → after appear only when both runs define
  them identically: the same members, the same fitted planes (plane values and
  exclusions), the same lineshape model on every axis, the same constraints, the
  same σ_stat and noise model, the same fitting samples and the same number of
  varied parameters. Otherwise the line names the first difference in that order, for example
  “Measures not compared: fitting samples changed (67 → 68 per plane)” or
  “Measures not compared: lineshape model changed”, and shows no number. A lower
  χ²ᵣ is never presented as a better fit.
- **Whole run**, as rows: **Clusters** before → after with regroupings and peaks
  added or removed, and **With ▲ warnings** (clusters with engine warnings); a count
  that did not change reads “121, unchanged”. No run-level χ²ᵣ is shown: a
  median over clusters whose fitting samples differ is not a comparison.
- **Largest position differences:** the five largest, as Δ in ppm and in fitted
  linewidths on the axis that moved most. They are neutral information, not a
  measure of fit quality (said in the heading's tooltip); there is no threshold.
  Differences that round to zero at display precision are not listed; when all
  do, one row reads “Positions: unchanged at display precision”.

Each name opens its cluster or peak; at most six clusters are listed with a count of
the rest. A stored baseline is read off Qt. A Summary-only baseline keeps its
recorded settings, clusters and peaks, so Settings and the **Clusters** count are
compared; one line names what is compared only against a run kept with full
evidence, which has the members, fitting samples and usability they need (cluster
changes, ▲ warnings, measures and position differences), and no value is
reconstructed for them. The first run has nothing to compare.

**Signal not explained on this plane**, also on the Run page (what it is, behind its
ⓘ), lists on request
(**Find on plane N**) the connected regions where the current plane's
whole-spectrum residual exceeds ±5 σ_stat; Display sets the threshold for the
session. Positive and negative excursions are separate regions, and samples touching
diagonally belong to one. Each row gives the location of the region's extreme
sample, that extreme in σ, the region's size in samples and the distance in native
samples to the nearest working peak, largest |extreme| first (eight rows, then a
count of the rest). Choosing a row frames the region with surrounding context and
places the slices through its extreme, keeping the plane, the comparison mode and the
selection. A list belongs to one plane, run and threshold: another plane needs
another request. For an Outdated run the list is that run's historical residual,
titled “· Run 3 · historical”; distances are to the current working peaks
and follow each edit, and a nearest peak added since the run is marked ☆. The list is descriptive,
captioned “Diagnostic only · residual beyond ±5 σ_stat”; the tooltip adds that it never
adds or proposes peaks and never flags clusters, and that a signal may be left
unmodelled on purpose.

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
Experimental, Overlay and Residual are one choice: Tab reaches the group once and
the arrow keys move between them.
**Residual** shows `(native experiment − total usable model) / σ_stat`, including
model tails outside fitting masks. Default signed contours are ±3, ±5, ±8, ±13 and
±21 σ_stat; Display can scale these levels by changing the first residual contour.
Absence of contours means below the visible threshold.

In Experimental and Overlay, each slice has its own intensity range: from
min(0, lowest) to max(0, highest) of its visible samples over every plane, padded
by 5 % of the span, so zero is always included and positive and negative signals
keep their own extents. The range follows what the slice shows (a new row or
column under the crosshair, zoom, pan, a framed region or cluster) and stays fixed
while you change plane, so amplitudes compare from plane to plane. The slices have
no intensity axis. Overlay draws the total model at the
same native samples (dashed). Optional **residual lanes** under each slice show
`(D − M) / σ_stat` on one shared ±N σ_stat scale (±8 by default), set separately
from the intensity scale. In **Residual** the slices themselves show the residual
on that scale, in the residual's colour. Once a run exists the lanes keep their
place in every comparison mode (empty in Experimental, which draws no fitted
contribution), so switching Experimental, Overlay and Residual never moves or
resizes the 2D plot; before the first fit no mode shows lanes. The wheel
over a slice magnifies both slices (a **gain**, shown as “Slices ×1.25”), which
persists through hover, planes and zoom until a double-click on the slice,
**Default slice intensities**, a new dataset or focusing another cluster; the wheel
over a lane sets the lane scale. A caption above the row slice appears only when
there is something to say: the residual or lane scale, the gain, and counts of
clipped samples in words (“3 data samples clipped · 1 model sample clipped”). The corner readout gives the
values at the sample, intensity and model in one power of ten with four significant
digits for the larger (the status bar
repeats them only when a collapsed column slice leaves no room for the readout).
Slices take a share of the spectral view (by default 22 % of its height for the row
slice and 18 % of its width for the column slice; 30 % and 26 % in cluster focus),
so they grow with the window ([measured sizes](validation/gui-audit-followup.md)). Drag the divider between a slice and the plot to
resize it (below 48 px it collapses; drag it back to reopen); the new size is kept
as a share, so it follows window resizes and panel toggles. Double-click a divider
for the default of the overview or of focus, whichever you are in. **View → Slices**
offers all of these from the menu and keyboard.

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
amber label (▲ N) that never elides. When the run's noise-power check found
processing the header does not record, the line shows **▲ Noise check (F2)**, naming
the axes, before the counts; the tooltip, and the ⓘ of the Run page's Noise check
row, reachable without a pointer, give what was found and what to do
([correlated-noise.md §2.6](correlated-noise.md#26-the-noise-power-check)). It is a
warning about the errors only, never about the fit.

The first completed run opens the navigator's **Clusters** tab: one row per
recorded cluster with its members and three measures: χ²ᵣ, the largest |recorded
residual| in σ (|r|max) and the largest |amplitude correlation| between members
("–" for a single peak, "n/a" when the covariance is unavailable). The first
column shows one mark, its tooltip naming every item. An amber ▲ is a numerical
warning: unusable, not converged, uncertainty unreliable, a varied parameter at a
recorded bound, or an amplitude split the fitted shapes do not determine
(**Amplitudes not identifiable**, **Ill-conditioned shapes** or **Near-collinear
shapes**, naming the members, whom the amplitude profile also withholds). Otherwise a subdued ↔ marks profile errors correlated across
planes (the correlated-profile flag, a limit of independent error bars, not a
warning), and else a subdued ◇ a highlight. Profiles tiles show the same mark.
Every one of these is also on the cluster and peak pages; the list lets you find
them without opening each cluster. Highlights are descriptive
rules, never warnings: amplitude correlation |ρ| ≥ 0.9, and a fitted linewidth
outside 0.5–2× the run's median for that axis over usable peaks. Each highlight
states its rule and value, for example "LW 0.43 × median (¹H)". Movement from the
initial position is not highlighted: reaching the right position from a rough
start is what the optimizer is for; initial and fitted positions remain on each
peak page. **Display** sets the thresholds for this session; they change no results,
warnings or run state. The menu above the list shows all clusters, any warning,
any highlight, one warning or highlight type, or, when any cluster carries it,
**↔ Profile errors correlated**, each with its count. Columns sort
on click; the default is χ²ᵣ, highest first. Choosing a
row, or **N**, opens the inspector's **Cluster** page and selects one of its
members (keeping the selected peak when it belongs to the cluster), so the cluster
is the unit of review while a peak stays the unit of editing. Under its header, a
compact amber line names the cluster's numerical warnings (“▲ Uncertainty
unreliable · see Review”); their explanations stay in Review. The cluster page
lists the members one line each: its number, a link to its peak page and its
recorded values at display precision under each axis's name and each quantity's
unit (± σ on hover, recorded bounds in amber), ↔ after a member whose profile is flagged; then the amplitude profile of all members; the recorded
amplitude correlation, “conditional on lineshapes” (a single ρ row for two
members, a matrix in member numbers shaded by |ρ| for more, why it is
unavailable when the covariance was not recorded, and no section for a single
peak); the cluster's review rows, ending with **Profile errors** for runs with
two planes or more: “↔ correlated across planes” with the flagged members, “within
tolerance” or “not assessed”, the diagnostics and their meaning on hover; and
provenance.
**Enter** or a double-click on a cluster row, **Focus cluster** on the cluster page,
or **View → Focus cluster** (Cmd/Ctrl+Shift+F) puts the selected peak's recorded
cluster in **focus**: the same spectral view, reframed in place on the displayed
member positions ± max(3 × fitted FWHM, 8 native samples), at most a quarter of the
spectral width and within the spectrum, with residual lanes on. The frame does not
depend on the optimizer's fitting samples; **Fitting region** in the focus bar tints
them, together with the Model page's toggle. The bar names the cluster (its run's number,
currentness and full ID in the title's tooltip) and says what differs from the
toolbar (“Outdated · model and residual are historical”), and every mode,
mark, legend, pin and display setting works as in the overview, since it is the
same view. **‹ Spectrum** or **Escape** (from the spectrum, the cluster list or the
inspector) returns to the overview exactly as it was: viewport, slice sizes and
scales, mode, fitting-region display and sample. Leaving focus never reverts an
edit or a fit. In an inspector field with a draft, or during a drag or placement,
the first Escape cancels that and the next one leaves focus. **Full spectrum**
(Cmd/Ctrl+0) reframes the cluster while in focus. Focus keeps its own slice sizes
from one focus to the next; Save records the overview it returns to and both sets
of slice shares. An analysis saved before slice sizes became shares reopens with
the default shares.

Corrections happen in focus with the overview's edits and the same Undo/Redo:
dragging the selected marker shows a ghost at the proposed position while the
committed marker and the accepted comparison stay as they were; releasing commits
one undoable move in unfolded ppm, so a folded peak keeps its aliased display.
**A** (or the add tool) places a peak at the next click; Enter on its name in the
inspector returns the keyboard to the spectrum, so **A** places the next one at once
(coordinate fields keep focus). Each commit makes the run Outdated. A peak the run
has not fitted, such as one just added, keeps focus on its cluster while it lies in
the cluster's frame; otherwise focus moves to the cluster with the nearest member.
**Fit** refits the whole dataset in focus, and focus follows the selected peak by
its stable ID, keeping the zoom when the members are unchanged. When the members
changed, a line under the bar says how: “Regrouped by run … · was … · joined by
…”, peaks not in the run, and links to the clusters former members are now fitted
in; a peak added since says which peaks it is now fitted with, with a link back to
the cluster that was on show. A canceled or failed fit keeps every working edit and
focus.
On a peak page, **Review cluster ›** sits beside the peak's name in the page
header, under the cluster size (“PEAK · CLUSTER OF 3”), and opens its cluster page; selecting a peak
on the spectrum or in the Peaks list opens its peak page. Whenever the selected peak belongs to the active run, the other members
of its recorded cluster carry a thin accent ring on the spectrum, while the
selected peak keeps its stronger ring and label; this never changes the viewport,
comparison mode or the total-model scope. **N** (Shift+N backward) in the cluster
list moves in list order to the next cluster the **Show** choice lists; with all
clusters shown it moves to the next one with a numerical warning, and with none
the status line says to choose a mark in Show. **Select next cluster with a
warning** on the Run page always moves to a warned cluster. Neither ever
selects a hidden row. The
inspector's **Cluster review** keeps three families apart: numerical warnings
(with the engine's own text, or the members whose amplitude split is undetermined),
highlights (with their rule), and measures, as label
rows: **Warnings** and **Highlights** (“none” when empty), **χ²ᵣ** (its sample count
in the tooltip), **Max residual** with its guide √(2 ln N), the typical largest of
N independent Gaussian values (a rough guide, not a threshold: correlated noise
makes neighbouring samples dependent), and **ρ** with the member the selected peak
correlates with most, captioned “conditional on lineshapes”. Near-collinear pairs
the peak belongs to appear as **Collinear**. The cluster page uses the same rows.
Every inspector line starts on one left edge; forms, tables and review rows share
one label column, and disclosures (▸ About …, Provenance) are quiet lines at the
end of their section. The cluster list's footer appears only for a filter
or numerical warnings (“13 of 121 clusters · N: next”), and its run line only when the run is Outdated: the
toolbar names the run. A peak page before any fit has no status line; the
positions' “Unfolded ppm” qualifier is a tooltip unless the spectrum shows the
peak at a displayed alias, which is then given beside it. Warnings and highlights that apply to the cluster stay in
view; **About these measures** opens the general explanation in place, and stays
open across selections. Recorded optimizer statistics refer to corrected fitting
samples and are not recomputed from the displayed whole-spectrum residual.
**Provenance** discloses the optimizer, termination message, correction revision
and run identity. **Fitted** is a table of the peak's recorded parameters, one line
each: positions then linewidths, the horizontal axis first, with unit, value and σ
(“fixed” for a fixed parameter); parameters at a recorded bound are marked. Under the
table, each fitted J has a caption: “J ¹⁵N F2 below LW: determined together” when J is
below that axis's linewidth, then its recorded ρ(J, LW) and ρ(J, phase) where the run
has them; the tooltip explains the qualification. Its last
line, **Amplitude**, follows the current plane: the recorded amplitude ± its
marginal σ, the uncertainty of that absolute amplitude including the lineshape
uncertainty (`intensity_err_marginal`), as the caption under it says, starting with
the plane. Its power of ten is in its name, as a unit would be: Amplitude (× 10⁹).
**Parameters** (collapsed) holds the next fit's starts and bounds and the peak's own
constraints. Each parameter, positions then linewidths, the horizontal axis first,
has a line with its name, **Hold** and ×, then its values: a position reads “at
8.256 ± [window] ppm”, how far it may move from where the peak is (moving the peak is
the position editor's); any other parameter reads “start [ ] range [ ] – [ ]” in
its unit. An empty field shows, greyed, what the Model settings give this peak in the
current preparation; a value typed there is the peak's own and applies to this peak
only, as `[parameters.peaks."<name>"]` does for `peakfit fit`. **Hold** shows whether
the parameter is held; changing it holds this peak's parameter, or fits it against a
rule that holds it. × resets the parameter to the Model settings. Values accept
plain or scientific entry, like the Model page's. The title counts the parameters
set (“Parameters · 2 set for this peak”). Own constraints belong to the working
peak, not to its assignment: they follow it through renames and moves, and two
peaks of the same assignment have their own; deleting the peak deletes them, and
Undo restores both. Each committed value is one undoable edit (**Undo constraints
of G45N-HN**) that marks the run Outdated; once a run exists, a parameter whose own
constraint differs from the one the run used is marked • on its name, with what the
run used in the tooltip, and so is the section's title. A peak whose own constraints
changed starts the next fit from the configured defaults (“constraints changed
since run 5d7b4768”), as a moved peak does. A typed value that cannot be used (not a
number, an empty range, a start outside the range typed beside it, a window that is
not positive) is explained under its parameter, outlines its fields, keeps the
committed value and disables **Fit** with the same reason until it is corrected or
the peak is left. A committed constraint that cannot be fitted as given (a start
outside the range the settings and the peak's own bounds give together, an empty
range, a parameter the current lineshape lacks) is explained under its parameter
and disables **Fit** with the peak's name; Fit then selects that peak with its
Parameters open. `peakfit fit` would move such a start to its bound without saying
so. While preparation runs or fails the section shows no values rather than old
ones.

**Profiles** (the toolbar's Profiles toggle, or Cmd/Ctrl+2; toggle off or Cmd/Ctrl+1 returns) shows
the active run's profiles in the centre, one tile per cluster, in the cluster list's
order and filter: sort or filter the list and the tiles follow. Tiles adapt to the
width: as many columns as keep each tile at least 210 px wide (logical pixels), tiles
at most 320 px wide, height following width; a wider window or panel adds columns
rather than enlarging tiles, and the selected tile stays selected and in view as
the columns reflow ([validation across window sizes](validation/gui-audit-followup.md)). Each tile overlays
every member's recorded signed amplitudes on one vertical scale that includes zero,
so a dip shared between correlated members, opposite signs or a large/small split
show at a glance; members keep the cluster page's colours and order, named under the
plot, a flagged member followed by ↔ (the correlated-profile flag; the tooltip names
the flagged members, or that the correlation was not assessed). The header gives the cluster's mark, first member and size, and its χ²ᵣ,
|r|max and |ρ|; in a narrow tile the measures are shortened, never the name. A member whose amplitude is not identifiable or not established is
not drawn, and the tile says so. The plot frames the same range as the inspector's
profile (a distant reference plane is left out) and a dashed line marks the current
plane. Opening Profiles marks the selected peak's tile and keeps the selection;
with nothing selected, no tile is chosen. Clicking a tile or moving to one with the
arrow keys selects its cluster, as a cluster row does; double-click or
Enter opens it in focus on the spectrum. Comparison and canvas tools step aside
while Profiles shows; the plane strip, selection and inspector stay, and returning
to the spectrum keeps the selection. Tiles show
no error bars: the inspector's profile carries them. The quantity chip over the tiles
is the inspector's: raw amplitudes or ratios to the chosen reference planes, one
choice for the session, made in either place. In ratios a dotted line marks 1; a
member whose ratio is unavailable is left out and named under its tile, and a tile
with no member to normalize keeps raw amplitudes and says why, as the inspector's
profile does. **Export PDF…** writes the tiles shown, in their order, to A4
landscape pages (twelve per page, whatever the on-screen columns) titled with the dataset, run, quantity and filter;
it is a figure of recorded values, written only when complete. After an edit the caption
says the tiles are that run's historical values; before a fit Profiles is
unavailable.

The inspector's **Amplitude profile** shows the selected peak's signed fitted
amplitudes across planes, with the other members of its cluster lighter on the
same vertical scale. It follows the accepted selection only: an invalid draft that
blocks selection leaves the profile unchanged. The inspector header names the
peak, its cluster size, the source run and Current/Outdated state.

Points are the recorded amplitudes, unconnected, against the supplied plane values
in acquisition order; repeated values stay separate observations. Without plane
values, or when the current values differ from those the run recorded, the axis is
the one-based plane number. Clicking a point opens the plane of the nearest
point; points sharing that exact value offer a choice. Error bars are the **profile
σ** that **Export results…** writes as `intensity_err`, plane by plane, never
rescaled by χ²ᵣ ([outputs.md](outputs.md#unavailable-and-unreliable-uncertainties)).
It includes the uncertainty of the fitted lineshapes except a common scale, which
moves all of a peak's amplitudes by one factor and cancels in ratios and in any
downstream fit with a free scale. For an isolated peak it is the conditional σ, the
same on every plane. It differs from the **marginal σ** of the inspector's Fitted
table, the uncertainty of one absolute amplitude with that scale included: its
scale part is shared by every plane, so as independent bars it would overstate the
uncertainty of the profile. Strongly overlapping peaks can also have errors
correlated across planes, which no independent bar shows. Where that exceeds the
validated tolerance, the **correlated-profile flag**
([outputs.md](outputs.md#unavailable-and-unreliable-uncertainties), `correlated_profile`)
marks the member with ↔ in the legend. When the selected peak is flagged, one
caption under the plot names the flagged members, for example “Errors of 53N-H and
147N-H are correlated across planes; error bars alone do not show this
covariance.” It says nothing about the size of any σ: the marginal and profile σ
are unaffected. Its tooltip gives the diagnostic found (largest correlated mode
relative to c, or |ρ| with a partner); **About these error bars** adds that
downstream fits treating the per-plane errors (`intensity_err`) as independent can
misstate parameter uncertainty, and that the results archive records the
covariance. It is a caption, not a warning: a limit of independent per-plane
errors, not of the data, and it changes nothing.
Where the cluster's lineshape factors are withheld, the caption says the correlation
was not assessed; runs with one plane show no flag. Where the run models the
processed noise correlation the errors are sandwich errors; otherwise they assume
independent points. The note beside the plot says so in a few words (“±1 profile
σ · noise correlation modelled”), followed by “noise check ▲ F2”, naming the axes,
when the run's noise-power check found a mismatch, then any points off scale, by
plane, and whether the current plane is one of them, in the same note, which wraps
only between phrases; χ²ᵣ is with the cluster's measures; **About these
error bars** below it gives the noise model and the noise-power finding in full. The inspector's provenance states it once for the
run.

Each member's amplitude is classified from the run's recorded shapes, with the
factorization and numerical conventions that produced the amplitudes and no
threshold of its own. The classification is conditional on the fitted lineshapes; it does not establish identifiability when positions and
linewidths are allowed to vary:

- **Not identifiable**: the fitted shapes are linearly dependent and this member's
  amplitude lies partly in their null space, so the data cannot determine it. Other
  members of the same cluster can still be identifiable.
- **Reliability not established**: the amplitude is mathematically determined but
  depends on an ill-conditioned direction (where the joint covariance is withheld),
  or its shape is near-collinear with another member's.
- **Identifiable**: drawn with its error bar. When the core reports no finite σ,
  the amplitudes are drawn without one and labeled **σ not available**.

Members that are not identifiable or not established are not drawn and do not set
the axes. The profile says which class applies and why, notes that the total fitted
model remains available for comparison although the individual decomposition is
unreliable (or, for a null-space amplitude, not determined by the data), and lists
the recorded values under **Numerical
diagnostics**; exports are unchanged. No combined amplitude is shown. On the
bundled data, the collapsed pairs a 20 Hz linewidth start produces are not
established, never not identifiable: they keep full numerical rank
([validation](validation/amplitude-identifiability.md)). When at most two observations sit
beyond a gap wider than half the span, such as a distant reference plane, the
plot frames the remaining range and labels the off-scale points; **Show all
points** shows them. Outdated runs stay visible and labeled; peaks added after a
run and unusable clusters show why there is no profile.

The profile shows **Raw amplitudes** by default. Its quantity chip offers
**Choose reference planes…**: enter one or more plane numbers (for example `1` or
`1-3`); the dialog points out isolated planes but never chooses one. The profile
then shows each peak's ratio A / A_ref to the mean of its own amplitudes on those
planes, with ±1σ propagated to first order using the full gradient, so a plane in a
multi-plane reference is treated as correlated with the mean. The propagation uses
the peak's covariance between planes, rebuilt from the stored covariance factors
([profile-uncertainty §4](profile-uncertainty.md)): it includes the lineshape
uncertainty, which moves a peak's amplitudes together from plane to plane, so ratio
error bars are not the raw amplitude σ rescaled. The note reads “±1σ of the ratio,
lineshape uncertainty included”. A single reference plane is drawn exactly at 1
without an error bar; reference planes use diamonds.
Normalization is a display: recorded amplitudes, σ, exports and saved results are
unchanged, and the reference lasts for the session.

When a ratio cannot be shown, the profile keeps the raw amplitudes and says so
directly under the plot (“▲ Showing raw amplitudes …”); the quantity chip reads
“÷ plane 1 · not applied” and the axis names what is plotted. A ratio is
**undefined** when the reference mean is zero or not finite, when the covariance
between planes is unavailable (its lineshape terms are withheld), or when a
reference plane is not in the run. It is **withheld as unreliable**, although
computable, when σ_R/|R| exceeds 0.1, σ_R being the total σ of the reference mean. A member whose
individual amplitudes are withheld is never normalized: a ratio would cancel the
arbitrary scale of a degenerate split and look well determined. The
reference choice is kept, so the next well-determined peak is normalized again.

Drafts retain the accepted comparison. A committed manual scientific change marks
the run **Outdated**; Overlay and Residual stay selectable and keep showing that
run's model as **historical** evidence, named “Run 3 · historical” in the
legend, the pending label and the mode tooltips. On
the spectrum, ☆ marks a working peak the model does not contain (legend
“☆ not in this model”; ◇ is kept for review highlights), and × the
model's recorded position of a peak moved or deleted since, with a dotted line to
the peak. The cluster and peak pages still report that run's statistics as
“Recorded by an outdated run”. The historical comparison is withdrawn, with the
reason in the mode tooltips and the unexplained list, when the spectrum,
axes, plane selection or plane values differ from those the run recorded; the
overview then returns to Experimental. Undo alone cannot recertify a run.
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

The **Display** tool holds contours (signed first contour, level factor, levels per
sign, stroke weight), peak labels, residual levels (first residual contour and the
unexplained-signal threshold) and the review highlights. Every field uses a decimal
point whatever the system locale; the first contour reads in the shared × 10ⁿ form
and accepts plain or scientific entry (2500, 1.5e5). Contours are graded from light low levels to strong high levels;
measured data is blue (negative plum, dashed), the model amber (negative dotted)
and the residual vermilion/teal, as named in the in-plot legend. Peak markers
keep a plot-coloured halo so they stay visible over contours; the selected and
hovered peaks are labelled with their assignment only, and resting on a marker
shows its exact unfolded (and displayed alias) coordinates in a tooltip. The slice
scales follow the rules under Overlay above. The **Full spectrum** tool changes only
frequency ranges. Appearance follows System or explicit Light/Dark under View.

Keyboard access includes Cmd/Ctrl+O, Cmd/Ctrl+F, Cmd/Ctrl+Z/Shift+Z,
Cmd/Ctrl+0, Ctrl+Page Up/Down, native list arrows/Enter and canvas arrows/Space/A.
On the spectrum, E, O and R choose Experimental, Overlay and Residual, and [ and ]
change plane; in the cluster list N and Shift+N move to the next and previous
cluster with a numerical warning. Cmd/Ctrl+Return fits.
Cmd/Ctrl+Shift+F focuses the selected peak's cluster and Escape returns; under
**View → Slices**, Cmd/Ctrl+Alt+arrows resize the slices, Cmd/Ctrl+Alt+=/− scale
their intensities and Cmd/Ctrl+Shift+L shows or hides the residual lanes.
View also provides exact sample coordinates and frequency ranges. **Help ▸ Keyboard
shortcuts** lists every key: each menu command with its key, in the platform's
notation, then the single keys of the spectrum, lists, slices and fields.
Single-letter keys, Delete and Space in text fields retain their ordinary text
behavior.

The control that has keyboard focus carries a blue ring, separate from the
selected-row and checked-button fills; a list or tree shows it on the view's
edge. The ring appears when you move focus with Tab, Backtab or a shortcut, or
press a key, and stays hidden after a mouse click. When the spectrum has focus,
even after a click, a frame in the same colour surrounds the 2D plot: A, Space,
E/O/R and the arrow keys then act on the spectrum.

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

**File → Analysis history…** inspects retained runs in detail, named by run number
as on the Run page; its status line names the active run (“Active run 4 · Current
numerical evidence”), and failed or canceled attempts read as such, not as
summaries. The latest ten completed runs
retain full evidence; older runs have explicitly labeled Summary-only journal
entries. **Keep full evidence** protects a full run from pruning. Historical evidence
has a 1 GiB budget, excluding the active and protected runs; their costs appear in
the manifest and Save status. Only the active complete result stays in memory;
history loads when selected. Each accepted result has a durable temporary snapshot
before resident replacement. If that snapshot cannot be written (for example, the
disk is full), the run is still accepted: the run line shows **▲ Not in recovery
copy**, its tooltip gives the reason, and **Save analysis** keeps the run. Save and normal closure clean owned scratch storage.
After an interruption, **File → Recover interrupted session…** offers retained POSIX
session snapshots and requires Save as to a permanent project. Recovery is not a
saved analysis or an autosave of later edits. On other systems an explicit Open
analysis can read the recovery path shown in diagnostics. Source-dependent MCMC, full refinement controls, source
embedding and reopening legacy local-fit exports remain outside this workflow.

See [native workflow evidence, measurements and limitations](validation/gui-v2-persistence.md)
and [project format](outputs.md#linked-source-analysis-projects).
