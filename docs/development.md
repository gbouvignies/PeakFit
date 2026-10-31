# Development Guide

This guide records the current contributor workflow and coding conventions. Keep
it short; implementation details should live in code and tests.

## Commands

Use `uv` for project commands.

```bash
uv sync --all-extras
uv run ruff check src tests
uv run ruff format --check src tests
uv run ty check --error-on-warning
uv run lint-imports
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest -q
uv run prek run --all-files
uv build
```

Use `prek`, not `pre-commit`, for hooks.

## Testing And Scientific Fixtures

Run the complete suite headlessly with `QT_QPA_PLATFORM=offscreen` and
`MPLBACKEND=Agg`, as shown above. This is the supported environment for
automated tests that may import GUI or plotting dependencies.

Tests protect both numerical/scientific behavior and refactoring compatibility
baselines. Treat golden files, regression data, tolerances, and reference values
as evidence with an explicit provenance and intended claim. Do not silently
regenerate them, weaken their assertions, or change them without explaining and
reviewing the scientific or contract reason.

## Architecture Rules

- Keep numerical code independent from Typer, Rich, Qt, and filesystem side
  effects unless that boundary is intentionally changed.
- Prefer direct functions and explicit data flow over managers, service layers,
  registries, factories, or adapters.
- Excellent visual design, readability, intuitiveness and consistency are explicit
  product requirements. A dedicated GUI UX and visual polish milestone remains
  required after establishment of the core whole-dataset workflows; functional
  changes should support that direction.
- Keep `engine` focused on numerical/domain primitives.
- Keep `fit` focused on loading, validation, orchestration, and fit-output
  assembly.
- Keep `auto_pick` isolated as an experimental workflow.
- Keep one shared scientific authority: `scientific.npz`, CSV tables under
  `tables/`, and a short report; JSON is explicit interoperability.

The import-linter contracts in `pyproject.toml` enforce the current package
layers.

## CLI And Terminal Output

PeakFit runs can be long. Terminal output should leave a useful scrollback:

- Show a compact input/config/output summary before work starts.
- Keep routine success terse.
- Make parse errors, failed clusters, boundary hits, and high reduced
  chi-squared easy to find.
- Prefer actionable messages over decorative panels.
- Do not expose internal architecture names in user-facing output.

Shared helpers under `src/peakfit/ui/` are useful only when they reduce
duplication and improve consistency. They may be merged or removed during
simplification.

## Typing

- Do not add `from __future__ import annotations`; Python 3.14 already defers
  annotations.
- Use modern syntax: `X | Y`, `list[str]`, `dict[str, float]`.
- For NumPy-style public inputs, prefer `numpy.typing.ArrayLike` and normalize
  immediately with `np.asarray(...)`.
- Return `numpy.typing.NDArray[...]` for vectorized numerical functions.
- Use overloads only when the public API guarantees scalar-in/scalar-out
  behavior.
- Enforce shape requirements at runtime; do not encode fragile shape types.

## Documentation

Documentation should be consolidated with code changes. Prefer one accurate page
over several overlapping pages. Delete stale notes when they stop matching the
current design.


## GUI responsiveness

PR #53 was tested natively on macOS with the bundled `pseudo3d.ft2` dataset
(131 × 256 × 546 samples, 166 peaks, 69.85 MiB input), ten contour levels per sign,
and thresholds in multiples of first-plane noise. These are representative local
measurements, not portable performance guarantees.

The prolonged freeze was reproduced while replacing a 664-row parameter table:
repeated ten-second stack dumps remained inside `QTableWidget.setItem()` for more
than 30 seconds. Clearing old rows, using interactive column sizing, and sizing
once after population reduced repeated native replacement to 20–21 ms. This fix
is retained. All-plane segmentation took about 7 ms, including under 1 ms for
boundary merging. Only the selected group is materialized; at 5× its arrays use
0.08 MiB versus the old all-cluster path's 12.1 MiB.

The current UX was compared directly with the original merged viewer at commit
`21e5cfc`, using the same dataset, threshold sequence, window size, and selected
spectral range. Both retain all 20 contour paths at every threshold. The original
inspection workspace has three plots; the revised Fit workspace has scope metadata.
Times below include event processing and frame capture, so visible side panels
account for part of the difference; they are not isolated renderer speedups.

| Threshold | Vertices | Original local pan / zoom | Revised local pan / zoom | Original / revised overview capture |
| --- | ---: | ---: | ---: | ---: |
| 5× | 29,206 | 62 / 94 ms | 23 / 21 ms | 29 / 15 ms |
| 1× | 229,856 | 59 / 58 ms | 28 / 35 ms | 29 / 22 ms |
| 0.1× | 1,097,210 | 86 / 82 ms | 43 / 42 ms | 53 / 43 ms |

The sweep 5 → 2 → 1 → 0.5 → 0.2 → 0.1 → 5× verified full contour visibility,
zoom/pan, recovery to ordinary scope, and unchanged plot/control geometry before,
during, and after threshold updates. At 0.1× the revised update settled in about
0.51 s, versus 0.30 s originally; this includes the Fit workspace's 150 ms debounce
and scope update. The previous contour set remains visible throughout. Dense
overviews still cost more to draw; there is no blanket contour-vertex cutoff.

One serial background lane handles preparation and numerical contour extraction,
with latest-pending requests and revision checks. Qt paths and widgets stay on the
GUI thread. The renderer uses full geometry, and completed sets replace the old
sets together. Fixed-height, width-independent status/metadata fields prevent
progress or warning text from reflowing the plot; full messages are available in
Details. Efficient parameter-table replacement and non-waiting shutdown are retained.

A large region does not suppress visualization. At 0.1×, the exact scope contains
139,776 spectral points and all 166 peaks. Its two grid-index arrays use about
2.13 MiB, versus 279 MiB for dense numerical data/correction arrays. Scope geometry,
all markers, and metadata remain available without allocating the latter. Numerical
arrays are limited to 128 MiB; that limit affects numerical preparation only. `PreparedLocalFit.cluster` and `.parameters` are absent
for a scope-only result, which must not be treated as ready numerical input.

The first all-plane noise estimate still has substantial temporary memory use, and
full low-threshold vector geometry requires more memory and painting time. No low
total-process memory bound or constant frame rate is promised. Regression scenarios
cover exact scientific geometry/data/identity, complete dense contours, stable
layout through progress and numerical limits, latest-work handling, and safe close.

### Prepared local execution (Step 2A)

Native Cocoa validation on 2026-10-28 used the same bundled experimental data on
macOS arm64/Python 3.14. The selected group contained three overlapping peaks,
75 sampled points per plane, and all 131 planes. It converged in eight function
and eight Jacobian evaluations, with reduced χ² = 1.36219. These are observations
from this machine and harness, not performance guarantees.

| Measurement | Observed |
| --- | ---: |
| VarPro plus terminal analytical evaluation | 24 ms |
| Spawn startup (including imports/handshake) | 831 ms |
| Input serialization plus owned copy | 0.95 ms |
| Serialized input payload | 183,481 bytes |
| End-to-end execution, transfer, and finalization | 862 ms |
| Child peak resident memory | 210.9 MiB |
| Parent high-water resident memory | 981.3 MiB |
| Cancel after child startup through process reaping | 20 ms |

The complete spectral data array is 69.85 MiB; only 78,600 bytes of selected sampled
data plus corrections, model geometry, parameters, and provenance cross into the
child. The parent high-water value includes loading, initial all-plane noise
estimation/preparation, rendering, and screen capture; it is not the incremental
cost of optimization. The existing noise-preparation peak remains substantial.
The child import footprint dominates this small fit. No speculative startup cache,
shared-memory transport, or persistent worker pool was added.

During execution the native harness switched Inspect/Fit, selected plane 25, moved
the crosshair, changed the displayed threshold from 5× to 6×, and requested updated
contours. The submitted threshold remained 60349.7535 while displayed contours used
72419.7042; working peak records stayed identical. Dataset replacement was rejected.
A 10 ms Qt timer continued throughout (251 callbacks; largest gap 178 ms, including
native screen capture/layout/navigation, so this is not an isolated optimizer stall
measurement). Cancellation and close with an active spawned child left zero fit
children. Canceling the native unsaved-result close dialog also kept the application
and completed result intact. Running/completed screenshots are in
`docs/images/local-fit-*.png`.

The two integration checks in `tests/test_gui_fit_execution.py` cover one-pass engine
agreement for overlapping multi-plane signals with duplicate assignments, unfolded
indirect coordinates, fixed linewidths, position bounds, signed amplitudes, and
unaltered inputs; they also exercise actual spawn/transport, provisional/unusable
outcomes, cancellation, child failure, stale delivery, retained-result replacement
guards, navigation/contour responsiveness, and shutdown. Existing GUI tests protect
manual editing and preview behavior. A full offscreen run passed 233 tests. One
earlier full run segfaulted inside native Qt/Python garbage collection; it did not
reproduce on the subsequent full run (clean main: 231 passed). The new lifecycle
test explicitly drains its preview worker and deletes its Qt objects on Qt's thread.

Cancellation terminates the private child; it never attempts to interrupt SciPy
inside a Qt task. A bridge thread owns spawn, serialization, receive, finalization,
and bounded terminate/join/kill cleanup. Application shutdown signals cancellation;
the bridge is non-daemon so interpreter exit cannot abandon a partly spawned child.
Qt never joins it synchronously. Failed/canceled attempts retain the earlier result;
completed numerical values are frozen using the same finalization seam as the CLI.
The 128 MiB preparation budget is not an optimizer/process memory bound: dense
Jacobian and finalization arrays add memory. No samples are dropped to fit a budget.

### PR #54 focused lifecycle review

The intermittent crash was investigated with a bounded comparison on macOS arm64,
CPython 3.14.5, PySide6 6.10.1, and NumPy 2.4.1. All runs enabled
`PYTHONFAULTHANDLER=1` and pytest's 120-second per-test timeout. Immutable source
archives of PR head `71536a4` and main `86a0528` used the same installed interpreter
and dependencies, with alternating native Cocoa full-suite runs:

| Scenario | Runs | Result |
| --- | ---: | --- |
| PR head, native Cocoa full suite | 3 | 233 passed each; 58–59 seconds |
| Main, native Cocoa full suite | 3 | 231 passed each; 56–58 seconds |
| Corrected PR, native Cocoa GUI/lifecycle tests | 3 | 10 passed each; about 6 seconds |
| Corrected PR, original offscreen full-suite invocation | 1 | 233 passed; 69 seconds |

No crash recurred. The saved original macOS report records `EXC_BAD_ACCESS` at
address `0x8` in `SignalManagerPrivate::qtMethodMetacall` during queued Qt dispatch.
It does not identify the receiving application object or establish a root cause.
This is not evidence that the crash was fixed or that it belongs to main, the fit
bridge, or the preview worker; no speculative lifetime refactor was made.

Teardown inspection confirmed that explicit close marks both lanes closed before
destroying the viewer. A temporary diagnostic also exercised the existing
`ViewerComputation` callback in two orders: completion after parent/worker QObject
deletion, and completion queued before deletion but dispatched afterward. Each
order passed five iterations, with invalidated C++ objects, completed callbacks,
and zero delivered UI completions. Temporary runners/logs stayed outside the repo;
no new testing infrastructure was introduced.

Native mouse/keyboard validation used the real bundled 131-plane spectrum: run a
fit, retain its result, make a working edit, start a replacement, switch Inspect/Fit,
and navigate to plane 25. Only the disposable numerical child was temporarily
SIGSTOP-paused to keep the modal sequence deterministic. Answering Yes to close
then Cancel at either the peak-discard or result-discard dialog preserved the same
running child, retained result, dirty peaks, and an unset cancellation flag.
Explicit Cancel fit reaped the child; accepting every close/discard decision on a
subsequent fit exited without remaining validation/fit processes. No numerical or
subprocess architecture changes were needed.

The existing close test now protects those transactional decisions, including a
fit completing during an earlier dialog. The existing spawned lifecycle test
checks direct `set_peaks()` rejection before any history/selection/revision changes.

### Step 2B-1: persistent inspection plots

Native Cocoa inspection used the bundled 131-plane spectrum and the overlapping
12N-H / 1002N-H / 1011N-H group (75 samples per plane at the default threshold).
The compact header, three experimental plots, inline details, and contour popup
were checked at 1360 × 845 and 1000 × 720 content sizes, including light/dark
appearance. Window-manager limits reduced the requested 900-pixel large height.
Qt widget captures are in the [GUI guide](gui.md); they show actual experimental
samples, with no model/residual rendering.

Recorded spectrum and plot positions/sizes were identical across Inspect,
preparation, starting/running, and completed states at a fixed window size.
Returning from numerical details restored all three plots. At the smaller size,
showing the retained-result action also preserved geometry. Native input checked
logarithmic threshold keys, precise numerical entry, group updates, plane 25,
crosshair movement, Inspect/Fit switching, and canceling result-replacement and
close confirmations. The popup itself was inspected through native Qt widget
captures: macOS accessibility/capture intermittently omitted this transient window.
No alternative contour algorithm was introduced or benchmarked in this layout
change; the existing full-resolution contour path and background lane are unchanged.

The existing GUI scenarios now protect plot visibility/geometry during preparation,
execution and results, popup focus/Escape dismissal, detail restoration, retained
thresholds, and numerical diagnostics. Native Cocoa: 10 passed; full offscreen
suite: 233 passed. The native scenarios include real subprocess cancellation,
stale delivery, unsaved-result guards, manual editing/history/export, and the
complete dense low-threshold contours. Ruff, formatting, strict ty and import-linter
pass. No new test functions or dependencies were added.

The final review refinement replaces the crowded `0781710` toolbar with one
fixed-width `Contours · 5.0× ▾` button in PyQtGraph's existing spectrum title band.
It is a normal Qt child of the plot viewport, positioned from the title geometry
on ViewBox resize; it does not cover spectral samples. This preserves native
accessibility (a QGraphicsProxyWidget trial did not expose the button on macOS).
Precise threshold entry now lives in the popup and receives focus immediately.
Reset remains in View and on Ctrl+0. Redundant fitting-state banners are removed;
instructions use tooltips, and contour progress shares the coordinate footer.
The visible button width stays fixed across threshold updates.

Native visual review checked the final control at normal and small sizes in both
appearances. It sits beside the centered plot title rather than competing with
the Fit header; the right-aligned popup stays within the spectrum column. The
threshold readout and all popup inputs were legible in dark mode. Actual native
input verified one-click numeric entry and group changes (5×: three peaks / 75
samples; 6×: two peaks / 48 samples), then restored 5× before execution. Background
fitting completed normally; plane 25, inline details, and View → Reset view remained
accessible. Spectrum/inspection geometry stayed fixed during popup visibility and
threshold updates. The existing scenario checks that the control stays outside
the data ViewBox and does not move at the 0.1× threshold.

The screenshots were refreshed, including normal settings-closed, settings-open,
and small light/dark views. macOS window capture omitted the transient popup, so
the settings-open images composite the actual Qt window and popup grabs at their
actual on-screen positions.
The final full offscreen suite passed all 233 tests; no test functions were added.

### Step 2B-2: completed-fit comparison

Native Cocoa validation on macOS arm64 used the bundled 131-plane spectrum and
12N-H / 1002N-H / 1011N-H fit: three overlapping peaks, 75 irregular samples per
plane, reduced χ² 1.362. Light/dark inspection covered Overlay, Experiment,
Residual, signed-amplitude profiles, crosshair movement inside/outside the mask,
plane 25 and rapid stepping through planes 25–31, threshold 5 down to 0.1 times
noise, the contour popup, and Inspect/Fit restoration. Spectrum/inspection-plot
geometry stayed fixed between comparison modes and during pending updates at
1360 × 845 and 1000 × 720. Images in the GUI guide are actual native Qt widget
captures; transient popup captures were inspected separately because macOS
screen capture/accessibility intermittently omits that window.

A temporary read-only timing harness recorded background request/operation times,
Qt path setup, viewport Paint-to-next-event-loop timing, retained cache payloads,
and process peak RSS. These are observations from this machine, not performance
assertions or a new benchmark suite:

| Operation | Observed time/payload |
| --- | --- |
| Default local grid and model contours (156 × 160) | 24 ms worker, 10 ms initial Qt setup |
| Native F3/F2 rows across all 131 planes | 3 ms worker, 1–4 ms Qt curve setup |
| Dense 0.1× experimental contours, rapid plane stepping | 103 ms extraction, 37 ms Qt setup |
| Latest dense contour request, queue plus extraction | 141 ms |
| Low-threshold model plane and paths (315 × 609) | approximately 64–113 ms worker |
| Cached model/curve Qt updates | approximately 0.3–3 ms |
| Ordinary viewport paint plus event-queue delay | approximately 5–28 ms; occasional 49–60 ms |
| Initial comparison caches (model + slices) | 0.23 + 1.60 MiB |
| After threshold/plane/crosshair exploration | 6.77 + 3.21 MiB |
| Process peak RSS after preparation and after exploration | approximately 974 MiB in both cases |

Native window capture/PNG encoding itself took 60–267 ms and also raised observed
Paint-to-next-loop timings; those capture events are excluded from the ordinary
painting range above. Peak RSS includes the loaded spectrum, preparation and Qt;
it is not an estimate of projection-only memory. Larger groups/plane counts will
cost more and have not been exhaustively benchmarked. Each comparison cache is
limited to eight entries and 16 MiB of numeric payload; active graphics/current
arrays and Qt path memory are additional. Native-row arrays scale with trace
length times plane count, never the whole spectral cube.

The existing ContourPy/retained Qt-path approach remains: measured extraction and
painting did not justify a new dependency or rendering rewrite. Experimental
requests have priority over pending model work; fresh-plane projection waits until
experimental contour replacement has been applied. Model grids cancel between
4096-point chunks; all-plane trace work cancels between 256-coordinate chunks.
Recent coordinates/planes are cached; changes to spacing/count reuse a suitable
retained grid. Revision checks reject late results. A plane change hides old
experiment/model pairs while the new pair is pending, without moving the viewport.

The two existing local-execution scenarios now cover projection equivalence,
signed amplitudes, repeated/unordered plane identities, aliases, irregular-mask
NaNs, clipping-warning refresh, mask visibility, comparison modes, rapid plane
changes/late delivery rejection, and outdated graphics. No new test functions,
optimization path, result persistence, or dependencies were introduced.

Verification: full offscreen suite 233 passed; final native Cocoa GUI scenarios
10 passed; final targeted offscreen GUI scenarios 10 passed and scientific
simulation scenarios 13 passed. Ruff, formatting, strict ty, import-linter and all
pre-commit hooks pass. Wheel/sdist build succeeds with the existing uv-build
version-range warning (installed 0.12.23 versus declared <0.12).


### Completed local-fit export validation (2026-10-28)

Step 2C used a small real two-peak/three-plane fit with signed amplitudes,
duplicate assignments, and experimental values `[2, 0, 2]`. Scientific round-trip
checks cover converged, usable non-converged and unusable classifications,
canonical JSON schema/ResultsLoader, parameter constraints/errors, amplitude
errors, original engine/GUI identities and point/plane ordering. Failure injection
leaves no published partial directory. GUI checks cover canceled exports,
write failures, independent result/peak saved states and unchanged working peaks.
The existing real-process lifecycle tests protect transactional close decisions.

Native macOS arm64 validation used Qt's Cocoa backend: 11 focused export/viewer
tests passed. An additional native Qt dialog harness exercised close Cancel,
directory-picker Cancel, successful threaded export with the graphical overlay
retained, replacement of a previously exported result without a result prompt,
and clean closing. The replacement test isolated file loading with the same
in-memory fixture. A native widget capture confirmed the export destination in
the existing footer while the comparison remained visible.

The external desktop UI tool could not attach to uv's unbundled Python process.
The OS directory picker also could not be driven by the Qt timer harness, so the
additional dialog checks used Qt's directory dialog on Cocoa
(`AA_DontUseNativeDialogs` in the disposable harness only). The application keeps
the normal platform picker. Interactive OS-picker automation remains unverified;
selection cancellation and export behavior are tested through the supported Qt
file-dialog seam. No validation harness or new test infrastructure is shipped.

Full offscreen suite: 238 passed. Ruff, formatting, ty, import-linter and package
build passed. The build reported the existing uv-build upper-bound warning;
both sdist and wheel were produced successfully.


### PR #57 review follow-up: publication and compact status

Finalization now uses a separate sibling lock and renames the validated staging
tree to an absent destination, avoiding Windows directory replacement. Focused
checks enforce Windows-style rejection of existing directory targets, verify
complete artifacts at publication, refuse late empty/populated destinations and
competing export locks, and check cleanup after write/rename failures. The
portable coordination and unrelated-writer race limits are documented in
[outputs](outputs.md#gui-local-fit-export); no native Windows runner was available.

The permanent footer now says only **Fit results exported**. A long-path check
asserts that the full path remains in the tooltip and confirmation without
changing widget minimum sizes, window size or spectrum/trace dimensions. All 11
export tests passed on native macOS Cocoa. A native 1000 × 720 widget capture with
a 30-component destination was visually inspected: the footer was compact and
the spectrum and inspection panels retained their space.

Follow-up verification: full offscreen suite 244 passed; Ruff, formatting, ty,
import-linter, all-file prek hooks and wheel/sdist build passed. The existing
uv-build version-bound warning remains. GitHub CI supplies Linux validation;
Windows compatibility is assessed by the documented rename contract and focused
Windows-style destination-refusal tests, not a native Windows run.

### Step 2D: applying fitted positions

The Fit workspace's **Apply fitted positions** confirms changed peaks and their
indirect/direct ppm values, then commits the group through the existing peak-edit
history. `gui.fit_positions` validates retained `FinalFitOutcome` chemical shifts
using the engine-to-working-ID mapping and retained spectral axis order. It rejects
incomplete/ambiguous mappings, invalid coordinates, unusable outcomes and changed
working inputs. Fixed coordinates come from the same final outcome. This path does
not optimize, evaluate the model, or rewrite retained scientific quantities.

`tests/test_gui_fit_positions.py` protects duplicate assignments, custom axis labels
in reversed lexical order, unfolded indirect shifts, fixed positions/no-op updates,
atomic multi-peak undo/redo, unaffected records, invalid-coordinate rejection,
current-input/running-fit guards, confirmation cancellation/revalidation, unchanged
selection/viewport, and independent result/peak saved states. Scientific JSON,
provenance JSON and amplitude CSV exports are byte-identical before and after Apply.
The conservative revision guard keeps the retained result outdated after Undo/Redo;
a new fit is required for a current comparison.

Native macOS Cocoa validation on 2026-10-28 used the bundled experimental
`pseudo3d.ft2`, `pseudo3d.list` and `b1_offsets.txt`. The real spawned fit selected
12N-H / 1002N-H / 1011N-H: three overlapping peaks, 75 samples per plane, all 131
planes, reduced χ² 1.362192. A disposable Qt harness clicked the actual Apply action
and native confirmation, checked one Undo/Redo for all three peaks, retained result
identity, modified working state, hidden stale comparison, unchanged selection and
spectrum viewport, then moved the crosshair and selected plane 31. Native widget
captures were inspected at 1360 × 845 and 1000 × 720; the confirmation is reproduced
in the [GUI guide](gui.md#apply-fitted-positions). A clipped button label found in
this check was corrected before final verification. Twenty focused position/history
tests also passed on Cocoa. These checks use Qt input on a real native backend;
they do not claim a separate human usability study.

Final verification: 262 tests passed with offscreen Qt/Agg; Ruff, formatting, ty,
import-linter, all-file prek hooks, and wheel/sdist build passed. The existing
uv-build version-bound warning remains. `graphify update .` refreshed the local
AST graph. No public API or file-format compatibility break was introduced.


### Whole-peak editor commits

The peak editor has no Apply/Revert controls. `PeakInspector` requests a commit on
Enter or when focus settles outside its fields and Delete peak control. One owned
zero-delay Qt timer checks the active window, application state, popups and modal
dialogs; null focus, text context menus and application deactivation preserve drafts.
The timer resolves the current focus rather than relying on the previous widget,
so returning through null focus cannot strand a draft in the workspace.
Tab/Shift+Tab inside the editor preserve the draft. `SpectrumViewer.commit_pending_fields()`
validates the whole draft and calls the existing `PeakEdits.update()` once. The
same path guards selection, closing/switching details, placement/dragging, fit
preparation/execution and working-peak export. Invalid fields remain visible with
an actionable error and focus for correction. Escape restores the complete record.
No generic transaction layer or additional scientific representation is involved.

Delete peak is the deliberate exception: it discards draft text without parsing or
validating it, then deletes the committed record with the existing history operation.
Native Tab preferences are preserved, including macOS skipping buttons unless
all-controls keyboard navigation is enabled. Focusing the button does not commit;
Space activates it, and Escape restores the fields. One Undo restores
the last committed record. No confirmation or intermediate draft commit is added.

Draft notifications and no-op commits leave scientific revisions, preparation,
comparison graphics and retained results intact. Actual commits invalidate current
preparation and comparison, label retained results outdated, and preserve their
original quantities and independent export state. Run reads the current preparation
after committing fields, so it cannot submit a scope invalidated by that commit.

`tests/test_gui_workspace.py` protects real key/focus input, Tab through partial
coordinates, Enter/selection/panel commits, atomic multi-field undo/redo, correction
and Escape, native text context menus and their dismissal, null focus and inactive
windows, atomic draft-discarding deletion by mouse/keyboard/callback, equivalent
numeric text preserving redo, invalid mouse selection/drag/panel actions, modal
export/close guards, no-op revision tracking, retained-result identity/export state
and stale-preparation refusal. The application-state regression simulates inactive
Qt state; the same suite runs with offscreen Qt and native macOS Cocoa for real
widget focus, key, mouse and popup behavior. Existing viewer tests cover
native text undo, plotting gestures, export errors and unsaved-work guards; local
fit and fitted-position tests continue protecting scientific results and revisions.
The intentional interaction change requires no file-format migration: use Enter or
leave the editor to commit, and Escape to revert.

Verification: the full offscreen Qt/Agg suite passed (376 tests). All 112 native
macOS Cocoa editor/viewer, fitted-position and result-export regressions passed,
including native Tab preferences and explicitly enabled all-controls navigation.
Ruff, formatting, type checking, import-linter, all-file prek hooks and wheel/sdist
build passed (the existing uv-build version-bound warning remains). Native 1280 × 800
captures of the editor and its validation error on the bundled experimental
spectrum were inspected for readable labels, complete error text and preserved
spectrum/slice context; these are included in the GUI guide. Native Qt required
display access outside the execution sandbox (the sandboxed attempt failed during
QScreen initialization).


## Whole-dataset desktop fitting and review

The GUI runs `run_pipeline_iter` in its existing disposable-process bridge, with a
reviewed two-pass default and threshold independent of display contours. Completed
review uses `CompletedFit`, retains original input/provenance snapshots, and evaluates
all usable model contributions without an amplitude solve or spatial cutoff. Shared
process/export helpers now use the scope-neutral `FitProcess` and `export_fit_result`
names. CLI defaults and canonical output schemas are unchanged.

Verification: full offscreen Qt/Agg suite **386 passed**; focused GUI pipeline/export
checks also passed after the final provenance adjustment. Ruff, formatting, type
checking, import-linter, commit hooks and wheel/sdist build passed. The existing
uv-build version-bound warning remains. `graphify update .` refreshed AST relationships.

The native Cocoa acceptance script drove the bundled 131-plane / 166-peak dataset
through settings review, actual two-pass execution/corrections, full-model overlay,
residuals, plane input, crosshair/selection, cancellation, protected close, edits and
Undo. All 121 clusters were numerically usable/converged; three uncertainty warnings
remained explicit. Scientific values, repeated plane identities and stale delivery
are protected by focused tests. See [native evidence and limitations](validation/gui-dataset-fit.md)
for before/after screenshots, exact native sizes and measured latency.

The unattended native GUI suite encountered focus-dependent failures; the first
Tab-focus failure was reproduced verbatim on unmodified main. It is not reported as
a passing native suite. The dedicated native acceptance run establishes Qt focus
explicitly for synthetic keyboard input and passed; normal application focus code
was not changed to accommodate the test environment.


## GUI v2 analysis persistence validation

`uv run python tools/validate_gui_persistence.py` fits the bundled 131-plane NMRPipe
dataset, saves, repeats Save, uses Save As, reopens verified numerical evidence,
inspects model/residual, edits positions, and performs a fresh real fit. It captures
native Qt light/dark and numerical-only views, JSON timings/payload sizes and
per-operation heartbeat gaps. Run on a desktop with Cocoa/Qt permissions; offscreen
captures are not native acceptance evidence. Results and limitations are recorded
in [validation](validation/gui-v2-persistence.md).

`tests/test_gui_project.py` protects source relocation/absence/mutation, same-size
same-mtime changes, checksum corruption, version refusal, interruption/disk-full,
immutable reuse, lossless numerical round trips, unsupported models, working
transition validation, protected/full/summary retention and trusted child-process
transport. Reopen tests prohibit preparation/optimization and compare reconstructed
models with retained analytical arrays. Existing editor tests protect the changed
analysis-loss guard and scientific Currentness; CLI outputs remain unchanged.


### PR #70: history lifetime, resident memory and compression

`uv run python tools/benchmark_gui_history.py --output /tmp/history.json --reopen`
runs twelve native process-based fits, samples current main-process RSS after
settling and GC, then checks normal-close cleanup and verified reopening. It also
records how many complete results are resident. Use a fresh process for each source
revision; the baseline was measured against the approved pre-review code at
`88d2fa2` via an isolated source copy. `--fits 1 --compression` measures gzip levels
independently so compression buffers do not contaminate repeated-fit RSS.
The saved reports, native captures and declared limitations are in the persistence
validation document. They show one active result and disk references for history;
RSS is variable and child-process peak memory is not claimed to be measured.

Tests cover the full ten-run save/reopen/eleventh-fit/demotion cycle and every
promised compact scientific field, byte-budget pruning, on-demand selection,
normal close while a preservation writer is in flight, non-coalesced cleanup,
recovery adoption/Save as and ownership transfer to numerical-only review.
Session cleanup is queued after writers and is never canceled by executor shutdown.
The internal gzip wrapper is lossless and bounded; typed-array scientific authority
is recommended for separate review rather than held back by compatibility concerns.
