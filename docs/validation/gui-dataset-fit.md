# Whole-dataset GUI acceptance

Native Cocoa validation on 2026-10-29 used the bundled 131×256×546 spectrum,
166 working peaks, 121 clusters, SP2 on both axes and two optimizer passes.
The actual pipeline completed one cross-cluster correction update; every cluster
was numerically usable and converged, with three analytical uncertainty warnings.
Convergence is not independent scientific validation.

Reproduce with a new output directory:

```bash
uv run python tools/validate_gui_dataset.py --output /tmp/peakfit-gui-review
```

The script uses native Qt mouse/keyboard events for settings, Run, plane entry,
comparison selection, spectral crosshair/peak selection, cancellation, close Cancel,
peak editing and Undo. It checks retained identity, outdated-result behavior and
displayed numerical values, and separately exercises the canonical export writer.
Normal Qt event loops release the GIL while workers run. The script explicitly
establishes Qt activation before testing the Undo shortcut: synthetic QTest input
does not foreground an unrelated native process on macOS. The application uses
normal window activation. Captures allow deferred Qt axis sizing to settle before
rendering. These captures are real Cocoa widget content, without
OS title bars; Retina captures have twice the logical pixel dimensions. Residual captures use
lossless WebP with pixel-for-pixel decoded equality checked against native pixels;
this keeps dense images under the repository file-size limit.

## Before and after

The original captures are the unmodified application from the initial assessment.
The new captures show an actual completed total-model comparison, with the lower
analysis panel collapsed. The primary action, smaller markers, lighter contour
strokes, major ticks, native toolbar and narrower slices change the working view,
not just its colors. Selecting a peak leaves the total-model scope intact.

| Native window | Before | Overlay after | Residual after |
| --- | --- | --- | --- |
| 1280×800, light | [Before](../images/dataset-before-1280-light.png) | [Overlay](../images/dataset-1280-light-overlay.png) | [Residual](../images/dataset-1280-light-residual.webp) |
| 1280×800, dark | [Before](../images/dataset-before-1280-dark.png) | [Overlay](../images/dataset-1280-dark-overlay.png) | [Residual](../images/dataset-1280-dark-residual.webp) |
| Near 1440×900, light | [Before](../images/dataset-before-1440-light.png) | [Overlay](../images/dataset-1440-light-overlay.png) | [Residual](../images/dataset-1440-light-residual.webp) |
| Near 1440×900, dark | [Before](../images/dataset-before-1440-dark.png) | [Overlay](../images/dataset-1440-dark-overlay.png) | [Residual](../images/dataset-1440-dark-residual.webp) |

The window manager limited the large captures to **1440×845** (the original
assessment reached 1440×849). No 900-pixel native height is claimed.

[Settings review](../images/dataset-settings.png) shows actual noise, threshold,
models and scope. [Advanced settings](../images/dataset-settings-advanced.png)
shows effective nonlinear starts, bounds, units and freedom. Parameter tables
remain fully scrollable; column sizing samples rows to avoid expensive sizing of
every native delegate. Detailed result inspection is explicitly opened, never a
side effect of background completion.

Critical visual assessment: the spectrum has more horizontal room and the toolbar
now expresses the workflow. Thin native model contours can still become dense in
the full-spectrum survey; zoom remains important for inspecting line shapes.
The full native residual raster deliberately exposes unmodeled signal and noise,
including outside fitting regions. No smoothing conceals that structure. The next
review slice should add cluster navigation and more focused numerical inspection.

## Responsiveness and correctness

[Raw measurements](gui-dataset-fit.json) describe this machine/workload, not a
general latency guarantee. Image capture/write overhead is excluded from heartbeat
sampling. Times include the real viewer's scheduling and contour installation.

| Operation | Measured |
| --- | --- |
| Numerical two-pass execution and completed capture | 4.021 s |
| Uncached plane input → calculated plane installed, five planes | 1.353–1.362 s |
| Experimental plane input handler | 1.46–1.59 ms |
| Cached plane navigation, including experimental contour debounce | 49.2 ms |
| Cancel running replacement and reap process | 31.4 ms |
| 20 ms heartbeat during model projection/navigation | p95 22.1 ms; maximum 29.4 ms |

The fitting/review and cancellation/review phases include constructing dialogs
and tables and had isolated heartbeat gaps of 319/331 ms (p95 22.2/33.0 ms).
They do not demonstrate a universal sub-30 ms GUI bound. Projection itself stayed
off Qt; experimental navigation remained responsive during uncached calculation.
Further settings-dialog latency work can follow without changing model science.

The acceptance script verifies the complete plane against the retained scientific
projector, checks both signed contributions and native slices, and compares full
display residual traces with `D − total model`. It exercises rapid plane requests
and rejects old calculated graphics while pending. The unit/integration suite also
checks multi-cluster pipeline parity, actual nonzero terminal corrections,
read-only transferred arrays/mappings, duplicate assignments, repeated/reordered
plane coordinates, child failure, canceled replacement guards and cancellation.
Existing independent Lorentzian and completed-fit tests protect failed contribution
omission and the distinction from recorded optimizer residuals.

No saved-run reader, cluster navigator, per-peak whole-dataset amplitude profile,
MCMC or refinement workflow was added. Canonical scientific export is retained;
the provenance companion is explicitly not a reopenable analysis archive.

## Verification limits

The full offscreen Qt/Agg regression suite and static checks are the automated
regression baseline. The dedicated native Cocoa acceptance workflow above passed.
The unattended native GUI suite also encountered focus-dependent failures: the
first Tab-focus assertion reproduced verbatim against unmodified main (10 tests
passed before the same assertion failed). This is not claimed as a fully passing
native regression suite. The acceptance harness explicitly establishes Qt focus
for keyboard shortcut tests without changing system preferences.

The final settled native capture run measured 4.021 s execution and
1.353–1.362 s uncached navigation. No kernel approximations, model truncation or
independent display amplitude solve was introduced.
