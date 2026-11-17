# Model settings: native workflow record and plane-exclusion equivalence (#216)

Visual, geometric and numerical record of W4: manual statistical noise, the
clustering threshold as a factor or an absolute level, plane exclusion and
axis-wide constraints on the Model page. The decision is in
[decisions.md](../decisions.md) (2026-11-17, “The remaining Model settings”).
Sizes are Qt logical pixels.

## How it was run

- **Code:** branch `dev/w4-model-settings-e3a3c8` at eda9dcc, after the review pass
  at 1280×800 (below), merged with `main` at 500fe88 (#227). PySide6/Qt 6.10.1,
  Fusion style.
- **Machine:** macOS 27.2.1 on Apple silicon, built-in Retina display, DPR 2.0.
- **Captures:** `uv run python tools/capture_model_settings.py --output <dir>`, and
  again with `--body-px 17 --sizes 1160x720 1280x800` for a large system font
  (native Cocoa, both appearances; 60 + 40 captures). The tool drives the real
  window on the bundled `02-advanced-fitting` dataset (131 planes, 166 peaks):
  1. the Model page as loaded;
  2. each setting set: Manual noise (starting from the estimate), Level (starting
     from the level the factor gave), planes 1, 41–42 excluded, a ¹⁵N position
     window of 0.5 ppm and a rule, linewidth on all axes in 5–80 Hz;
  3. the same, scrolled to the constraints;
  4. each setting not applied: noise 0, level −5, planes 1–131, a rule start of
     120 Hz outside 5–80 Hz, and an η rule (these axes are SP2: no η);
  5. the same, scrolled to the constraints;
  6. after Undo of every edit, plane 41 excluded from the plane strip's command;
  7. the GUI's own two-pass fit with plane 41 excluded, and its Run page;
  8. plane 41 in Overlay with 2N-H's profile;
  9. Model edits that differ from that run;
  10. Undo of the last of them.

  The tool fails on sideways scrolling, a clipped label or a field narrower than
  its text.
- **Review:** I looked at every capture of every run. The first runs found the
  manual noise clipped, sideways scrolling once the inspector scrolled, the
  Constraints toggle centred, the excluded-plane note colliding with the plot
  legend, a Level choice made while a preparation ran that left Fit blocked after
  Undo, and, in the review pass, a value field 6 px too narrow for its first digit
  (the theme's line-edit padding is 18 px, not 12) and the numbering range broken
  across lines. Each was fixed and the captures rerun; the figures below are from
  the final runs.

## Review pass at 1280×800

What was tried and what was kept, measured natively (13 px body; the form cell is
176 px, 166 px once the inspector scrolls; 230 px and more at 17 px):

| Arrangement | Estimated | Manual | Factor | Level | Kept |
| --- | --- | --- | --- | --- | --- |
| Choice line, value line under it (before) | fits | fits | fits | fits | no: stacked, one line more per setting |
| Combo, then value, one line | 165 | about 180–197 | 141 for the spin box with “× noise” alone, about 209 with the combo | about 159–176 | no: a manual value, a level or the factor overflows |
| Value, then a ▾ menu naming its source | 165 | 164 | 147 | 148 | **yes**: every state fits at 13 px, keyboard order follows the line (value, then source), the menu's accessible name gives its current choice |

The value field sizes to its text (18 px of padding, border and margin), so no
number is ever shown cut; the menu beside it uses a compact padding. The header
drops its ANALYSIS overline (the Run | Model switch names the page) and keeps the
dataset and its counts: one line less. An entry that is not applied says so
(“Not applied: …”) and hides what the setting in effect derives: the estimate, the
factor or level, the planes fitted. The planes caption says how planes are
numbered (“numbered 1–131”); the files and the CLI count from 0
([outputs.md](../outputs.md#excluded-planes)). On an excluded plane, “No model on
this plane” sits in the header strip above the plot, never over the contours; the
plot and both slices keep their place (`test_gui_consistency.py`).

**CI fonts (eb2f6a9).** On Linux, where DejaVu Sans renders the UI font, the
Statistical noise row in Manual mode with 1.234 × 10⁻⁵ needed 293 px of 288. The
source menu is now caption-sized, 2 px from its value: 285 px in DejaVu Sans, 271
natively. The width tests run under both fonts and name the widest row. Rerun at
1280×800 and at 17 px (1160, 1280), 60 captures: no sideways scrolling, clipped
label or cut field; `model-as-loaded-…` and `settings-set-1280x800-light.webp` are
from that run.

## Measurements

From `measurements.json` (13 px, 60 captures) and `measurements-17px.json` (40):

| | 13 px | 17 px |
| --- | --- | --- |
| Inspector viewport (no scrollbar / scrollbar) | 320 / 310 | 382–418 / 372–408 |
| Widest Model page minimum (any state) | 272 | 314 |
| Sideways scrolling, clipped labels, cut fields | none | none |
| First setting's top, Model page unscrolled (fold: 607 at 1160×720) | 132–153 | 146–171 |
| Fit enabled while an entry is not applied | never | never |

## Plane exclusion: desktop fit equals `peakfit fit --exclude`

`uv run python tools/validate_gui_plane_exclusion.py` fits the bundled example
with the golden reference's arguments and `--exclude 0 40 41 130` (the planes
`validate_plane_exclusion.py` fixed), then prepares and fits the same dataset as
the desktop does with the configuration that run recorded. The gate, fixed before
the first run, is bit-for-bit agreement. Record (`plane-exclusion.json`, byte-identical when rerun at 806465d, after the review-pass code changes): 122
clusters, identical peaks and classifications, and every compared quantity
bit-identical (largest relative difference 0): noise, global χ² and degrees of
freedom, every nonlinear parameter's value, standard error, bounds and vary flag,
amplitudes, their conditional standard errors and covariance, and each cluster's
χ². `tests/test_gui_model_settings.py` checks the same, and that a manual noise or
an absolute level gives the clusters and outcome of the equivalent TOML, on the
synthetic six-plane dataset in every test run.

## Figures

In [gui-model-settings/](gui-model-settings/), with the two measurement records and
`plane-exclusion.json`. The `before-` figures are from the run before the review
pass, at the same state and size.

| Figure | State | Window |
| --- | --- | --- |
| `model-as-loaded-1280x800-light.webp` / `before-model-as-loaded-…` | Model page as loaded (inspector) | 1280×800, light |
| `settings-set-1280x800-light.webp` | Manual noise, Level, planes excluded (inspector) | 1280×800, light |
| `settings-set-1920x1080-light.webp` | the same, whole window | 1920×1080, light |
| `settings-set-17px-1160x720-light.webp` | the same at a 17 px system font (inspector) | 1160×720, light |
| `constraints-1280x800-dark.webp` | Constraints: a window and a rule (inspector) | 1280×800, dark |
| `invalid-1280x800-light.webp` / `before-invalid-…` | entries not applied; Fit blocked (inspector) | 1280×800, light |
| `invalid-17px-1280x800-dark.webp` | the same at 17 px (inspector) | 1280×800, dark |
| `invalid-constraints-1160x720-dark.webp` | rule values not applied, an η rule with no parameter (inspector) | 1160×720, dark |
| `plane-excluded-1160x720-light.webp` | plane 41 excluded: the strip says not fitted | 1160×720, light |
| `run-after-fit-1920x1080-dark.webp` | Run page: noise, threshold, planes as submitted (inspector) | 1920×1080, dark |
| `excluded-plane-overlay-1280x800-light.webp` / `before-excluded-plane-overlay-…` | Overlay on plane 41: status in the header strip; profile note | 1280×800, light |
| `excluded-plane-overlay-1280x800-dark.webp` | the same | 1280×800, dark |
| `differs-from-run-1280x800-light.webp` | settings marked • against the run (inspector) | 1280×800, light |
| `after-undo-1280x800-dark.webp` | after Undo of the last edit: the run stays Outdated (inspector) | 1280×800, dark |

## Limits

- Offscreen tests check the unscrolled inspector (288 px at 13 px, and the widened
  inspector at 17 px); the scrolling inspector's 10 px less is checked natively by
  the capture tool, since offscreen fonts run wider than macOS's.
- The excluded plane's dotted line in the peak profile is faint by design (the
  theme's faint colour); the note under the profile names the plane.
