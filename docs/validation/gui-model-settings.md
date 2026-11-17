# Model settings: native workflow record and plane-exclusion equivalence (#216)

Visual, geometric and numerical record of W4: manual statistical noise, the
clustering threshold as a factor or an absolute level, plane exclusion and
axis-wide constraints on the Model page. The decision is in
[decisions.md](../decisions.md) (2026-11-17, “The remaining Model settings”).
Sizes are Qt logical pixels.

## How it was run

- **Code:** branch `dev/w4-model-settings-e3a3c8` at the review and capture fixes,
  merged with `main` at 500fe88 (#227). PySide6/Qt 6.10.1, Fusion style.
- **Machine:** macOS 27.2.1 on Apple silicon, built-in Retina display, DPR 2.0.
- **Captures:** `uv run python tools/capture_model_settings.py --output <dir>`
  (native Cocoa) drives the real window on the bundled `02-advanced-fitting`
  dataset (131 planes, 166 peaks) and saves each state at 1160×720, 1280×800 and
  1920×1080, light and dark, 54 in all:
  1. the Model page as loaded;
  2. each setting set: Manual noise (starting from the estimate), Level (starting
     from the level the factor gave), planes 1, 41–42 excluded, a ¹⁵N position
     window of 0.5 ppm and a rule, linewidth on all axes in 5–80 Hz;
  3. the same, scrolled to the constraints;
  4. each setting invalid: noise 0, level −5, planes 1–131, a rule start of 120 Hz
     outside 5–80 Hz, and an η rule (these axes are SP2: no η);
  5. the same, scrolled to the constraints;
  6. after Undo of every edit, plane 41 excluded from the plane strip's command;
  7. the GUI's own two-pass fit with plane 41 excluded, and its Run page;
  8. plane 41 in Overlay with 2N-H's profile;
  9. Model edits that differ from that run.
- **Review:** I looked at every capture. The first run found the manual noise
  clipped (“192 × 10⁴” for 1.192 × 10⁴), sideways scrolling once the inspector
  scrolled, the Constraints toggle centred, a long note on the excluded plane
  colliding with the plot legend, and, earlier, a Level choice made while a
  preparation ran that left Fit blocked after Undo. Each was fixed and the
  captures rerun; the figures below are from the final run.

## Measurements

From `measurements.json` (54 captures):

| | Value |
| --- | --- |
| Inspector viewport (no scrollbar / scrollbar) | 320 / 310 |
| Widest Model page minimum (any state) | 261 |
| Sideways scrolling, clipped labels | none in any capture |
| First setting's top, Model page unscrolled (fold: 607 at 1160×720) | 153–174 |
| Fit enabled while a setting is invalid | never; its tooltip gives the first reason |

The noise and threshold choices sit above their values: a choice and a value side
by side need more than the 166 px the cell has once the inspector scrolls.

## Plane exclusion: desktop fit equals `peakfit fit --exclude`

`uv run python tools/validate_gui_plane_exclusion.py` fits the bundled example
with the golden reference's arguments and `--exclude 0 40 41 130` (the planes
`validate_plane_exclusion.py` fixed), then prepares and fits the same dataset as
the desktop does with the configuration that run recorded. The gate, fixed before
the first run, is bit-for-bit agreement. Record (`plane-exclusion.json`): 122
clusters, identical peaks and classifications, and every compared quantity
bit-identical (largest relative difference 0): noise, global χ² and degrees of
freedom, every nonlinear parameter's value, standard error, bounds and vary flag,
amplitudes, their conditional standard errors and covariance, and each cluster's
χ². `tests/test_gui_model_settings.py` checks the same, and that a manual noise or
an absolute level gives the clusters and outcome of the equivalent TOML, on the
synthetic six-plane dataset in every test run.

## Figures

In [gui-model-settings/](gui-model-settings/), with `measurements.json` and
`plane-exclusion.json`:

| Figure | State | Window |
| --- | --- | --- |
| `model-as-loaded-1280x800-light.webp` | Model page as loaded (inspector) | 1280×800, light |
| `settings-set-1280x800-light.webp` | Manual noise, Level, planes excluded (inspector) | 1280×800, light |
| `settings-set-1920x1080-light.webp` | the same, whole window | 1920×1080, light |
| `constraints-1280x800-dark.webp` | Constraints: a window and a rule (inspector) | 1280×800, dark |
| `invalid-1280x800-light.webp` | invalid noise, level and planes; Fit blocked (inspector) | 1280×800, light |
| `invalid-constraints-1160x720-dark.webp` | invalid rule values and an η rule with no parameter (inspector) | 1160×720, dark |
| `plane-excluded-1160x720-light.webp` | plane 41 excluded: the strip says not fitted | 1160×720, light |
| `run-after-fit-1920x1080-dark.webp` | Run page: noise, threshold, planes as submitted (inspector) | 1920×1080, dark |
| `excluded-plane-overlay-1280x800-dark.webp` | Overlay on plane 41: experimental only; profile note | 1280×800, dark |
| `differs-from-run-1280x800-light.webp` | settings marked • against the run (inspector) | 1280×800, light |

## Limits

- A 17 px system font was not captured for W4; the width test runs at the default
  scale only.
- The excluded plane's dotted line in the peak profile is faint by design (the
  theme's faint colour); the note under the profile names the plane.
