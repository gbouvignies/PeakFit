# Run and Model pages: native workflow record (#215)

Visual and geometric record of W3: the inspector's analysis split into a Run and a
Model page, and the per-axis lineshape model as the compact table (A′) the user
chose from the layout prototypes. Display and interaction only: no stored or
exported value changes.

Sizes are Qt logical pixels. The decision and the element inventory are in
[decisions.md](../decisions.md) (2026-11-17, “Run and Model pages”).

## How it was run

- **Code:** branch `dev/w3-layout-eval-dacf8d` at the commit that adds this record,
  with the two-axis review fixes. PySide6/Qt 6.10.1, Fusion style.
- **Machine:** macOS 27.2.1 on Apple silicon, built-in Retina display, DPR 2.0.
- **Tool:** `uv run python tools/capture_run_model_pages.py --output <dir>` (native
  Cocoa, both appearances), and again with `--body-px 17 --no-fit` for a large
  system font. The tool drives the real window on the bundled
  `02-advanced-fitting` dataset (131 planes, 166 peaks, 121 clusters) through one
  workflow:
  1. the Model page before a fit;
  2. **+ J** on ¹⁵N F2, then J 92 Hz (range 46–184 Hz by default);
  3. Tab from J;
  4. an invalid lower bound (100 Hz);
  5. a phase on a Gaussian axis;
  6. removing the doublet (this spectrum has none);
  7. the GUI's own two-pass fit and the Run page;
  8. Model edits after the fit (3 passes, SP1 on ¹H F3), then Undo.
- **Review:** I looked at every capture, 28 at the 13 px body and 19 at 17 px.

## Measurements

| | 13 px body | 17 px body |
| --- | --- | --- |
| Inspector width (1160, 1280, 1920 px windows) | 320 | 382–418 |
| Table width needed: singlets / with a doublet | 240 / 259 | 274 / 294 |
| Sideways scrolling, clipped labels (every capture) | none | none |
| Planes row bottom, worst case (invalid range) / fold at 1160×720 | 497 / 607 | 551 / 607 |

Every Model setting down to Planes is above the fold in every capture, at every
size and at both fonts. When an entry is invalid, Fit is disabled and its tooltip
names the axis and the reason. **Undo settings change** restores a setting, and the
run stays Outdated.

## Figures

In [gui-run-model-pages/](gui-run-model-pages/), with `measurements.json` and
`measurements-17px.json` from the two runs:

| Figure | State | Window |
| --- | --- | --- |
| `model-before-fit-1920x1080-dark.webp` | Model page before the first fit | 1920×1080, dark |
| `adding-j-1280x800-light.webp` | + J pressed: J field and its hint (inspector) | 1280×800, light |
| `doublet-1160x720-light.webp` | ¹⁵N doublet with its range line | 1160×720, light |
| `doublet-1160x720-dark.webp` | the same (inspector) | 1160×720, dark |
| `invalid-range-1280x800-light.webp` | J outside its range: message under the axis, Fit disabled | 1280×800, light |
| `run-after-fit-1280x800-dark.webp` | Run page after the fit | 1280×800, dark |
| `run-after-fit-1920x1080-light.webp` | Run page after the fit | 1920×1080, light |
| `model-differs-1280x800-light.webp` | Model edits marked • against the run (inspector) | 1280×800, light |
| `doublet-17px-1160x720-light.webp` | the doublet at a 17 px system font (inspector) | 1160×720, light |
