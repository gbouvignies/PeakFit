# THROWAWAY: W3 step 1 per-axis model layouts (issue #215)

Not product code. This folder exists only on `dev/w3-layout-eval-dacf8d` to capture
layout alternatives for the per-axis lineshape model in the inspector's future Model
page. Nothing imports it. Delete it when W3 proper starts from the chosen layout.

- `model_page.py`: (A) full table, (A′) compact table, (B) stacked groups. `ModelPage`
  places one above the other Model settings (clustering, noise, passes, planes,
  optimizer limits).
- `capture.py`: opens the real window on `examples/02-advanced-fitting`, waits for the
  real preparation, swaps the prototype page into the inspector and captures every
  layout × scenario × window size × appearance, with `measurements.json`.

```bash
uv run python prototypes/w3_axis_model/capture.py --output <directory outside the repo>
uv run python prototypes/w3_axis_model/capture.py --output <dir> --body-px 17 --sizes 1280x800 --appearance Light
```

Run it natively (no `QT_QPA_PLATFORM=offscreen`) for review captures.
