# Spectral workspace: native slices, focus and window policy

Evidence for the one spectral view (native slices, residual lanes, region intensity
scale), cluster focus and the minimum-window policy
([decisions](../decisions.md), 2026-11-06). Reproduce with
`uv run python tools/validate_gui_spectral.py` from `tools/` on an exposed native
window; it writes [spectral-workspace.json](spectral-workspace.json). Window
captures (`--captures`) are kept outside the repository.

The tool loads the bundled 131×256×546 pseudo-3D dataset with 166 peaks and fits it
with the GUI's own Fit (default setup, 121 clusters), then measures on that run.
Pointer input is a deterministic scene delivery at 60 Hz with exposed Qt paint
completion, not OS pointer synthesis; real-mouse gestures are covered by the
separate native walkthrough.

| Check | Result (macOS, Cocoa, 2× DPR) |
| --- | --- |
| Window never enlarged; both side panels, slices and lanes | 1160×720, 1280×800, 1440×900 |
| 2D plot (logical px) | 360×364, 430×424, 590×524; minimum 360×300 kept |
| Slice and lane extents vs 2D plot; same ppm, same pixel | 0 px at every size |
| Region scale, full view | −1.31 … 2.74 ×10⁶ a.u., zero included |
| Scale through hover and planes 1/60/131 | unchanged |
| Scale after zooming on 53N-H | recomputed from the visible region |
| Hover delivery-to-paint, Overlay with lanes (500 events) | p50 4.4 ms, p95 6.1 ms, max 8.2 ms (target p95 ≤ 8 ms); no contour work, no rescale |
| Residual mode | lanes hidden, preference kept; view, sample and selection unchanged |
| Focus on 53N-H | members 53N-H, 147N-H; opens in 78 ms |
| Leaving focus | viewport, slices, manual scale, mode and sample restored exactly |
| Saved view state | strict JSON; slices and focus sizes restored |

Timings vary run to run; the JSON holds the latest run.
