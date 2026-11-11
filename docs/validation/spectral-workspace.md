# Spectral workspace: native slices, focus and window policy

Evidence for the one spectral view (native slices, residual lanes, a range per
slice), cluster focus and the minimum-window policy
([decisions](../decisions.md), 2026-11-06 and 2026-11-11). Reproduce with
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
| Slice ranges through 53N-H, full view | row −0.22 … 2.16, column −0.16 … 2.16 ×10⁶ a.u.; zero included |
| Slice ranges at another sample | row −0.10 … 0.69, column −0.06 … 0.06 ×10⁶ a.u. |
| Ranges through hover and zoom | each slice's own visible samples, every plane |
| Ranges through planes 1/60/131 | unchanged |
| Hover delivery-to-paint, Overlay with lanes (500 events) | p50 8.3 ms, p95 9.2 ms, max 12.1 ms (target p95 ≤ 10 ms); no contour work |
| Residual mode | lanes hidden, preference kept; view, sample and selection unchanged |
| Focus on 53N-H | members 53N-H, 147N-H; opens in 65 ms |
| Leaving focus | viewport, slices, gain, mode and sample restored exactly |
| Saved view state | strict JSON; slices and focus sizes restored |

Timings vary run to run; the JSON holds the latest run.
