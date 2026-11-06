"""Real-data evidence for the shared spectral view: alignment, scales, hover, transitions.

Run with ``uv run python tools/validate_gui_spectral.py`` on an exposed native
window. It fits the bundled pseudo-3D dataset with the GUI's own Fit, then checks
on that run: slice/plot alignment at the reference window sizes, the region
intensity scale through hover, planes and zoom, hover delivery-to-paint latency
with residual lanes in Overlay, mode and focus transitions, and saved view state.
Inputs are deterministic scene deliveries, not OS pointer synthesis; a separate
native walkthrough with a real mouse checks gestures. Captures, when requested,
go to ``--captures`` (keep them outside the repository).
"""

# ruff: noqa: PLR2004

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter
from typing import Any
from unittest.mock import patch

from PySide6 import QtCore, QtWidgets
from validate_gui_editor import percentiles, settle, wait_until

from peakfit.gui.editor import slices
from peakfit.gui.editor.spectral_geometry import MAIN_MIN
from peakfit.gui.editor.window import EditorWindow
from peakfit.gui.spectrum import read_dataset

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/validation/spectral-workspace.json"
_EVENTS = 500
_SIZES = ((1160, 720), (1280, 800), (1440, 900))
_HOVER_P95_MS = 8.0


def alignment(canvas) -> dict[str, float]:
    """Largest scene-pixel mismatch of each slice and lane with the 2D plot's extent."""
    main = canvas.box.sceneBoundingRect()
    errors = {}
    for name, item, vertical in (
        ("row", canvas.row_box, False),
        ("row_lane", canvas.row_lane, False),
        ("column", canvas.column_box, True),
        ("column_lane", canvas.column_lane, True),
    ):
        if not item.isVisible():
            continue
        rect = item.sceneBoundingRect()
        if vertical:
            errors[name] = max(abs(rect.top() - main.top()), abs(rect.height() - main.height()))
        else:
            errors[name] = max(abs(rect.left() - main.left()), abs(rect.width() - main.width()))
    # The same ppm maps to the same screen coordinate in the plot and its slice.
    (x0, x1), (y0, y1) = canvas.box.viewRange()
    x, y = (x0 + x1) / 2, (y0 + y1) / 2
    point = canvas.box.mapViewToScene(QtCore.QPointF(x, y))
    if canvas.row_box.isVisible():
        row = canvas.row_box.mapViewToScene(QtCore.QPointF(x, 0))
        errors["row_ppm"] = abs(row.x() - point.x())
    if canvas.column_box.isVisible():
        column = canvas.column_box.mapViewToScene(QtCore.QPointF(0, y))
        errors["column_ppm"] = abs(column.y() - point.y())
    return errors


def hover(canvas) -> dict[str, Any]:
    """Delivery-to-paint latency of pointer previews at 60 Hz over native samples."""
    canvas.set_pinned(False)
    latencies: list[float] = []
    pending: list[float | None] = [None]
    expected: list[tuple[int, int] | None] = [None]
    count = [0]

    def painted():
        if pending[0] is not None and expected[0] == (canvas.ix, canvas.iy):
            latencies.append((perf_counter() - pending[0]) * 1000)
            pending[0] = None

    def deliver():
        if count[0] >= _EVENTS:
            timer.stop()
            return
        ix = 1 + (count[0] * 29) % (len(canvas.ppm_x) - 2)
        iy = 1 + (count[0] * 13) % (len(canvas.ppm_y) - 2)
        x, y = float(canvas.ppm_x[ix]), float(canvas.ppm_y[iy])
        expected[0], pending[0] = (ix, iy), perf_counter()
        canvas._pointer_moved(canvas.box.mapViewToScene(QtCore.QPointF(x, y)))
        count[0] += 1

    canvas.painted.connect(painted)
    jobs, bounds = canvas.contour_jobs, canvas.slice_bounds()
    timer = QtCore.QTimer()
    timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
    timer.setInterval(16)
    timer.timeout.connect(deliver)
    timer.start()
    wait_until(lambda: count[0] >= _EVENTS and pending[0] is None, timeout=30)
    canvas.painted.disconnect(painted)
    assert canvas.contour_jobs == jobs, "hover launched contour work"
    assert canvas.slice_bounds() == bounds, "hover changed the slice scale"
    return {**percentiles(latencies), "events": count[0], "painted": len(latencies)}


def main() -> None:  # noqa: PLR0915
    """Fit the bundled dataset, then measure the spectral view on that run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--captures", type=Path, help="directory for window captures")
    args = parser.parse_args()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert isinstance(app, QtWidgets.QApplication)
    data = ROOT / "examples/02-advanced-fitting/data"
    dataset = read_dataset(data / "pseudo3d.ft2", data / "pseudo3d.list", data / "b1_offsets.txt")
    window = EditorWindow()
    window.resize(*_SIZES[1])
    window.show()
    window.install_dataset(dataset)
    w = window.workspace
    assert w is not None
    c = w.canvas
    w.fitting.setup_seen = True
    wait_until(lambda: w.fitting.prepared is not None, timeout=60)
    start = perf_counter()
    w.fitting.fit()
    wait_until(lambda: not w.fitting.busy, timeout=600)
    assert w.fitting.current, w.fitting.failure
    record: dict[str, Any] = {
        "platform": platform.platform(),
        "qt_platform": app.platformName(),
        "dpr": window.devicePixelRatioF(),
        "fit_s": perf_counter() - start,
        "run": w.fitting.result.run_id[:8] if w.fitting.result else None,
        "input_method": "deterministic scene deliveries; 16 ms timer; exposed Qt paint",
    }
    wait_until(lambda: not c.contour_pending)
    peak = next(p for p in w.peaks if p.name == "53N-H")
    w.select_peak(peak.id)
    w.set_lanes(True)

    # Alignment and room at the reference sizes, both side panels open.
    layouts = []
    for width, height in _SIZES:
        window.resize(width, height)
        settle(100)
        main = c.arrangement.main
        layouts.append(
            {
                "window": [window.width(), window.height()],
                "enlarged": window.width() > width or window.height() > height,
                "navigator": not w.navigator.isHidden(),
                "inspector": not w.panel.isHidden(),
                "lanes": c.arrangement.sizes.lanes,
                "plot": [main[2], main[3]],
                "plot_minimum": main[2] >= MAIN_MIN[0] and main[3] >= MAIN_MIN[1],
                "alignment_px": alignment(c),
            }
        )
        if args.captures:
            args.captures.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(args.captures / f"overview-{width}.webp"), "WEBP", 95)
    record["layouts"] = layouts
    window.resize(*_SIZES[1])
    settle(100)

    # The region scale: fixed through hover and planes, recomputed on a view change.
    c.full_spectrum()
    region = c.slice_bounds()
    for ix, iy in ((10, 20), (200, 40), (500, 100)):
        c.sample(float(c.ppm_x[ix % c.ppm_x.size]), float(c.ppm_y[iy % c.ppm_y.size]))
    hover_stable = c.slice_bounds() == region
    planes = []
    for plane in (1, 60, 131):
        w.set_plane(plane)
        wait_until(lambda: not c.contour_pending)
        planes.append(c.slice_bounds() == region)
    x, y = c.display(peak)
    c.set_ranges((x - 0.3, x + 0.3), (y - 3, y + 3))
    rows = slices.index_range(c.y_axis, c.box.viewRange()[1])
    columns = slices.index_range(c.x_axis, c.box.viewRange()[0])
    zoomed = c.slice_bounds()
    record["scale"] = {
        "full_view": list(region),
        "zero_included": region[0] <= 0 <= region[1],
        "hover_stable": hover_stable,
        "planes_stable": all(planes),
        "zoomed": list(zoomed),
        "zoom_follows_region": zoomed == slices.intensity_range(c.spectra.data, rows, columns),
    }
    w.set_plane(1)
    wait_until(lambda: not c.contour_pending)

    # Hover delivery-to-paint in Overlay with lanes (the heaviest slice drawing).
    c.full_spectrum()
    settle(250)
    record["hover_overlay_lanes"] = hover(c)
    record["hover_target_p95_ms"] = _HOVER_P95_MS

    # Mode and focus transitions keep coordinates, selection and view state.
    w.select_peak(peak.id)
    view, sample, state = c.box.viewRange(), (c.ix, c.iy), c.slice_state()
    w.set_mode(2)
    wait_until(lambda: not c.contour_pending)
    residual_hides_lanes = not c.row_lane.isVisible() and c.sizes.lanes
    w.set_mode(1)
    wait_until(lambda: not c.contour_pending)
    mode_kept = (
        c.box.viewRange() == view
        and (c.ix, c.iy) == sample
        and w.selected_id == peak.id
        and c.row_lane.isVisible()
    )
    c.scale_intensity(120)
    overview = (c.box.viewRange(), c.slice_state(), c.comparison_mode, (c.ix, c.iy, c.pinned))
    start = perf_counter()
    w.open_focus(peak.id)
    wait_until(lambda: not c.contour_pending)
    settle(50)
    focus_ms = (perf_counter() - start) * 1000
    members = [p.name for p in w.peaks if p.id in w._focus_ids]
    focused = c.box.viewRange()
    if args.captures:
        window.grab().save(str(args.captures / "focus-1280.webp"), "WEBP", 95)
    w.close_focus()
    settle(50)
    restored = (c.box.viewRange(), c.slice_state(), c.comparison_mode, (c.ix, c.iy, c.pinned))
    record["transitions"] = {
        "residual_hides_lanes_keeps_preference": residual_hides_lanes,
        "mode_keeps_view_sample_selection": mode_kept,
        "focus_members": members,
        "focus_frame": focused,
        "focus_open_ms": focus_ms,
        "focus_restores_overview_exactly": restored == overview,
    }

    # Saved view state round-trips through strict JSON on the same workspace.
    saved = w.capture()
    json.dumps(saved.view or {}, allow_nan=False)
    state = c.slice_state()
    w.restore(saved)
    settle(50)
    record["persistence"] = {
        "slices_restored": c.slice_state() == state,
        "focus_slices": (saved.view or {}).get("focus_slices"),
    }
    checks = [
        all(not item["enlarged"] and item["plot_minimum"] for item in layouts),
        all(v <= 0.5 for item in layouts for v in item["alignment_px"].values()),
        layouts[0]["navigator"] and layouts[0]["inspector"] and layouts[0]["lanes"],
        record["scale"]["zero_included"],
        record["scale"]["hover_stable"],
        record["scale"]["planes_stable"],
        record["scale"]["zoom_follows_region"],
        record["hover_overlay_lanes"]["p95_ms"] <= _HOVER_P95_MS,
        residual_hides_lanes,
        mode_kept,
        record["transitions"]["focus_restores_overview_exactly"],
        record["persistence"]["slices_restored"],
    ]
    record["passed"] = all(checks)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    w.stop()
    with patch.object(EditorWindow, "protect_edits", return_value=True):  # not saved
        window.close()
    print(json.dumps({"output": str(args.output), "passed": record["passed"]}))  # noqa: T201


if __name__ == "__main__":
    main()
