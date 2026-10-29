"""Reproduce real-data Cocoa/Qt captures and delivered-event performance evidence.

Run with ``uv run python tools/validate_gui_editor.py`` on an exposed native
window. Inputs are deterministic scene deliveries, not OS pointer synthesis or
input-to-photon measurements. A separate native walkthrough checks real gestures.
"""

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PySide6 import QtCore, QtWidgets

from peakfit.gui.editor.loading import read_dataset
from peakfit.gui.editor.window import EditorWindow

ROOT = Path(__file__).resolve().parents[1]
_EVENTS = 500
_LAST_RAPID_PLANE = 40


def wait_until(predicate, timeout=15):
    """Keep Qt painting and worker threads progressing while awaiting one delivery."""
    start = perf_counter()
    while not predicate() and perf_counter() - start < timeout:
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(4, loop.quit)
        loop.exec()
    if not predicate():
        raise TimeoutError("Native editor validation did not settle")


def settle(milliseconds=50):
    """Let actual native paint complete; this is a verification harness only."""
    loop = QtCore.QEventLoop()
    QtCore.QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


def percentiles(values):
    """Report tails and extrema rather than omit expensive frames."""
    return {
        name: float(value)
        for name, value in zip(
            ("p50_ms", "p95_ms", "p99_ms", "max_ms"),
            np.percentile(values, [50, 95, 99, 100]),
            strict=True,
        )
    }


def rss_mib():
    """Read the current process RSS, not a monotonically increasing maximum."""
    value = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], check=True, capture_output=True, text=True
    )
    return float(value.stdout.strip()) / 1024


def main():  # noqa: PLR0915
    """Capture sizes/themes, exact samples, bounded navigation and timed preview."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/validation/gui-v2-editor")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert isinstance(app, QtWidgets.QApplication)
    app.setStyle("Fusion")
    data = ROOT / "examples/02-advanced-fitting/data"
    start = perf_counter()
    dataset = read_dataset(data / "pseudo3d.ft2", data / "pseudo3d.list", data / "b1_offsets.txt")
    read_ms = (perf_counter() - start) * 1000
    window = EditorWindow()
    start = perf_counter()
    window.install_dataset(dataset)
    window.show()
    window.raise_()
    window.activateWindow()
    workspace = window.workspace
    assert workspace is not None
    canvas = workspace.canvas
    wait_until(lambda: not canvas.contour_pending)
    settle()
    startup_ms = (perf_counter() - start) * 1000
    records: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "qt_platform": app.platformName(),
        "dpr": window.devicePixelRatioF(),
        "source_shape": list(dataset.spectra.data.shape),
        "peaks": len(dataset.peaks),
        "sources": dict(dataset.sources),
        "cold_read_ms": read_ms,
        "install_and_contours_ms": startup_ms,
        "policy": "32 MiB / 8 planes numerical geometry; two active cached Qt sign paths",
        "input_method": "deterministic scene deliveries through preview coalescer; "
        "16 ms timer; exposed Qt paint completion",
        "input_to_photon": False,
        "captures": [],
        "preview": {},
        "rss_mib": [],
    }
    peak = next(p for p in workspace.peaks if p.name == "5N-H")
    workspace.select_peak(peak.id)
    for width, height in ((1280, 800), (1440, 900), (1040, 720)):
        window.resize(width, height)
        for theme in ("Light", "Dark"):
            window.apply_appearance(theme)
            canvas.full_spectrum()
            settle()
            name = f"editor-{width}-{theme.lower()}.webp"
            assert window.grab().save(str(args.output / name), "WEBP", 95)
            rect = canvas.main.getViewBox().sceneBoundingRect()
            records["captures"].append(
                {
                    "file": name,
                    "window": [window.width(), window.height()],
                    "plot_rect": [rect.width(), rect.height()],
                    "theme": theme,
                }
            )
    window.resize(1280, 800)
    records["rss_mib"].append(rss_mib())
    heartbeat = []
    last = [perf_counter()]
    heart = QtCore.QTimer()
    heart.setInterval(10)

    def beat():
        now = perf_counter()
        heartbeat.append((now - last[0]) * 1000)
        last[0] = now

    heart.timeout.connect(beat)
    heart.start()
    for theme in ("Light", "Dark"):
        window.apply_appearance(theme)
        canvas.set_pinned(False)
        latencies = []
        timings = []
        pending = [None]
        expected_sample = [None]
        count = [0]
        completed = [0]

        def painted(
            pending=pending,
            latencies=latencies,
            completed=completed,
            expected_sample=expected_sample,
        ):
            if pending[0] is not None and expected_sample[0] == (canvas.ix, canvas.iy):
                latencies.append((perf_counter() - pending[0]) * 1000)
                pending[0] = None
                completed[0] += 1

        canvas.painted.connect(painted)
        jobs, markers = canvas.contour_jobs, canvas.marker_rebuilds
        deliveries = canvas.preview_deliveries
        coalesced = canvas.preview_coalesced

        timer = QtCore.QTimer()

        def deliver(
            count=count,
            pending=pending,
            timings=timings,
            timer=timer,
            expected_sample=expected_sample,
        ):
            if count[0] >= _EVENTS:
                timer.stop()
                return
            if pending[0] is not None:
                # Record superseded events, preserving only the latest display request.
                timings.append("superseded")
            ix = 1 + (count[0] * 29) % (len(canvas.ppm_x) - 2)
            iy = 1 + (count[0] * 13) % (len(canvas.ppm_y) - 2)
            x, y = float(canvas.ppm_x[ix]), float(canvas.ppm_y[iy])
            expected_sample[0] = (ix, iy)
            pending[0] = perf_counter()
            canvas._pointer_moved(canvas.main.getViewBox().mapViewToScene(QtCore.QPointF(x, y)))
            count[0] += 1

        timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        timer.setInterval(16)
        timer.timeout.connect(deliver)
        started = perf_counter()
        timer.start()
        wait_until(
            lambda count=count, pending=pending: count[0] >= _EVENTS and pending[0] is None,
            timeout=30,
        )
        elapsed = perf_counter() - started
        canvas.painted.disconnect(painted)
        records["preview"][theme] = {
            **percentiles(latencies),
            "events": count[0],
            "painted": completed[0],
            "superseded": len(timings),
            "delivered_hz": count[0] / elapsed,
            "preview_deliveries": canvas.preview_deliveries - deliveries,
            "coalesced": canvas.preview_coalesced - coalesced,
            "contour_jobs": canvas.contour_jobs - jobs,
            "marker_rebuilds": canvas.marker_rebuilds - markers,
        }
        assert canvas.contour_jobs == jobs
        assert canvas.marker_rebuilds == markers
        records["rss_mib"].append(rss_mib())
    plane_times = []
    for plane in (1, 60, 131):
        start = perf_counter()
        workspace.set_plane(plane)
        wait_until(lambda: not canvas.contour_pending)
        settle(10)
        plane_times.append((perf_counter() - start) * 1000)
        iy, ix = canvas.iy, canvas.ix
        np.testing.assert_array_equal(
            canvas.curve_x.getData()[1], dataset.spectra.data[plane - 1, iy, :]
        )
        np.testing.assert_array_equal(
            canvas.curve_y.getData()[0], dataset.spectra.data[plane - 1, :, ix]
        )
    records["uncached_plane_ms"] = plane_times
    roundtrips = []
    for index in range(100):
        start = perf_counter()
        workspace.set_plane((1, 60, 131)[index % 3])
        wait_until(lambda: not canvas.contour_pending)
        settle(1)
        roundtrips.append((perf_counter() - start) * 1000)
        if index % 10 == 0:
            records["rss_mib"].append(rss_mib())
    records["cached_plane"] = percentiles(roundtrips)
    records["cache_bytes"] = canvas.cache_bytes
    records["cached_planes"] = len(canvas._cache)
    workspace.set_plane(1)
    start = perf_counter()
    for plane in range(1, 41):
        workspace.set_plane(plane)
    wait_until(lambda: not canvas.contour_pending)
    records["rapid_40_planes_ms"] = (perf_counter() - start) * 1000
    records["rapid_final_plane"] = canvas.plane + 1
    assert canvas.plane == _LAST_RAPID_PLANE - 1
    heart.stop()
    records["heartbeat"] = percentiles(heartbeat)
    records["rss_mib"].append(rss_mib())
    records["exact_samples"] = True
    records["scientific_revision"] = workspace.edits.revision
    (args.output / "measurements.json").write_text(json.dumps(records, indent=2) + "\n")
    window.close()
    print(  # noqa: T201
        json.dumps(
            {"measurements": str(args.output / "measurements.json"), "preview": records["preview"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
