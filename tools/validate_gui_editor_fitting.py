"""Native v2 whole-dataset science, lifecycle, visual and performance evidence."""

# ruff: noqa: PLR2004

import json
import platform
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from time import perf_counter
from typing import Any
from unittest.mock import patch

import numpy as np
from PySide6 import QtCore, QtTest, QtWidgets
from validate_gui_editor import percentiles, rss_mib, settle, wait_until

from peakfit.gui.dataset_dialog import DatasetFitDialog
from peakfit.gui.editor.window import EditorWindow
from peakfit.gui.fit_display import dataset_model_plane
from peakfit.gui.spectrum import read_dataset
from peakfit.io.readers.results import ResultsLoader

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/validation/gui-v2-fitting"


def click(widget):
    """Exercise the visible native control through Qt's input dispatch."""
    QtTest.QTest.mouseClick(widget, QtCore.Qt.MouseButton.LeftButton)


def main():  # noqa: PLR0915
    """Use real data and numerical outcomes, with no replayed fit fixture."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication([])
    app.setStyle("Fusion")
    data = ROOT / "examples/02-advanced-fitting/data"
    started = perf_counter()
    dataset = read_dataset(data / "pseudo3d.ft2", data / "pseudo3d.list", data / "b1_offsets.txt")
    window = EditorWindow()
    window.install_dataset(dataset)
    window.show()
    window.raise_()
    window.activateWindow()
    w = window.workspace
    assert w is not None
    c = w.canvas
    wait_until(lambda: not c.contour_pending)
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "qt_platform": app.platformName(),
        "dpr": window.devicePixelRatioF(),
        "shape": list(dataset.spectra.data.shape),
        "peaks": len(w.peaks),
        "load_install_ms": (perf_counter() - started) * 1000,
        "input_method": "QtTest controls and deterministic scene delivery; native Cocoa paint",
        "input_to_photon": False,
        "captures": [],
        "rss_mib": [rss_mib()],
    }
    beats = []
    phase = ["settings"]
    last = [perf_counter()]
    heartbeat = QtCore.QTimer()
    heartbeat.setInterval(10)

    def beat():
        now = perf_counter()
        beats.append((phase[0], (now - last[0]) * 1000))
        last[0] = now

    heartbeat.timeout.connect(beat)
    heartbeat.start()
    chosen = next(p for p in w.peaks if p.name == "5N-H")
    w.select_peak(chosen.id)
    w.set_plane(60)
    view = c.main.getViewBox().viewRange()
    seeds = w.peaks
    dialog_timer = QtCore.QTimer()
    dialog_timer.setInterval(30)

    def review_settings():
        dialogs = w.findChildren(DatasetFitDialog)
        if dialogs and dialogs[-1].run.isEnabled():
            dialog = dialogs[-1]
            assert dialog.passes.value() == 2
            dialog.grab().save(str(OUTPUT / "settings.webp"), "WEBP", 95)
            assert dialog.prepared is not None
            report["prepared_clusters"] = len(dialog.prepared.clusters)
            report["prepared_peaks"] = len(dialog.prepared.identities)
            dialog_timer.stop()
            click(dialog.run)

    dialog_timer.timeout.connect(review_settings)
    dialog_timer.start()
    started = perf_counter()
    click(w.fit_button)
    phase[0] = "fit"
    wait_until(lambda: not w.fit_busy, timeout=240)
    assert w.fitting.current, w.fitting.failure
    report["first_fit_including_settings_s"] = perf_counter() - started
    result = w.fitting.result
    assert result is not None
    assert result.inputs.snapshot.peaks == seeds
    assert result.outcome.n_optimizer_passes == 2
    assert result.outcome.n_correction_updates == 1
    assert len(w.peaks) == 166
    assert c.comparison_mode == 1
    assert c.selected_id == chosen.id
    assert c.pin_peak_id == chosen.id
    np.testing.assert_allclose(c.main.getViewBox().viewRange(), view)
    report["summary"] = w.summary.text()
    report["positions_changed"] = sum(
        p.positions != q.positions for p, q in zip(seeds, w.peaks, strict=True)
    )
    assert w.fitting.completion is not None
    report["completion_warnings"] = list(w.fitting.completion.warnings)
    phase[0] = "capture_pending"
    for mode, name in ((1, "Overlay"), (2, "Residual")):
        w.set_mode(mode)
        wait_until(lambda: bool(c.paths) and c._displaying_experiment, timeout=1)
        assert c.contour_pending
        assert name in c.pending_label.toPlainText()
        window.grab().save(str(OUTPUT / f"pending-{name.lower()}.webp"), "WEBP", 95)
    w.set_mode(1)
    settle(20)
    phase[0] = "projection"
    wait_until(lambda: not c.contour_pending, timeout=60)
    with ThreadPoolExecutor(max_workers=1) as worker:
        expected = worker.submit(
            dataset_model_plane, result, c.spectra.spectral_params, c.plane, cancel=Event()
        )
        wait_until(expected.done, timeout=60)
        assert c._model is not None
        np.testing.assert_array_equal(c._model.values, expected.result().values)
    report["projection_max_abs_error"] = 0.0
    phase[0] = "capture"
    for size in ((1280, 800), (1440, 900), (1040, 720)):
        window.resize(*size)
        for theme in ("Light", "Dark"):
            window.apply_appearance(theme)
            for mode in (1, 2):
                click(w.mode_buttons[mode])
                wait_until(lambda: not c.contour_pending, timeout=60)
                settle(40)
                filename = (
                    f"{size[0]}-{theme.lower()}-{'overlay' if mode == 1 else 'residual'}.webp"
                )
                window.grab().save(str(OUTPUT / filename), "WEBP", 95)
                rect = c.main.getViewBox().sceneBoundingRect()
                report["captures"].append(
                    {"file": filename, "plot_rect": [rect.width(), rect.height()]}
                )
                if mode == 2:
                    assert c._model is not None
                    native = (
                        dataset.spectra.data[c.plane] - c._model.values
                    ) / result.outcome.noise
                    np.testing.assert_array_equal(c.curve_x.getData()[1], native[c.iy])
                    np.testing.assert_array_equal(c.curve_y.getData()[0], native[:, c.ix])
    window.resize(1280, 800)
    window.apply_appearance("Light")
    # Review lives in the inspector: select the next cluster with an engine warning.
    w.fitting.next_warning(None)
    settle(100)
    window.grab().save(str(OUTPUT / "review.webp"), "WEBP", 95)
    phase[0] = "hover"
    c.set_pinned(False)
    pending, count, superseded = [None], [0], [0]
    latencies = []
    jobs, markers = c.contour_jobs, c.marker_rebuilds
    timer = QtCore.QTimer()
    timer.setInterval(16)
    timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)

    def painted():
        if pending[0] is not None:
            latencies.append((perf_counter() - pending[0]) * 1000)
            pending[0] = None

    def deliver():
        if count[0] == 500:
            timer.stop()
            return
        if pending[0] is not None:
            superseded[0] += 1
        ix = 1 + count[0] * 29 % (len(c.ppm_x) - 2)
        iy = 1 + count[0] * 13 % (len(c.ppm_y) - 2)
        pending[0] = perf_counter()
        c._pointer_moved(
            c.main.getViewBox().mapViewToScene(
                QtCore.QPointF(float(c.ppm_x[ix]), float(c.ppm_y[iy]))
            )
        )
        count[0] += 1

    c.painted.connect(painted)
    timer.timeout.connect(deliver)
    started = perf_counter()
    timer.start()
    wait_until(lambda: count[0] == 500 and pending[0] is None, timeout=30)
    c.painted.disconnect(painted)
    report["hover"] = {
        **percentiles(latencies),
        "events": count[0],
        "painted": len(latencies),
        "superseded": superseded[0],
        "delivered_hz": count[0] / (perf_counter() - started),
        "contour_jobs": c.contour_jobs - jobs,
        "marker_rebuilds": c.marker_rebuilds - markers,
    }
    assert c.contour_jobs == jobs
    assert c.marker_rebuilds == markers
    settle(50)
    phase[0] = "navigation_uncached"
    times = []
    for plane in (1, 60, 131):
        started = perf_counter()
        w.set_plane(plane)
        wait_until(lambda: not c.contour_pending, timeout=60)
        times.append((perf_counter() - started) * 1000)
    report["uncached_plane_ms"] = times
    phase[0] = "navigation_cached"
    cached = []
    for index in range(100):
        started = perf_counter()
        w.set_plane((1, 60, 131)[index % 3])
        wait_until(lambda: not c.contour_pending, timeout=60)
        cached.append((perf_counter() - started) * 1000)
        # Deliver separate input turns rather than starving our own heartbeat
        # with 100 cached requests in one synchronous Python loop.
        settle(16)
        if index % 10 == 0:
            report["rss_mib"].append(rss_mib())
    report["cached_plane"] = percentiles(cached)
    report["cached_input_interval_ms"] = 16
    phase[0] = "navigation_rapid"
    started = perf_counter()
    for plane in range(1, 41):
        w.set_plane(plane)
        settle(2)
    wait_until(lambda: not c.contour_pending, timeout=60)
    report["rapid_40_plane_ms"] = (perf_counter() - started) * 1000
    assert c.plane == 39
    assert (result.run_id, 39) in c._model_cache
    report["cache_bytes"] = c.cache_bytes + sum(m.nbytes for m in c._model_cache.values())
    report["rss_mib"].append(rss_mib())
    settle(50)
    phase[0] = "export"
    with patch.object(QtWidgets.QFileDialog, "getExistingDirectory", return_value="/tmp"):
        started = perf_counter()
        w.fitting.export()
        report["export_s"] = perf_counter() - started
        report["export"] = str(Path("/tmp") / f"dataset-fit-{result.run_id}")
        assert result.run_id in w.fitting.exported
        assert not w.edits.dirty
    phase[0] = "cancel"
    prior = w.peaks
    click(w.fit_button)
    wait_until(lambda: w.fitting.process.busy, timeout=60)
    started = perf_counter()
    click(w.cancel_fit)
    report["cancel_ack_ms"] = (perf_counter() - started) * 1000
    wait_until(lambda: not w.fit_busy, timeout=10)
    report["cancel_reap_ms"] = (perf_counter() - started) * 1000
    assert w.fitting.result is result
    assert w.peaks == prior
    phase[0] = "refit"
    selected = w.peaks[0]
    w.move_peak(selected.id, (selected.positions[0], selected.positions[1] + 0.00001))
    assert not w.fitting.current
    assert c.comparison_mode == 0
    new_seeds = w.peaks
    started = perf_counter()
    click(w.fit_button)
    wait_until(lambda: not w.fit_busy, timeout=240)
    report["refit_s"] = perf_counter() - started
    assert w.fitting.current, w.fitting.failure
    assert w.fitting.result is not None
    assert w.fitting.result.inputs.snapshot.peaks == new_seeds
    phase[0] = "export_refit"
    with patch.object(QtWidgets.QFileDialog, "getExistingDirectory", return_value="/tmp"):
        w.fitting.export()
        assert w.fitting.result.run_id in w.fitting.exported
        assert not w.edits.dirty
    heartbeat.stop()
    # Reader round-trip verification belongs to the harness, outside the measured
    # application export path. The canonical writer validates before publication.
    summary = ResultsLoader(Path(report["export"])).load_summary()
    for loaded, original in zip(summary.clusters, result.outcome.clusters, strict=True):
        assert loaded.cluster_id == original.cluster_id
        if original.analytical_evaluation is not None:
            assert loaded.analytical_evaluation is not None
            np.testing.assert_array_equal(
                loaded.analytical_evaluation.amplitudes, original.analytical_evaluation.amplitudes
            )
    report["heartbeat"] = {
        name: percentiles([v for p, v in beats if p == name])
        for name in sorted({p for p, v in beats})
    }
    (OUTPUT / "measurements.json").write_text(json.dumps(report, indent=2) + "\n")
    assert window.protect_edits()
    window.close()
    assert not window.isVisible()
    print(  # noqa: T201
        json.dumps(
            {key: value for key, value in report.items() if key not in ("captures", "sources")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
