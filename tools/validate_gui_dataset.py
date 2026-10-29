"""Exercise the bundled whole-dataset workflow with native Qt input and captures.

Run ``uv run python tools/validate_gui_dataset.py --output /tmp/peakfit-gui-review``.
Uses the normal Qt event loop while workers run; QTest.qWait starves Python worker
threads on some PySide builds and must not be used to benchmark this application.
The exported numerical results are disposable validation artifacts, not a session.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from peakfit.fit.simulation import project_final_outcome
from peakfit.gui.app import PeakFitWindow
from peakfit.gui.dataset_dialog import DatasetFitDialog
from peakfit.gui.dataset_fit import DatasetFitResult
from peakfit.gui.fit_export import export_fit_result


def wait(condition, timeout=30):
    """Process genuine Qt events and release the GIL while waiting for workers."""
    deadline = perf_counter() + timeout
    while not condition() and perf_counter() < deadline:
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(10, loop.quit)
        loop.exec()
    assert condition(), "Timed out waiting for GUI work"


def click(widget):
    """Send mouse input through the actual widget event handlers."""
    QtTest.QTest.mouseClick(widget, QtCore.Qt.MouseButton.LeftButton)


def plane_input(viewer, plane):
    """Commit an experimental plane number using normal text and keyboard input."""
    field = viewer.plane_spin.lineEdit()
    click(field)
    field.selectAll()
    QtTest.QTest.keyClicks(field, str(plane))
    QtTest.QTest.keyClick(field, QtCore.Qt.Key.Key_Return)
    assert viewer.plane == plane - 1


def comparison_input(viewer, index):
    """Use the comparison control's keyboard interface, including both end modes."""
    control = viewer.comparison.mode
    control.setFocus()
    QtTest.QTest.keyClick(control, QtCore.Qt.Key.Key_Home)
    for _ in range(index):
        QtTest.QTest.keyClick(control, QtCore.Qt.Key.Key_Down)
    assert control.currentIndex() == index


def main():  # noqa: PLR0915
    """Assert real workflow behavior, measure latency, and save inspectable evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/peakfit-gui-review"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication([])
    app.setApplicationName("PeakFit")
    window = PeakFitWindow()
    root = Path(__file__).resolve().parents[1] / "examples/02-advanced-fitting/data"
    window.open_spectrum(root / "pseudo3d.ft2", root / "pseudo3d.list", root / "b1_offsets.txt")
    viewer = window.viewer
    assert viewer is not None
    window.resize(1280, 800)
    window.show()
    window.raise_()
    window.activateWindow()
    wait(lambda: bool(viewer.contours))
    report: dict[str, Any] = {
        "platform": sys.platform,
        "qt_backend": app.platformName(),
        "sizes": [],
    }
    ticks = []
    phase = "initial"
    last_tick = perf_counter()

    def tick():
        nonlocal last_tick
        now = perf_counter()
        ticks.append((phase, now - last_tick))
        last_tick = now

    heartbeat = QtCore.QTimer()
    heartbeat.setInterval(20)
    heartbeat.timeout.connect(tick)
    heartbeat.start()

    def capture(name, target=window):
        nonlocal last_tick
        # Capturing/writing Retina images is test work, not a user interaction.
        heartbeat.stop()
        # Axis widths follow newly formatted tick text on a deferred layout pass.
        # Capture the settled native view, rather than forcing its first paint.
        for _ in range(3):
            loop = QtCore.QEventLoop()
            QtCore.QTimer.singleShot(20, loop.quit)
            loop.exec()
        native = target.grab().toImage()
        if name.endswith("-residual"):
            # Qt's quality=100 WebP mode is lossless. Verify the decoded pixels
            # explicitly; dense residual PNGs exceed the repository's file limit.
            destination = args.output / (name + ".webp")
            assert native.save(str(destination), b"WEBP", 100)
            original = native.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
            decoded = QtGui.QImage(str(destination)).convertToFormat(
                QtGui.QImage.Format.Format_RGBA8888
            )
            assert original.size() == decoded.size()
            assert bytes(original.constBits()) == bytes(decoded.constBits())
        else:
            assert native.save(str(args.output / (name + ".png")))
        last_tick = perf_counter()
        heartbeat.start()

    errors = []

    def confirm_replacement():
        try:
            dialog = app.activeModalWidget()
            assert isinstance(dialog, QtWidgets.QMessageBox)
            click(dialog.button(QtWidgets.QMessageBox.StandardButton.Discard))
        except Exception as error:
            errors.append(error)
            active = app.activeModalWidget()
            if isinstance(active, QtWidgets.QDialog):
                active.reject()

    def review(*, replacement=False):
        try:
            dialog = app.activeModalWidget()
            assert isinstance(dialog, DatasetFitDialog)
            wait(lambda: dialog.prepared is not None)
            assert dialog.prepared is not None
            assert dialog.passes.value() == 2  # noqa: PLR2004
            assert len(dialog.prepared.clusters) == 121  # noqa: PLR2004
            assert dialog.prepared.contour_level != viewer.lowest_contour_level
            capture("settings", dialog)
            click(dialog.advanced_toggle)
            assert dialog.parameters.rowCount() == len(dialog.prepared.parameters)
            capture("settings-advanced", dialog)
            if replacement:
                QtCore.QTimer.singleShot(50, confirm_replacement)
            click(dialog.run)
        except Exception as error:
            errors.append(error)
            active = app.activeModalWidget()
            if isinstance(active, QtWidgets.QDialog):
                active.reject()

    try:
        phase = "fitting"
        started = perf_counter()
        QtCore.QTimer.singleShot(50, review)
        click(viewer.workspace.dataset_fit)
        assert not errors, errors
        wait(lambda: not viewer.fit_busy)
        assert viewer._fit_process.state == "completed", viewer.workspace.summary.toolTip()
        report["fit_wall_seconds_including_review"] = perf_counter() - started
        retained = viewer.fit_result
        assert retained is not None
        assert isinstance(retained.scientific, DatasetFitResult)
        scientific = retained.scientific
        report["fit_execution_seconds"] = scientific.execution_seconds
        report["input_bytes"] = retained.input_bytes
        report["summary"] = viewer.workspace.summary.text()
        report["optimizer_passes"] = scientific.outcome.n_optimizer_passes
        report["correction_updates"] = scientific.outcome.n_correction_updates
        assert report["optimizer_passes"] == 2  # noqa: PLR2004
        assert report["correction_updates"] == 1
        assert viewer.workspace.details.isHidden()
        phase = "projection"
        wait(
            lambda: viewer.comparison._current is not None and viewer.comparison._slice is not None
        )
        assert viewer.comparison._current is not None
        assert np.any(viewer.comparison._current.grid.values < 0)
        assert np.any(viewer.comparison._current.grid.values > 0)
        # Measure uncached plane navigation through the real numeric control.
        navigation = []
        for number in (2, 3, 25, 60, 131):
            started = perf_counter()
            plane_input(viewer, number)
            input_seconds = perf_counter() - started
            assert viewer.comparison._current is None
            assert not any(path.isVisible() for path in viewer.comparison.paths)
            assert "Calculating" in viewer.spectrum_plot.getPlotItem().titleLabel.text
            wait(
                lambda: viewer.comparison._current is not None
                and viewer.comparison._slice is not None
            )
            navigation.append(
                {
                    "plane": number,
                    "input_seconds": input_seconds,
                    "model_ready_seconds": perf_counter() - started,
                }
            )
        report["uncached_navigation"] = navigation
        phase = "cached_navigation"
        started = perf_counter()
        plane_input(viewer, 60)
        wait(lambda: viewer.comparison._current is not None)
        report["cached_navigation_seconds"] = perf_counter() - started
        # Rapid input must leave only the final requested experimental/model pair.
        for number in (8, 19, 35, 1):
            plane_input(viewer, number)
        wait(
            lambda: viewer.comparison._current is not None and viewer.comparison._slice is not None
        )
        assert viewer.plane == 0
        phase = "verification"
        coordinates = tuple(index.ravel() for index in np.indices(viewer.spectra.data.shape[1:]))
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                project_final_outcome,
                scientific.outcome,
                scientific.simulation_snapshot,
                coordinates,
                planes=[0],
            )
            wait(future.done)
            projected = future.result()
            assert projected is not None
            expected = projected[:, 0].reshape(viewer.spectra.data.shape[1:])
        assert viewer.comparison._current is not None
        np.testing.assert_array_equal(viewer.comparison._current.grid.values, expected)
        np.testing.assert_allclose(
            viewer.comparison.model_x.yData, expected[viewer.iy], rtol=1e-12, atol=1e-9
        )
        np.testing.assert_allclose(
            viewer.comparison.model_y.yData, expected[:, viewer.ix], rtol=1e-12, atol=1e-9
        )
        phase = "interaction"
        # Click an actual spectral position; both measured and calculated slices follow.
        point = viewer.spectrum_plot.getViewBox().mapViewToScene(
            QtCore.QPointF(float(viewer.ppm_x[220]), float(viewer.ppm_y[180]))
        )
        click_point = viewer.spectrum_plot.mapFromScene(point)
        QtTest.QTest.mouseClick(
            viewer.spectrum_plot.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=click_point
        )
        wait(lambda: viewer.comparison._slice is not None)
        assert (viewer.iy, viewer.ix) == (180, 220)
        np.testing.assert_allclose(
            viewer.comparison.model_x.yData, expected[180], rtol=1e-12, atol=1e-9
        )
        comparison_input(viewer, 0)
        assert not any(path.isVisible() for path in viewer.comparison.paths)
        comparison_input(viewer, 2)
        wait(
            lambda: viewer.comparison.residual.isVisible() and viewer.comparison._slice is not None
        )
        np.testing.assert_allclose(
            viewer.comparison.model_x.yData,
            (viewer.spectra.data[0, viewer.iy] - expected[viewer.iy]) / scientific.outcome.noise,
            rtol=1e-11,
            atol=1e-9,
        )
        # Capture a useful peak-centered pair of traces after also testing a
        # background crosshair. Selection must retain the total comparison scope.
        from peakfit.gui.spectrum import display_position  # noqa: PLC0415

        peak = viewer.peaks[0]
        point = viewer.spectrum_plot.getViewBox().mapViewToScene(
            QtCore.QPointF(
                display_position(viewer.x_axis, peak.positions[1]),
                display_position(viewer.y_axis, peak.positions[0]),
            )
        )
        QtTest.QTest.mouseClick(
            viewer.spectrum_plot.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            pos=viewer.spectrum_plot.mapFromScene(point),
        )
        wait(lambda: viewer.comparison._slice is not None)
        assert viewer.comparison.scope.text() == "Total model"
        for scheme, name in (
            (QtCore.Qt.ColorScheme.Light, "light"),
            (QtCore.Qt.ColorScheme.Dark, "dark"),
        ):
            app.styleHints().setColorScheme(scheme)
            for width, height in ((1280, 800), (1440, 900)):
                window.resize(width, height)
                app.processEvents()
                report["sizes"].append(
                    {
                        "theme": name,
                        "requested": [width, height],
                        "actual": list(window.size().toTuple()),
                    }
                )
                capture(f"dataset-{width}-{name}-residual")
                comparison_input(viewer, 1)
                wait(
                    lambda: viewer.comparison._current is not None
                    and viewer.comparison._slice is not None
                )
                capture(f"dataset-{width}-{name}-overlay")
                comparison_input(viewer, 2)
        comparison_input(viewer, 1)
        phase = "cancellation"
        QtCore.QTimer.singleShot(50, lambda: review(replacement=True))
        click(viewer.workspace.dataset_fit)
        assert not errors, errors
        wait(lambda: viewer._fit_process.state == "running")
        started = perf_counter()
        click(viewer.cancel_fit)
        wait(lambda: not viewer.fit_busy)
        report["cancel_seconds"] = perf_counter() - started
        assert viewer._fit_process.state == "canceled"
        assert viewer.fit_result is retained

        # Result-loss protection is a real native close interaction.
        def cancel_close():
            dialog = app.activeModalWidget()
            assert isinstance(dialog, QtWidgets.QMessageBox)
            click(dialog.button(QtWidgets.QMessageBox.StandardButton.Cancel))

        QtCore.QTimer.singleShot(50, cancel_close)
        assert not window.close()
        assert window.isVisible()
        assert viewer.fit_result is retained
        phase = "export"
        export_fit_result(scientific, args.output / "scientific-export")
        viewer.exported_result_id = scientific.run_id
        # Edit through the inspector and confirm that even Undo does not recertify.
        phase = "editing"
        peak = viewer.peaks[0]
        point = viewer.spectrum_plot.getViewBox().mapViewToScene(
            QtCore.QPointF(
                display_position(viewer.x_axis, peak.positions[1]),
                display_position(viewer.y_axis, peak.positions[0]),
            )
        )
        QtTest.QTest.mouseClick(
            viewer.spectrum_plot.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            pos=viewer.spectrum_plot.mapFromScene(point),
        )
        assert viewer.selected_id == peak.id, (
            viewer.selected_id,
            peak.id,
            point,
            window.isActiveWindow(),
        )
        click(viewer.workspace.edit)
        field = viewer.inspector.name
        click(field)
        field.selectAll()
        QtTest.QTest.keyClicks(field, "Edited assignment")
        QtTest.QTest.keyClick(field, QtCore.Qt.Key.Key_Return)
        assert not viewer.comparison.active
        assert "Outdated" in viewer.workspace.summary.text()
        click(viewer.workspace.close_button)
        # Direct QTest input does not ask macOS to foreground a background process.
        # Establish the same Qt activation/focus precondition as an interactive user.
        QtWidgets.QApplication.setActiveWindow(window)
        viewer.spectrum_plot.setFocus()
        wait(lambda: app.focusWidget() is viewer.spectrum_plot)
        QtTest.QTest.keySequence(
            viewer.spectrum_plot, QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Undo)
        )
        assert viewer.peaks[0] == peak, (
            viewer.undo_action.isEnabled(),
            viewer.undo_action.shortcuts(),
            app.focusWidget(),
            window.isActiveWindow(),
            viewer.edits.can_undo,
        )
        assert not viewer.comparison.active
        assert viewer.fit_result is retained
        report["interactions"] = [
            "settings and starting values",
            "run two-pass fit",
            "plane text entry",
            "rapid navigation",
            "comparison keyboard control",
            "crosshair mouse input",
            "cancel replacement",
            "cancel close",
            "canonical export",
            "edit and undo without recertification",
        ]
        heartbeat.stop()
        report["heartbeat_interval_ms"] = 20
        report["heartbeat_note"] = (
            "Normal Qt event loop; synchronous test image capture/writes excluded."
        )
        report["heartbeat"] = {
            category: {
                "max_ms": max(values) * 1000,
                "p95_ms": float(np.percentile(values, 95)) * 1000,
                "samples": len(values),
            }
            for category in sorted({category for category, _ in ticks})
            if (values := [value for group, value in ticks if group == category])
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))  # noqa: T201
    finally:
        heartbeat.stop()
        viewer.stop_preparation()
        window.hide()
        app.quit()


if __name__ == "__main__":
    main()
