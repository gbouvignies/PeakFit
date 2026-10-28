"""Small scientific handoff regressions for the read-only spectrum viewer."""

import sys
from pathlib import Path

import nmrglue as ng
import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from peakfit.gui.app import PeakFitWindow
from peakfit.gui.peaks import WorkingPeak
from peakfit.gui.spectrum import contour_paths, display_position, load_peaks, load_spectrum
from peakfit.gui.viewer import SpectrumViewer

DATA = Path(__file__).resolve().parents[1] / "examples" / "02-advanced-fitting" / "data"
pytestmark = pytest.mark.gui


def test_experimental_spectrum_inspection_preserves_coordinates_and_raw_data() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    spectra = load_spectrum(DATA / "pseudo3d.ft2", DATA / "b1_offsets.txt")
    spectra.data.flags.writeable = False
    viewer = SpectrumViewer(spectra)
    viewer.resize(1200, 800)
    viewer.show()
    app.processEvents()
    try:
        # Both traces follow zooming and panning in the spectrum, including F2 → trace X.
        for x_range, y_range in [((7.5, 9.0), (115.0, 120.0)), ((8.0, 9.5), (120.0, 125.0))]:
            viewer.spectrum_plot.getViewBox().setRange(xRange=x_range, yRange=y_range, padding=0)
            app.processEvents()
            np.testing.assert_allclose(viewer.trace_x.getViewBox().viewRange()[0], x_range)
            np.testing.assert_allclose(viewer.trace_y.getViewBox().viewRange()[0], y_range)
            assert viewer.trace_y.getViewBox().state["xInverted"]
        viewer.reset_view()
        app.processEvents()
        # Click in the real plotted coordinate system, then inspect a different plane.
        point = viewer.spectrum_plot.getViewBox().mapViewToScene(
            QtCore.QPointF(float(viewer.ppm_x[250]), float(viewer.ppm_y[100]))
        )
        QtTest.QTest.mouseClick(
            viewer.spectrum_plot.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            pos=viewer.spectrum_plot.mapFromScene(point),
        )
        viewer.set_plane(8)
        iy, ix = viewer.iy, viewer.ix
        assert abs(ix - 250) <= 1
        assert abs(iy - 100) <= 1
        assert viewer.plane == 7
        np.testing.assert_array_equal(viewer.curve_x.getData()[1], spectra.data[7, iy, :])
        np.testing.assert_array_equal(viewer.curve_y.getData()[1], spectra.data[7, :, ix])
        np.testing.assert_array_equal(viewer.curve_profile.getData()[0], spectra.z_values)
        np.testing.assert_array_equal(viewer.curve_profile.getData()[1], spectra.data[:, iy, ix])
        assert viewer.cross_x.value() == viewer.ppm_x[ix]
        assert viewer.cross_y.value() == viewer.ppm_y[iy]
        viewer.set_peaks(load_peaks(DATA / "pseudo3d.list", spectra))
        assert len(viewer.peaks) == 166
        peak_x, peak_y = viewer.peak_scatter.getData()
        first = viewer.peaks[0]
        assert peak_x[0] == pytest.approx(first.positions[1])
        assert peak_y[0] == pytest.approx(display_position(viewer.y_axis, first.positions[0]))
    finally:
        viewer.close()
        viewer.deleteLater()
        app.processEvents()


def test_2d_loading_signed_contours_and_indirect_peak_aliasing(tmp_path: Path) -> None:
    # Write a genuine 2D NMRPipe file from one bundled experimental plane.
    spectra = load_spectrum(DATA / "pseudo3d.ft2")
    udic = ng.pipe.guess_udic(spectra.dic, spectra.data)
    plane_udic = {"ndim": 2, 0: udic[1], 1: udic[2]}
    path = tmp_path / "plane.ft2"
    ng.pipe.write(str(path), ng.pipe.create_dic(plane_udic), spectra.data[0])
    single = load_spectrum(path)
    assert single.data.shape == (1, *spectra.data.shape[1:])
    np.testing.assert_array_equal(single.data[0], spectra.data[0])
    y_axis, x_axis = single.spectral_params
    plane = np.zeros(single.data.shape[1:])
    plane[80, 120] = 10
    plane[100, 200] = -10
    paths = contour_paths(plane, single.spectral_params, np.array([5, -5]))
    for contour, (iy, ix) in zip(paths, [(80, 120), (100, 200)], strict=True):
        bounds = contour.boundingRect()
        assert bounds.center().x() == pytest.approx(x_axis.pts2ppm(ix))
        assert bounds.center().y() == pytest.approx(y_axis.pts2ppm(iy))
        assert bounds.width() == pytest.approx(abs(x_axis.delta))
        assert bounds.height() == pytest.approx(abs(y_axis.delta))
    unfolded = y_axis.pts2ppm(y_axis.size + 3.25)
    assert display_position(y_axis, unfolded) == pytest.approx(y_axis.pts2ppm(3.25))
    direct = x_axis.pts2ppm(x_axis.size + 3.25)
    assert display_position(x_axis, direct) == direct
    bad_values = tmp_path / "values.txt"
    bad_values.write_text("42\n")
    np.testing.assert_array_equal(load_spectrum(path, bad_values).z_values, [42])
    bad_values.write_text("1\n2\n")
    with pytest.raises(ValueError, match="exactly 1"):
        load_spectrum(path, bad_values)


def test_peak_selection_precision_overlap_and_edit_history() -> None:  # noqa: PLR0915
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    spectra = load_spectrum(DATA / "pseudo3d.ft2")
    viewer = SpectrumViewer(spectra)
    peak = WorkingPeak.new("overlap", (120.123456789123, 8.123456789123))
    other = WorkingPeak.new("overlap", peak.positions)
    viewer.set_peaks([peak, other])
    viewer.resize(1200, 800)
    viewer.show()
    app.processEvents()
    point = viewer.spectrum_plot.getViewBox().mapViewToScene(
        QtCore.QPointF(peak.positions[1], display_position(viewer.y_axis, peak.positions[0]))
    )
    try:
        QtTest.QTest.mouseClick(
            viewer.spectrum_plot.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            pos=viewer.spectrum_plot.mapFromScene(point),
        )
        app.processEvents()
        menu = viewer.findChild(QtWidgets.QMenu)
        assert menu is not None
        assert len(menu.actions()) == 2
        menu.triggered.emit(menu.actions()[1])
        menu.close()
        assert viewer.selected_id == other.id
        viewer.open_details(0)
        assert viewer.inspector.x_field.text() == repr(other.positions[1])
        assert viewer.peaks == (peak, other)
        viewer.inspector.name.setText("renamed")
        viewer.inspector.x_field.setText("8.12345678912345")
        QtTest.QTest.keyClick(viewer.inspector.x_field, QtCore.Qt.Key.Key_Return)
        assert viewer.peaks[1].id == other.id
        assert viewer.peaks[1].positions[1] == 8.12345678912345
        viewer.delete_peak()
        assert viewer.peaks == (peak,)
        viewer.undo()
        assert viewer.peaks[1].name == "renamed"
        # Native text-field editing must not delete peaks or consume peak history.
        viewer.select_peak(other.id)
        field = viewer.inspector.name
        field.setFocus()
        field.selectAll()
        QtTest.QTest.keyClicks(field, "typed")
        QtTest.QTest.keySequence(field, QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Undo))
        assert field.text() == "renamed"
        QtTest.QTest.keyClick(field, QtCore.Qt.Key.Key_Backspace)
        pending_name = field.text()
        committed = viewer.peaks
        assert viewer.inspector.pending
        assert viewer.edits.can_undo
        assert viewer.edits.can_redo
        # Direct callbacks (including the spectrum's Delete shortcut) also preserve drafts.
        viewer.undo()
        viewer.redo()
        viewer.delete_peak()
        assert viewer.peaks == committed
        assert field.text() == pending_name
        assert not viewer.undo_action.isEnabled()
        assert not viewer.redo_action.isEnabled()
        assert not viewer.inspector.delete.isEnabled()
        assert viewer.edits.can_undo
        assert viewer.edits.can_redo
        assert "Apply" in viewer.inspector.details.text()
        assert "Escape" in viewer.inspector.details.text()
        QtTest.QTest.keyClick(field, QtCore.Qt.Key.Key_Escape)
        assert not viewer.inspector.pending
        assert viewer.undo_action.isEnabled()
        assert viewer.redo_action.isEnabled()
        assert viewer.inspector.delete.isEnabled()
        viewer.redo()
        assert viewer.peaks == (peak,)
    finally:
        viewer.close()
        viewer.deleteLater()
        app.processEvents()


def test_peak_placement_and_drag_preserve_fractional_unfolded_coordinates(monkeypatch) -> None:  # noqa: PLR0915
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    spectra = load_spectrum(DATA / "pseudo3d.ft2")
    viewer = SpectrumViewer(spectra)
    imported = load_peaks(DATA / "pseudo3d.list", spectra)
    original = imported[0]
    # Put an explicitly unfolded alias at a visible fractional position.
    displayed_y = display_position(viewer.y_axis, original.positions[0])
    unfolded_y = displayed_y + viewer.y_axis.size * viewer.y_axis.delta
    peak = WorkingPeak.new(original.name, (unfolded_y, original.positions[1]))
    viewer.set_peaks([peak, *imported[1:]])
    viewer.resize(1200, 800)
    viewer.show()
    viewer.spectrum_plot.getViewBox().setRange(
        xRange=(peak.positions[1] - 0.3, peak.positions[1] + 0.3),
        yRange=(displayed_y - 1.0, displayed_y + 1.0),
        padding=0,
    )
    app.processEvents()
    contours = [item.path() for item in viewer.contours]
    labels = tuple(viewer.peak_labels)
    monkeypatch.setattr(
        viewer, "_update_contours", lambda: pytest.fail("edit recalculated contours")
    )
    try:
        viewer.select_peak(peak.id, inspect=True)
        viewer._preview_peak_move(0.0123456789, 0.123456789, False)
        assert viewer.peaks[0] == peak  # preview has no scientific effects
        assert tuple(viewer.peak_labels) == labels
        viewer.cancel_interaction()
        assert viewer.peaks[0] == peak
        viewer._preview_peak_move(0.0123456789, 0.123456789, True)
        assert viewer.peaks[0].positions == (
            unfolded_y + 0.123456789,
            peak.positions[1] + 0.0123456789,
        )
        viewer.undo()
        assert viewer.peaks[0] == peak  # one undo restores the complete drag
        # Send a real held-button gesture through the scene, at the crosshair intersection.
        viewer.select_peak(peak.id, inspect=True)
        view = viewer.spectrum_plot.getViewBox()
        start = viewer.spectrum_plot.mapFromScene(
            view.mapViewToScene(QtCore.QPointF(peak.positions[1], displayed_y))
        )
        end = start + QtCore.QPoint(30, 20)
        viewport = viewer.spectrum_plot.viewport()
        QtTest.QTest.mouseMove(viewport, start)
        QtTest.QTest.qWait(30)
        QtTest.QTest.mousePress(viewport, QtCore.Qt.MouseButton.LeftButton, pos=start)
        move = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            QtCore.QPointF(end),
            QtCore.QPointF(viewport.mapToGlobal(end)),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        QtWidgets.QApplication.sendEvent(viewport, move)
        assert viewer.peaks[0] == peak
        QtTest.QTest.mouseRelease(viewport, QtCore.Qt.MouseButton.LeftButton, pos=end)
        delta = view.mapSceneToView(viewer.spectrum_plot.mapToScene(end)) - view.mapSceneToView(
            viewer.spectrum_plot.mapToScene(start)
        )
        assert viewer.peaks[0].positions == pytest.approx(
            (peak.positions[0] + delta.y(), peak.positions[1] + delta.x())
        )
        assert viewer.peaks[0] != peak
        viewer.undo()
        assert viewer.peaks[0] == peak
        viewer.select_peak(None)
        viewer._select_position(peak.positions[1] + 0.2, displayed_y + 0.5)
        before_pan = viewer.spectrum_plot.getViewBox().viewRange()
        QtTest.QTest.mouseMove(viewport, start)
        QtTest.QTest.qWait(30)
        QtTest.QTest.mousePress(viewport, QtCore.Qt.MouseButton.LeftButton, pos=start)
        QtWidgets.QApplication.sendEvent(viewport, move)
        QtTest.QTest.mouseRelease(viewport, QtCore.Qt.MouseButton.LeftButton, pos=end)
        assert viewer.peaks[0] == peak
        assert viewer.spectrum_plot.getViewBox().viewRange() != before_pan
        viewer.start_adding_peak()
        QtTest.QTest.keyClick(viewer.spectrum_plot, QtCore.Qt.Key.Key_Escape)
        assert not viewer.adding_peak
        assert len(viewer.peaks) == 166
        viewer.start_adding_peak()
        position = QtCore.QPointF(peak.positions[1] + 0.07, displayed_y + 0.27)
        scene = viewer.spectrum_plot.getViewBox().mapViewToScene(position)
        local = viewer.spectrum_plot.mapFromScene(scene)
        actual = viewer.spectrum_plot.getViewBox().mapSceneToView(
            viewer.spectrum_plot.mapToScene(local)
        )
        QtTest.QTest.mouseClick(
            viewer.spectrum_plot.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            pos=local,
        )
        assert not viewer.adding_peak
        assert len(viewer.peaks) == 167
        assert viewer.peaks[-1].positions == pytest.approx((actual.y(), actual.x()))
        assert contours == [item.path() for item in viewer.contours]
    finally:
        viewer.close()
        viewer.deleteLater()
        app.processEvents()


def test_export_and_loss_guards_include_typed_fields_and_keep_failed_saves_dirty(  # noqa: PLR0915
    tmp_path: Path, monkeypatch
) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = PeakFitWindow()
    source = DATA / "pseudo3d.ft2"
    peaks = DATA / "pseudo3d.list"
    window.open_spectrum(source, peaks)
    viewer = window.viewer
    assert viewer is not None
    viewer.select_peak(viewer.peaks[0].id)
    revision = viewer._scientific_revision
    viewer.inspector.name.setText("revert this draft")
    assert window.isWindowModified()
    viewer.inspector.revert.click()
    assert not window.isWindowModified()
    assert viewer._scientific_revision == revision
    viewer.inspector.name.setText("unsaved typed assignment")
    assert window.isWindowModified()
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "warning", lambda *_args: QtWidgets.QMessageBox.StandardButton.Cancel
    )
    window.open_spectrum(source)
    assert window.viewer is viewer
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", lambda *_args: (str(peaks), ""))
    window._open_peaks()
    assert viewer.inspector.name.text() == "unsaved typed assignment"
    close = QtGui.QCloseEvent()
    window.closeEvent(close)
    assert not close.isAccepted()
    # A close is transactional: rejecting any later confirmation leaves the fit running.
    with monkeypatch.context() as closing:
        closing.setattr(viewer._fit_process, "state", "running")
        canceled = []
        stopped = []
        closing.setattr(viewer._fit_process, "cancel", lambda: canceled.append(True))
        closing.setattr(viewer, "stop_preparation", lambda: stopped.append(True))
        closing.setattr(
            QtWidgets.QMessageBox,
            "question",
            lambda *_args: QtWidgets.QMessageBox.StandardButton.Yes,
        )
        closing.setattr(viewer, "confirm_result_loss", lambda _action: True)
        window.closeEvent(close)  # Running-fit Yes, then peak-edits Cancel.
        assert not close.isAccepted()
        assert not canceled
        assert not stopped
        closing.setattr(
            QtWidgets.QMessageBox,
            "warning",
            lambda *_args: QtWidgets.QMessageBox.StandardButton.Discard,
        )
        closing.setattr(viewer, "confirm_result_loss", lambda _action: False)
        window.closeEvent(close)  # Unsaved-result Cancel also preserves the running fit.
        assert not close.isAccepted()
        assert not canceled
        assert not stopped
        closing.setattr(viewer, "confirm_result_loss", lambda _action: True)
        window.closeEvent(close)
        assert close.isAccepted()
        assert stopped == [True]
        assert not canceled  # Shutdown owns cancellation, after every confirmation.
        stopped.clear()

        def finish_during_peak_confirmation():
            viewer._fit_process.state = "completed"
            return True

        closing.setattr(window, "_protect_edits", finish_during_peak_confirmation)
        closing.setattr(viewer, "confirm_result_loss", lambda _action: False)
        window.closeEvent(close)  # A result arriving during a dialog is still protected.
        assert not close.isAccepted()
        assert not stopped
    path = tmp_path / "working.csv"
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "exec", lambda _dialog: QtWidgets.QDialog.DialogCode.Accepted
    )
    monkeypatch.setattr(QtWidgets.QFileDialog, "selectedFiles", lambda _dialog: [str(path)])
    try:
        assert not window._export_peaks()  # Export never silently commits pending text.
        assert viewer.inspector.pending
        viewer.inspector.apply.click()
        assert window._export_peaks()
        assert not window.isWindowModified()
        restored = load_peaks(path, viewer.spectra)
        assert restored[0].name == "unsaved typed assignment"
        assert restored[0].positions == viewer.peaks[0].positions
        viewer.undo()
        assert window.isWindowModified()
        viewer.redo()
        assert not window.isWindowModified()
        viewer.inspector.x_field.setText("not a number")
        assert not window._export_peaks()
        assert window.isWindowModified()
        viewer.inspector.x_field.setText("8.123456789012345")
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "selectedFiles",
            lambda _dialog: [str(tmp_path / "missing" / "x.csv")],
        )
        assert not window._export_peaks()
        assert window.isWindowModified()
        assert load_peaks(path, viewer.spectra)[0].positions == restored[0].positions
    finally:
        viewer.edits.mark_saved()
        viewer.inspector.show_peak(None)
        window.close()
        window.deleteLater()
        app.processEvents()


def test_add_peak_lifecycle_and_editing_sequence(tmp_path: Path, monkeypatch) -> None:  # noqa: PLR0915
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = PeakFitWindow()
    window.open_spectrum(DATA / "pseudo3d.ft2", DATA / "pseudo3d.list")
    viewer = window.viewer
    assert viewer is not None
    window.show()
    viewer.spectrum_plot.getViewBox().setRange(xRange=(7.8, 8.8), yRange=(122, 128), padding=0)
    viewer.open_details(0)
    app.processEvents()
    viewport = viewer.spectrum_plot.viewport()
    add = next(
        b for b in viewer.inspector.findChildren(QtWidgets.QPushButton) if b.text() == "Add peak…"
    )
    callback_errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *error: callback_errors.append(error))

    def location(x, y):
        return viewer.spectrum_plot.mapFromScene(
            viewer.spectrum_plot.getViewBox().mapViewToScene(QtCore.QPointF(x, y))
        )

    try:
        for index in range(2):
            QtTest.QTest.mouseClick(add, QtCore.Qt.MouseButton.LeftButton)
            assert viewer.adding_peak
            assert not viewer.inspector.name.isEnabled()
            QtTest.QTest.mouseClick(
                viewport, QtCore.Qt.MouseButton.LeftButton, pos=location(8.4 + index * 0.1, 125.4)
            )
            assert len(viewer.peaks) == 167 + index
            assert len(viewer.peak_labels) == len(viewer.peaks)
            assert len(viewer.peak_scatter.points()) == len(viewer.peaks)
            assert not callback_errors
            assert not viewer.adding_peak
            assert viewer.selected_id == viewer.peaks[-1].id
            assert viewer.inspector.name.isEnabled()
            assert not add.isChecked()
        # Placement is visible and cancelable without adding a record or changing graphics.
        QtTest.QTest.mouseClick(add, QtCore.Qt.MouseButton.LeftButton)
        assert add.isChecked()
        assert add.text() == "Cancel placement"
        assert "PLACING PEAK" in viewer.spectrum_plot.toolTip()
        QtTest.QTest.mouseClick(add, QtCore.Qt.MouseButton.LeftButton)
        assert not viewer.adding_peak
        QtTest.QTest.mouseClick(add, QtCore.Qt.MouseButton.LeftButton)
        QtTest.QTest.keyClick(viewer.spectrum_plot, QtCore.Qt.Key.Key_Escape)
        assert not viewer.adding_peak
        assert len(viewer.peaks) == 168
        assert len(viewer.peak_labels) == 168
        added = viewer.peaks[-1]
        # Select with a real click after first clicking empty background.
        QtTest.QTest.mouseClick(
            viewport, QtCore.Qt.MouseButton.LeftButton, pos=location(8.0, 127.6)
        )
        assert viewer.selected_id is None
        start = location(added.positions[1], added.positions[0])
        QtTest.QTest.mouseClick(viewport, QtCore.Qt.MouseButton.LeftButton, pos=start)
        assert viewer.selected_id == added.id
        QtTest.QTest.mouseMove(viewport, start)
        QtTest.QTest.qWait(30)
        QtTest.QTest.mousePress(viewport, QtCore.Qt.MouseButton.LeftButton, pos=start)
        end = start + QtCore.QPoint(30, 15)
        QtWidgets.QApplication.sendEvent(
            viewport,
            QtGui.QMouseEvent(
                QtCore.QEvent.Type.MouseMove,
                QtCore.QPointF(end),
                QtCore.QPointF(viewport.mapToGlobal(end)),
                QtCore.Qt.MouseButton.NoButton,
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
            ),
        )
        assert viewer.peaks[-1] == added  # preview is not a committed edit
        QtTest.QTest.mouseRelease(viewport, QtCore.Qt.MouseButton.LeftButton, pos=end)
        moved = viewer.peaks[-1]
        assert moved.positions != added.positions
        QtTest.QTest.mouseClick(viewer.inspector.name, QtCore.Qt.MouseButton.LeftButton)
        viewer.inspector.name.selectAll()
        QtTest.QTest.keyClicks(viewer.inspector.name, "Native sequence")
        QtTest.QTest.mouseClick(viewer.inspector.apply, QtCore.Qt.MouseButton.LeftButton)
        assert viewer.peaks[-1].name == "Native sequence"
        QtTest.QTest.keySequence(
            viewer.spectrum_plot, QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Undo)
        )
        assert viewer.peaks[-1] == moved  # undo rename, keep the completed move
        path = tmp_path / "manual.csv"
        original_exec = QtWidgets.QFileDialog.exec

        def export_dialog(dialog):
            # Exercise the save dialog with Qt events; native OS dialog appearance is manual QA.
            dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)
            dialog.setDirectory(str(tmp_path))

            def enter_filename():
                field = dialog.findChild(QtWidgets.QLineEdit, "fileNameEdit")
                assert field is not None
                QtTest.QTest.mouseClick(field, QtCore.Qt.MouseButton.LeftButton)
                field.selectAll()
                QtTest.QTest.keyClicks(field, path.name)
                assert field.text() == path.name
                assert dialog.selectedFiles() == [str(path)]
                QtTest.QTest.keyClick(field, QtCore.Qt.Key.Key_Tab)
                buttons = dialog.findChild(QtWidgets.QDialogButtonBox)
                assert buttons is not None
                save = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Save)
                assert save is not None
                QtTest.QTest.mouseClick(save, QtCore.Qt.MouseButton.LeftButton)

            QtCore.QTimer.singleShot(0, enter_filename)
            QtCore.QTimer.singleShot(3000, dialog.reject)
            return original_exec(dialog)

        monkeypatch.setattr(QtWidgets.QFileDialog, "exec", export_dialog)
        QtTest.QTest.keySequence(window, QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Save))
        assert not callback_errors
        restored = load_peaks(path, viewer.spectra)
        assert [(p.name, p.positions) for p in restored] == [
            (p.name, p.positions) for p in viewer.peaks
        ]
        assert not window.isWindowModified()
        assert len(viewer.peak_labels) == len(viewer.peak_scatter.points()) == 168
        assert not callback_errors
    finally:
        viewer.edits.mark_saved()
        viewer.inspector.show_peak(None)
        window.close()
        window.deleteLater()
        app.processEvents()
