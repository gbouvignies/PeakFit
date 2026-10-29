"""Scientific and interaction protection for the production open/inspect/edit slice."""

from pathlib import Path
from threading import Event
from time import perf_counter

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from peakfit.gui.editor.loading import read_assignments, read_dataset
from peakfit.gui.editor.window import EditorWindow
from peakfit.gui.editor.workspace import EditorWorkspace
from peakfit.gui.peaks import PeakEdits, WorkingPeak, export_peaks
from peakfit.gui.spectrum import display_position

DATA = Path(__file__).resolve().parents[1] / "examples/02-advanced-fitting/data"
pytestmark = pytest.mark.gui


def wait_for(condition, timeout=8):
    started = perf_counter()
    while not condition() and perf_counter() - started < timeout:
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(10, loop.quit)
        loop.exec()
    assert condition(), "Timed out waiting for editor delivery"


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="module")
def dataset():
    return read_dataset(DATA / "pseudo3d.ft2", DATA / "pseudo3d.list", DATA / "b1_offsets.txt")


@pytest.fixture
def workspace(app, dataset):
    widget = EditorWorkspace(dataset)
    widget.resize(1280, 750)
    widget.show()
    app.processEvents()
    wait_for(lambda: not widget.canvas.contour_pending)
    yield widget
    widget.stop()
    widget.close()
    widget.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    app.processEvents()


def test_native_arrays_plane_order_alignment_and_resize(workspace, app):
    canvas = workspace.canvas
    assert not canvas.spectra.data.flags.writeable
    assert not canvas.spectra.z_values.flags.writeable
    initial_range = canvas.trace_limit
    workspace.select_peak(workspace.peaks[3].id)
    for width, height in ((1280, 750), (1440, 850), (1040, 670)):
        workspace.resize(width, height)
        app.processEvents()
        canvas.main.getViewBox().setRange(xRange=(7.5, 9), yRange=(115, 125), padding=0)
        app.processEvents()
        for plane in (1, 60, 131):
            workspace.set_plane(plane)
            ix, iy = canvas.ix, canvas.iy
            np.testing.assert_array_equal(
                canvas.curve_x.getData()[1], canvas.spectra.data[plane - 1, iy, :]
            )
            np.testing.assert_array_equal(
                canvas.curve_y.getData()[0], canvas.spectra.data[plane - 1, :, ix]
            )
            assert canvas.trace_limit == initial_range
            np.testing.assert_allclose(
                canvas.top.getViewBox().viewRange()[1], [-initial_range, initial_range]
            )
            np.testing.assert_allclose(
                canvas.right.getViewBox().viewRange()[0], [-initial_range, initial_range]
            )
        main = canvas.main.getViewBox()
        top = canvas.top.getViewBox()
        right = canvas.right.getViewBox()
        for x, y in ((7.75, 118), (8.5, 122)):
            mp = canvas.main.mapToGlobal(
                canvas.main.mapFromScene(main.mapViewToScene(QtCore.QPointF(x, y)))
            )
            tp = canvas.top.mapToGlobal(
                canvas.top.mapFromScene(top.mapViewToScene(QtCore.QPointF(x, 0)))
            )
            rp = canvas.right.mapToGlobal(
                canvas.right.mapFromScene(right.mapViewToScene(QtCore.QPointF(0, y)))
            )
            assert abs(mp.x() - tp.x()) <= 1
            assert abs(mp.y() - rp.y()) <= 1
    assert canvas.main.getViewBox().state["xInverted"]
    assert canvas.main.getViewBox().state["yInverted"]


def test_hover_preview_pinning_and_draft_are_independent(workspace):
    first, second = workspace.peaks[:2]
    workspace.select_peak(first.id)
    original_pin = (workspace.canvas.ix, workspace.canvas.iy)
    workspace.inspector.x_field.setText("8.123456789012345")
    workspace.canvas.hover(second.id, from_list=True)
    assert workspace.selected_id == first.id
    assert workspace.inspector.x_field.text() == "8.123456789012345"
    assert (workspace.canvas.ix, workspace.canvas.iy) == original_pin
    assert workspace.edits.revision == 0
    workspace.canvas.set_pinned(False)
    workspace.canvas.hover(second.id, from_list=True)
    assert workspace.selected_id == first.id
    assert workspace.canvas.ix == round(workspace.canvas.x_axis.ppm2pts(second.positions[1]))
    assert workspace.inspector.pending
    workspace.inspector.show_peak(first)
    assert not workspace.inspector.pending
    workspace.rail.search.setText("nonexistent assignment")
    assert workspace.selected_id == first.id
    assert workspace.rail.reveal.isVisible()


def test_group_edit_keeps_untouched_precision_and_invalid_draft(workspace, app):
    peak = WorkingPeak.new("Exact", (117.12345678901234, 8.234567890123456))
    workspace.edits = PeakEdits((peak, workspace.peaks[0]))
    workspace._refresh()
    workspace.select_peak(peak.id)
    editor = workspace.inspector
    assert editor.x_field.text() == "8.23456789"
    editor.name.setText("Renamed")
    assert workspace.commit_fields()
    assert workspace.peaks[0].positions == peak.positions
    assert workspace.edits.revision == 1
    workspace.history(False)
    assert workspace.peaks[0] == peak
    assert workspace.edits.revision == 2
    workspace.select_peak(peak.id)
    editor.x_field.setFocus()
    app.processEvents()
    assert editor.x_field.text() == repr(peak.positions[1])
    assert not editor.pending
    editor.x_field.setText("8.345678901234567e0")
    QtTest.QTest.keyClick(editor.x_field, QtCore.Qt.Key.Key_Tab)
    assert editor.pending
    assert workspace.peaks[0] == peak
    QtTest.QTest.keyClick(editor.y_field, QtCore.Qt.Key.Key_Return)
    assert workspace.peaks[0].positions == (peak.positions[0], 8.345678901234567)
    assert workspace.edits.revision == 3
    editor.x_field.setText("nan")
    workspace.select_peak(workspace.peaks[1].id)
    assert workspace.selected_id == peak.id
    assert editor.x_field.text() == "nan"
    assert workspace.error.isVisible()
    workspace.set_plane(60)  # Invalid text never traps scientific navigation.
    assert workspace.canvas.plane == 59
    QtTest.QTest.keyClick(editor.x_field, QtCore.Qt.Key.Key_Escape)
    assert not editor.pending
    assert workspace.edits.revision == 3


def test_continuous_drag_ghost_cancel_and_single_commit(workspace):
    peak = workspace.peaks[3]
    workspace.select_peak(peak.id)
    canvas = workspace.canvas
    start_pin = (canvas.ix, canvas.iy, canvas.pinned, canvas.pin_peak_id)
    jobs = canvas.contour_jobs
    markers = canvas.marker_rebuilds
    canvas._move_peak(0.000123456789, 0.000987654321, False)
    assert workspace.peaks[3] == peak
    assert workspace.edits.revision == 0
    assert canvas._drag_peak == peak
    assert len(canvas.ghost.points()) == 1
    assert canvas.marker_rebuilds == markers
    canvas.cancel_gesture()
    assert (canvas.ix, canvas.iy, canvas.pinned, canvas.pin_peak_id) == start_pin
    assert workspace.edits.revision == 0
    canvas._move_peak(0.000123456789, 0.000987654321, False)
    canvas._move_peak(0.000123456789, 0.000987654321, True)
    moved = workspace.peaks[3]
    assert moved.positions == (
        peak.positions[0] + 0.000987654321,
        peak.positions[1] + 0.000123456789,
    )
    assert workspace.edits.revision == 1
    assert canvas.contour_jobs == jobs
    workspace.history(False)
    assert workspace.peaks[3] == peak
    assert workspace.edits.revision == 2
    assert not workspace.edits.can_undo


def test_free_placement_delete_and_exact_export_roundtrip(workspace, tmp_path):
    original = workspace.peaks
    position = (117.00000000000003, 8.000000000000002)
    workspace.start_placing()
    workspace.place_peak(position)
    placed = workspace.peaks[-1]
    assert placed.positions == position
    assert placed.id == workspace.selected_id
    assert not workspace.canvas.placing
    assert workspace.edits.revision == 1
    path = tmp_path / "peaks.csv"
    export_peaks(path, workspace.peaks)
    loaded = read_assignments(path, workspace.dataset.spectra)
    assert [(p.name, p.positions) for p in loaded] == [
        (p.name, p.positions) for p in workspace.peaks
    ]
    workspace.inspector.x_field.setText("invalid")
    workspace.delete_peaks()
    assert workspace.peaks == original
    workspace.history(False)
    assert workspace.peaks[-1] == placed
    assert workspace.edits.revision == 3


def test_keyboard_sample_text_and_hidden_rail(workspace, app):
    workspace.select_peak(workspace.peaks[0].id)
    workspace.canvas.main.setFocus()
    ix = workspace.canvas.ix
    QtTest.QTest.keyClick(workspace.canvas.main, QtCore.Qt.Key.Key_Right)
    assert workspace.canvas.ix == ix + 1
    QtTest.QTest.keyClick(workspace.canvas.main, QtCore.Qt.Key.Key_Space)
    assert not workspace.canvas.pinned
    assert workspace.selected_id == workspace.peaks[0].id
    workspace.inspector.name.setFocus()
    workspace.inspector.name.setCursorPosition(len(workspace.inspector.name.text()))
    QtTest.QTest.keyClick(workspace.inspector.name, QtCore.Qt.Key.Key_Space)
    assert workspace.inspector.name.text().endswith(" ")
    assert not workspace.canvas.pinned
    QtTest.QTest.keyClick(workspace.inspector.name, QtCore.Qt.Key.Key_Escape)
    workspace.peaks_button.setChecked(False)
    workspace.toggle_rail()
    assert workspace.rail.isHidden()
    workspace.select_peak(workspace.peaks[1].id)
    assert workspace.rail.isHidden()
    workspace.resize(1440, 850)
    app.processEvents()
    assert workspace.rail.isHidden()


def test_preview_does_no_contour_or_marker_work_and_cache_is_bounded(workspace):
    canvas = workspace.canvas
    jobs, markers = canvas.contour_jobs, canvas.marker_rebuilds
    limit = canvas.trace_limit
    for i in range(500):
        canvas.sample(
            float(canvas.ppm_x[i % len(canvas.ppm_x)]), float(canvas.ppm_y[i % len(canvas.ppm_y)])
        )
    assert canvas.contour_jobs == jobs
    assert canvas.marker_rebuilds == markers
    assert canvas.trace_limit == limit
    for plane in range(1, 13):
        workspace.set_plane(plane)
        wait_for(lambda: not canvas.contour_pending)
    assert canvas.cache_bytes <= 32 * 1024**2
    assert len(canvas._cache) <= 8
    workspace.set_plane(60)
    workspace.set_plane(131)
    workspace.set_plane(1)
    wait_for(lambda: not canvas.contour_pending)
    assert canvas.plane == 0
    np.testing.assert_array_equal(canvas.curve_x.getData()[1], canvas.spectra.data[0, canvas.iy, :])
    assert len(canvas.paths) == 2


def test_theme_preserves_science_and_geometry(app, dataset):
    window = EditorWindow()
    window.install_dataset(dataset)
    window.show()
    workspace = window.workspace
    assert workspace is not None
    wait_for(lambda: not workspace.canvas.contour_pending)
    workspace.select_peak(dataset.peaks[3].id)
    workspace.canvas.main.getViewBox().setRange(xRange=(7.5, 9), yRange=(115, 120), padding=0)
    app.processEvents()
    before = workspace.canvas.main.getViewBox().viewRange()
    paths = [p.path() for p in workspace.canvas.paths]
    jobs = workspace.canvas.contour_jobs
    for appearance in ("Dark", "Light", "Dark", "System"):
        window.apply_appearance(appearance)
        app.processEvents()
        assert workspace.edits.revision == 0
        assert workspace.selected_id == dataset.peaks[3].id
        assert workspace.canvas.main.getViewBox().viewRange() == before
        assert [p.path() for p in workspace.canvas.paths] == paths
        assert workspace.canvas.contour_jobs == jobs
    window.close()


def test_imported_plane_values_preserve_duplicates_and_axis_mapping(dataset, tmp_path):
    values = tmp_path / "values.txt"
    original = np.arange(131, dtype=float)[::-1]
    original[10] = original[9]
    np.savetxt(values, original)
    loaded = read_dataset(DATA / "pseudo3d.ft2", DATA / "pseudo3d.list", values)
    np.testing.assert_array_equal(loaded.spectra.z_values, original)
    swapped = read_assignments(DATA / "pseudo3d.list", dataset.spectra, swap=True)
    assert swapped[0].positions == tuple(reversed(dataset.peaks[0].positions))
    y_axis, x_axis = dataset.spectra.spectral_params
    indirect = y_axis.pts2ppm(y_axis.size + 3.123456789)
    assert display_position(y_axis, indirect) == pytest.approx(y_axis.pts2ppm(3.123456789))
    direct = x_axis.pts2ppm(x_axis.size + 3.123456789)
    assert display_position(x_axis, direct) == direct


def test_async_cancel_failure_and_atomic_replacement(app, dataset, monkeypatch):
    app.processEvents()
    window = EditorWindow()
    window.install_dataset(dataset)
    previous = window.workspace
    entered, release = Event(), Event()

    def slow_read():
        entered.set()
        release.wait(2)
        return dataset

    window._start_loading(slow_read)
    wait_for(entered.is_set)
    window.cancel_loading()
    assert window.workspace is previous
    release.set()
    QtTest.QTest.qWait(50)
    assert window.workspace is previous
    errors = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: errors.append(args))
    window.load(Path("/missing/invalid.ft2"))
    wait_for(lambda: not window._busy)
    assert window.workspace is previous
    assert errors
    assert "invalid.ft2" in str(errors)
    monkeypatch.setattr(window, "confirm_dataset", lambda _: True)
    window.load(dataset.path)
    wait_for(lambda: window.workspace is not previous)
    assert window.workspace is not None
    assert window.workspace.dataset.spectra.data.shape == dataset.spectra.data.shape
    assert not window.workspace.peaks  # Optional peaks never leak from previous source.
    window.close()


def test_manual_revision_is_monotonic_and_noop_edits_preserve_provenance(dataset):
    edits = PeakEdits(dataset.peaks)
    peak = edits.peaks[0]
    submitted = edits.peaks
    edits.update(peak.id, peak.name, peak.positions)
    assert edits.revision == 0
    edits.update(peak.id, peak.name, (peak.positions[0] + 1e-9, peak.positions[1]))
    assert submitted[0] == peak
    assert edits.revision == 1
    edits.undo()
    assert edits.peaks == submitted
    assert edits.revision == 2  # Exact equality cannot silently recertify a previous fit.
    edits.redo()
    assert edits.revision == 3
    edits.mark_saved()
    assert edits.revision == 3
    assert not edits.dirty
    before = edits.peaks
    with pytest.raises(ValueError, match="unique"):
        edits.replace_all((peak, peak))
    assert edits.peaks == before
    assert edits.revision == 3


def test_clean_editor_undo_routes_to_peak_history(workspace, app):
    peak = workspace.peaks[0]
    workspace.select_peak(peak.id)
    field = workspace.inspector.x_field
    field.setFocus()
    field.setText(repr(peak.positions[1] + 0.0001))
    QtTest.QTest.keyClick(field, QtCore.Qt.Key.Key_Return)
    assert workspace.edits.revision == 1
    assert not workspace.inspector.pending
    assert field.text() == repr(workspace.peaks[0].positions[1])
    QtTest.QTest.keySequence(field, QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Undo))
    app.processEvents()
    assert workspace.peaks[0] == peak
    assert workspace.edits.revision == 2


def test_real_qt_marker_drag_cancel_and_blank_pan(workspace, app):
    canvas = workspace.canvas
    peak = workspace.peaks[3]
    workspace.select_peak(peak.id)
    view = canvas.main.getViewBox()
    x, y = canvas.display(peak)
    view.setRange(xRange=(x - 0.4, x + 0.4), yRange=(y - 2, y + 2), padding=0)
    app.processEvents()
    start = canvas.main.mapFromScene(view.mapViewToScene(QtCore.QPointF(x, y)))
    end = start + QtCore.QPoint(25, 15)
    viewport = canvas.main.viewport()
    QtTest.QTest.mouseMove(viewport, start)
    QtTest.QTest.qWait(30)
    app.processEvents()
    QtTest.QTest.mousePress(viewport, QtCore.Qt.MouseButton.LeftButton, pos=start)
    move = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseMove,
        QtCore.QPointF(end),
        QtCore.QPointF(viewport.mapToGlobal(end)),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    app.sendEvent(viewport, move)
    app.processEvents()
    assert canvas._drag_peak == peak
    assert workspace.peaks[3] == peak
    QtTest.QTest.keyClick(canvas.main, QtCore.Qt.Key.Key_Escape)
    QtTest.QTest.mouseRelease(viewport, QtCore.Qt.MouseButton.LeftButton, pos=end)
    assert workspace.peaks[3] == peak
    assert workspace.edits.revision == 0
    # A background pan changes only ranges; no release pin or peak history.
    canvas.set_pinned(False)
    start = QtCore.QPoint(viewport.width() - 50, viewport.height() - 90)
    end = start - QtCore.QPoint(30, 30)
    QtTest.QTest.mouseMove(viewport, start)
    QtTest.QTest.qWait(30)
    before_pan = view.viewRange()
    QtTest.QTest.mousePress(viewport, QtCore.Qt.MouseButton.LeftButton, pos=start)
    move = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseMove,
        QtCore.QPointF(end),
        QtCore.QPointF(viewport.mapToGlobal(end)),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    app.sendEvent(viewport, move)
    QtTest.QTest.mouseRelease(viewport, QtCore.Qt.MouseButton.LeftButton, pos=end)
    assert not canvas.pinned
    assert workspace.edits.revision == 0
    assert canvas.selected_id == peak.id
    assert view.viewRange() != before_pan
