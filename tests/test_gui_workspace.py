"""Workspace editing commits complete peaks and preserves scientific state."""

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtTest, QtWidgets
from test_gui_fit_execution import _example, _wait

from peakfit.gui.fit_process import FitProcessResult
from peakfit.gui.local_fit import execute_local_fit
from peakfit.gui.spectrum import display_position
from peakfit.gui.viewer import SpectrumViewer

pytestmark = pytest.mark.gui


@pytest.fixture
def viewer():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    spectra, peaks, prepared, _config = _example()
    widget = SpectrumViewer(spectra)
    widget.set_peaks(peaks)
    widget.select_peak(peaks[0].id)
    widget.show()
    app.processEvents()
    yield widget, prepared
    widget.stop_preparation()
    widget.close()
    widget.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    app.processEvents()


@pytest.fixture
def fitted_viewer(viewer):
    widget, prepared = viewer
    widget.fit_result = FitProcessResult(execute_local_fit(prepared), 0, 0, 0, 0)
    widget._result_revision = widget._scientific_revision
    widget.show_fit_result(activate=False)
    widget.exported_result_id = widget.fit_result.scientific.run_id
    _wait(lambda: widget.comparison._current is not None and widget.comparison._slice is not None)
    widget.open_details(0)
    return widget


def test_pending_coordinates_and_escape_preserve_completed_scientific_state(fitted_viewer):
    widget = fitted_viewer
    revision = widget._scientific_revision
    preparation, result = widget.fit_preparation, widget.fit_result
    model = widget.comparison._current
    paths = tuple(widget.comparison.paths)
    peaks = widget.peaks
    field = widget.inspector.x_field
    field.setFocus()
    field.setText(str(float(field.text()) + 0.001))
    assert widget.inspector.pending
    assert "pending" in widget.workspace.edit.text()
    assert not widget.inspector.delete.isEnabled()
    assert widget._scientific_revision == revision
    assert widget.fit_preparation is preparation
    assert widget.comparison.active
    QtTest.QTest.keyClick(field, QtCore.Qt.Key.Key_Escape)
    assert not widget.inspector.pending
    assert widget.inspector.delete.isEnabled()
    assert widget._scientific_revision == revision
    assert widget.fit_preparation is preparation
    assert widget.fit_result is result
    assert widget.result_exported
    assert widget.comparison._current is model
    assert tuple(widget.comparison.paths) == paths
    assert any(path.isVisible() for path in paths)
    assert widget.peaks == peaks
    assert not widget.edits.can_undo


@pytest.mark.parametrize("equivalent_text", [False, True])
def test_committing_unchanged_coordinates_keeps_comparison(fitted_viewer, equivalent_text):
    widget = fitted_viewer
    revision = widget._scientific_revision
    model = widget.comparison._current
    preparation = widget.fit_preparation
    if equivalent_text:
        widget.inspector.x_field.setText(widget.inspector.x_field.text() + " ")
        assert widget.inspector.pending
    QtTest.QTest.keyClick(widget.inspector.x_field, QtCore.Qt.Key.Key_Return)
    assert not widget.inspector.pending
    assert widget._scientific_revision == revision
    assert widget.fit_preparation is preparation
    assert widget.comparison._current is model
    assert widget.comparison.active
    assert not widget.edits.dirty
    assert not widget.edits.can_undo


def test_enter_invalidates_only_after_a_valid_coordinate_change(fitted_viewer):
    widget = fitted_viewer
    revision = widget._scientific_revision
    result = widget.fit_result
    coordinate = widget.peaks[0].positions[1]
    widget.inspector.x_field.setText("nan")
    QtTest.QTest.keyClick(widget.inspector.x_field, QtCore.Qt.Key.Key_Return)
    assert widget.inspector.pending
    assert widget._scientific_revision == revision
    assert widget.comparison.active
    widget.inspector.x_field.setText(str(coordinate + 0.001))
    QtTest.QTest.keyClick(widget.inspector.x_field, QtCore.Qt.Key.Key_Return)
    assert not widget.inspector.pending
    assert widget._scientific_revision == revision + 1
    assert widget.peaks[0].positions[1] == coordinate + 0.001
    assert widget.edits.can_undo
    assert widget.edits.dirty
    assert widget.fit_preparation is None
    assert widget.fit_result is result
    assert widget.result_exported
    assert not widget.comparison.active
    assert not any(path.isVisible() for path in widget.comparison.paths)


def test_pending_fields_preserve_preparation_and_block_execution(viewer, monkeypatch):
    widget, prepared = viewer
    widget._apply_fit_preparation(prepared)
    revision, scope_revision = widget._scientific_revision, widget._fit_revision
    widget.open_details(0)
    widget.inspector.y_field.setText("nan")
    assert widget.fit_preparation is prepared
    assert widget._scientific_revision == revision

    def unexpected(*_args):
        pytest.fail("Pending fields must prevent fitting")

    monkeypatch.setattr(widget._fit_process, "start", unexpected)
    widget.run_local_fit()
    widget.preview_fit_group()
    QtTest.QTest.keyClick(widget.inspector.y_field, QtCore.Qt.Key.Key_Escape)
    assert widget.fit_preparation is prepared
    assert widget._scientific_revision == revision
    assert widget._fit_revision == scope_revision
    assert widget.fit_mask.isVisible()


def test_noop_history_and_drag_keep_completed_comparison(fitted_viewer):
    widget = fitted_viewer
    revision = widget._scientific_revision
    model = widget.comparison._current
    widget.undo()
    widget.redo()
    widget._preview_peak_move(0, 0, True)
    assert widget._scientific_revision == revision
    assert widget.comparison._current is model
    assert widget.comparison.active
    assert not widget.edits.can_undo


@pytest.mark.parametrize("size", [(1280, 800), (1440, 900)])
@pytest.mark.parametrize("dark", [False, True])
def test_details_and_slices_preserve_context_and_dimensions(viewer, size, dark):
    widget, _prepared = viewer
    app = QtWidgets.QApplication.instance()
    assert app is not None
    palette = widget.palette()
    for role, light, dim in (
        (QtGui.QPalette.ColorRole.Base, "#ffffff", "#202020"),
        (QtGui.QPalette.ColorRole.Text, "#202020", "#eeeeee"),
    ):
        palette.setColor(role, QtGui.QColor(dim if dark else light))
    widget.setPalette(palette)
    widget.resize(*size)
    widget.spectrum_plot.getViewBox().setRange(xRange=(-0.3, 0.3), yRange=(-0.2, 0.2), padding=0)
    widget.set_plane(2)
    app.processEvents()
    context = (widget.plane, widget.ix, widget.iy, widget.selected_id)
    ranges = widget.spectrum_plot.getViewBox().viewRange()
    original = widget.spectra.data.copy()
    assert widget.workspace.details.isHidden()
    assert not widget.profile.isVisible()
    widget.workspace.profile.click()
    app.processEvents()
    assert widget.profile.isVisible()
    assert widget.spectrum_plot.width() >= 560
    assert widget.spectrum_plot.height() >= 300
    assert all(p.isVisible() and p.height() >= 170 for p in (widget.trace_x, widget.trace_y))
    widget.workspace.vertical.setSizes([widget.workspace.vertical.height() - 220, 220])
    app.processEvents()
    expanded = widget.workspace.vertical.sizes()[1]
    widget.workspace.close_button.click()
    app.processEvents()
    widget.workspace.profile.click()
    app.processEvents()
    assert widget.workspace.vertical.sizes()[1] == expanded
    widget.resize(1440, 900)
    app.processEvents()
    widget.resize(1280, 800)
    app.processEvents()
    widget.workspace.slices_action.setChecked(False)
    app.processEvents()
    assert not widget.trace_x.isVisible()
    widget.workspace.slices_action.setChecked(True)
    app.processEvents()
    assert widget.trace_x.isVisible()
    assert widget.trace_y.isVisible()
    assert (widget.plane, widget.ix, widget.iy, widget.selected_id) == context
    np.testing.assert_allclose(widget.spectrum_plot.getViewBox().viewRange(), ranges)
    np.testing.assert_array_equal(widget.spectra.data, original)
    np.testing.assert_array_equal(widget.curve_profile.yData, original[:, widget.iy, widget.ix])
    assert widget.trace_x.backgroundBrush().color().name() == ("#202020" if dark else "#ffffff")


@pytest.mark.parametrize("finish", ["enter", "focus", "select", "marker", "close", "profile"])
def test_complete_peak_is_one_undo_step_when_editing_finishes(  # noqa: PLR0915
    fitted_viewer, finish
):
    widget = fitted_viewer
    app = QtWidgets.QApplication.instance()
    assert app is not None
    original = widget.peaks
    result = widget.fit_result
    revision = widget._scientific_revision
    inspector = widget.inspector
    inspector.name.setFocus()
    inspector.name.selectAll()
    QtTest.QTest.keyClicks(inspector.name, "renamed")
    QtTest.QTest.keyClick(inspector.name, QtCore.Qt.Key.Key_Tab)
    assert inspector.x_field.hasFocus()
    inspector.x_field.selectAll()
    QtTest.QTest.keyClicks(inspector.x_field, "0.123456789012345")
    QtTest.QTest.keyClick(inspector.x_field, QtCore.Qt.Key.Key_Tab)
    assert inspector.y_field.hasFocus()
    inspector.y_field.selectAll()
    QtTest.QTest.keyClicks(inspector.y_field, "1.05")
    QtTest.QTest.keyClick(inspector.y_field, QtCore.Qt.Key.Key_Backtab)
    app.processEvents()
    assert inspector.x_field.hasFocus()
    assert widget.peaks == original
    assert widget._scientific_revision == revision
    assert not widget.edits.can_undo
    assert widget.comparison.active
    if finish == "enter":
        QtTest.QTest.keyClick(inspector.x_field, QtCore.Qt.Key.Key_Return)
    elif finish == "focus":
        widget.spectrum_plot.setFocus()
    elif finish == "select":
        widget.select_peak(original[1].id)
    elif finish == "marker":
        other = original[1]
        position = widget.spectrum_plot.getViewBox().mapViewToScene(
            QtCore.QPointF(
                display_position(widget.x_axis, other.positions[1]),
                display_position(widget.y_axis, other.positions[0]),
            )
        )
        QtTest.QTest.mouseClick(
            widget.spectrum_plot.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            pos=widget.spectrum_plot.mapFromScene(position),
        )
    elif finish == "close":
        QtTest.QTest.mouseClick(widget.workspace.close_button, QtCore.Qt.MouseButton.LeftButton)
        assert widget.workspace.details.isHidden()
    else:
        QtTest.QTest.mouseClick(widget.workspace.profile, QtCore.Qt.MouseButton.LeftButton)
        assert widget.workspace.pages.currentIndex() == 2
    app.processEvents()
    committed = widget.peaks
    assert committed[0].name == "renamed"
    assert committed[0].positions == (1.05, 0.123456789012345)
    assert committed[0].id == original[0].id
    assert committed[1:] == original[1:]
    if finish in ("select", "marker"):
        assert widget.selected_id == original[1].id
    assert not inspector.pending
    assert widget._scientific_revision == revision + 1
    assert widget.fit_result is result
    assert widget.result_exported
    assert not widget.comparison.active
    assert "outdated" in widget.fit_status.text()
    assert "Outdated" in widget.fit_preview.status.text()
    assert not widget.fit_preview.apply_positions.isEnabled()
    widget.undo()
    assert widget.peaks == original
    assert not widget.edits.can_undo
    widget.redo()
    assert widget.peaks == committed
    assert widget.fit_result is result
    assert widget.result_exported
    # Reverting history cannot make a result current again under revision tracking.
    assert widget._result_revision != widget._scientific_revision


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    [
        ("name", " "),
        ("x_field", ""),
        ("x_field", "-"),
        ("x_field", "nan"),
        ("y_field", "inf"),
        ("y_field", "1e999"),
    ],
)
@pytest.mark.parametrize("finish", ["enter", "focus", "select", "close"])
def test_invalid_draft_is_retained_explained_and_revertible(
    fitted_viewer, field_name, invalid, finish
):
    widget = fitted_viewer
    original = widget.peaks
    revision = widget._scientific_revision
    model = widget.comparison._current
    field = getattr(widget.inspector, field_name)
    field.setFocus()
    field.setText(invalid)
    if finish == "enter":
        QtTest.QTest.keyClick(field, QtCore.Qt.Key.Key_Return)
    elif finish == "focus":
        widget.spectrum_plot.setFocus()
    elif finish == "select":
        widget.select_peak(original[1].id)
    else:
        widget.close_details()
    assert widget.workspace.details.isVisible()
    assert widget.workspace.pages.currentIndex() == 0
    assert field.text() == invalid
    assert widget.inspector.pending
    assert "Correct it or press Escape" in widget.inspector.details.text()
    assert field.hasFocus()
    assert widget.peaks == original
    assert widget.selected_id == original[0].id
    assert widget._scientific_revision == revision
    assert widget.comparison._current is model
    assert widget.comparison.active
    assert widget.result_exported
    # Control refreshes must not overwrite the validation error.
    message = widget.inspector.details.text()
    widget._update_edit_actions()
    assert widget.inspector.details.text() == message
    QtTest.QTest.keyClick(field, QtCore.Qt.Key.Key_Escape)
    assert not widget.inspector.pending
    assert widget.peaks == original
    assert widget._scientific_revision == revision


def test_tab_leaves_the_field_group_and_commits_once(viewer):
    widget, _ = viewer
    widget.open_details(0)
    original = widget.peaks
    field = widget.inspector.y_field
    field.setFocus()
    field.setText("0.025")
    QtTest.QTest.keyClick(field, QtCore.Qt.Key.Key_Tab)
    assert not widget.inspector.pending
    assert widget.peaks[0].positions[0] == 0.025
    widget.undo()
    assert widget.peaks == original
    assert not widget.edits.can_undo


def test_tab_preserves_partial_coordinates_until_the_whole_peak_is_valid(viewer):
    widget, prepared = viewer
    widget._apply_fit_preparation(prepared)
    widget.open_details(0)
    original = widget.peaks
    revision = widget._scientific_revision
    inspector = widget.inspector
    inspector.name.setFocus()
    inspector.name.setText("new assignment")
    QtTest.QTest.keyClick(inspector.name, QtCore.Qt.Key.Key_Tab)
    inspector.x_field.setText("-")
    QtTest.QTest.keyClick(inspector.x_field, QtCore.Qt.Key.Key_Tab)
    assert inspector.y_field.hasFocus()
    assert inspector.x_field.text() == "-"
    assert "finite" not in inspector.details.text()
    inspector.y_field.setText("0.025")
    QtTest.QTest.keyClick(inspector.y_field, QtCore.Qt.Key.Key_Return)
    assert "finite" in inspector.details.text()
    assert inspector.x_field.hasFocus()
    assert inspector.name.text() == "new assignment"
    assert inspector.y_field.text() == "0.025"
    assert widget.peaks == original
    assert widget._scientific_revision == revision
    assert widget.fit_preparation is prepared
    inspector.x_field.setText("-0.02")
    QtTest.QTest.keyClick(inspector.x_field, QtCore.Qt.Key.Key_Return)
    assert widget._scientific_revision == revision + 1
    assert widget.peaks[0].name == "new assignment"
    assert widget.peaks[0].positions == (0.025, -0.02)
    widget.undo()
    assert widget.peaks == original
    assert not widget.edits.can_undo


def test_run_cannot_submit_preparation_invalidated_by_committing_fields(viewer, monkeypatch):
    widget, prepared = viewer
    widget._apply_fit_preparation(prepared)
    widget.open_details(0)
    widget.inspector.x_field.setText("0.025")

    def unexpected(*_args):
        pytest.fail("Committing changed coordinates must invalidate the reviewed preparation")

    monkeypatch.setattr(widget._fit_process, "start", unexpected)
    widget.run_local_fit()
    assert widget.peaks[0].positions[1] == 0.025
    assert not widget.inspector.pending
    assert widget.fit_preparation is None
    assert not widget.fit_busy


def test_native_text_context_menu_preserves_draft(viewer):
    widget, _ = viewer
    widget.open_details(0)
    field = widget.inspector.name
    field.setFocus()
    field.setText("pending name")
    revision = widget._scientific_revision
    menu = field.createStandardContextMenu()
    try:
        menu.popup(field.mapToGlobal(QtCore.QPoint()))
        QtWidgets.QApplication.processEvents()
        assert widget.inspector.pending
        assert widget._scientific_revision == revision
        assert not widget.edits.can_undo
    finally:
        menu.close()
        menu.deleteLater()


def test_fit_completion_modes_and_cancellation_never_open_details(viewer, monkeypatch):
    widget, prepared = viewer
    app = QtWidgets.QApplication.instance()
    assert app is not None
    widget._apply_fit_preparation(prepared)
    widget.open_details(1)
    widget.run_local_fit()
    widget.close_details()
    app.processEvents()
    assert widget.cancel_fit.isVisible()
    assert widget.fit_status.text()
    geometry = widget.spectrum_plot.geometry()
    ranges = widget.spectrum_plot.getViewBox().viewRange()
    _wait(lambda: not widget.fit_busy)
    app.processEvents()
    assert widget.workspace.details.isHidden()
    assert not widget.cancel_fit.isVisible()
    assert "completed" in widget.fit_status.text()
    assert widget.spectrum_plot.geometry() == geometry
    result = widget.fit_result
    assert result is not None
    comparison = widget.comparison
    assert comparison.scope.isVisible()
    assert "Local group" in comparison.scope.text()
    for mode in (2, 0, 1):
        comparison.mode.setCurrentIndex(mode)
        assert widget.workspace.details.isHidden()
        np.testing.assert_allclose(widget.spectrum_plot.getViewBox().viewRange(), ranges)
    widget.workspace.profile.click()
    comparison.profile_mode.setCurrentIndex(1)
    evaluation = result.scientific.outcome.clusters[0].analytical_evaluation
    np.testing.assert_array_equal(comparison.profile_model.yData, evaluation.amplitudes[0])
    assert widget.profile.getAxis("left").labelText == "Signed peak amplitude"
    comparison.profile_mode.setCurrentIndex(0)
    assert widget.profile.getAxis("left").labelText == "Coordinate intensity"
    assert widget.trace_x.isVisible()
    assert widget.trace_y.isVisible()
    widget.close_details()
    widget._apply_fit_preparation(prepared)
    monkeypatch.setattr(widget, "confirm_result_loss", lambda _action: True)
    widget.run_local_fit()
    widget.cancel_fit.click()
    _wait(lambda: not widget.fit_busy)
    assert widget.workspace.details.isHidden()
    assert "canceled" in widget.fit_status.text()
    assert widget.fit_result is result


def test_scope_failure_can_be_retried_without_background_panel_changes(viewer, monkeypatch):
    widget, _prepared = viewer

    def fail(*_args, **_kwargs):
        raise ValueError("Cannot prepare this scope")

    with monkeypatch.context() as failure:
        failure.setattr("peakfit.gui.viewer.prepare_local_fit", fail)
        widget.open_details(1)
        widget.close_details()
        _wait(lambda: "Scope unavailable" in widget.fit_preview.status.text())
        assert widget.workspace.details.isHidden()
        widget.open_details(1)
        assert widget.fit_preview.prepare.isVisible()
        assert "Cannot prepare" in widget.fit_preview.diagnostics.toPlainText()
    widget.fit_preview.prepare.click()
    widget.close_details()
    _wait(lambda: widget.fit_preparation is not None)
    assert widget.workspace.details.isHidden()
    assert widget.fit_preview.run.isEnabled()
