"""Workspace interactions preserve scientific state and require explicit edits."""

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtTest, QtWidgets
from test_gui_fit_execution import _example, _wait

from peakfit.gui.fit_process import FitProcessResult
from peakfit.gui.local_fit import execute_local_fit
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


@pytest.mark.parametrize("revert", ["button", "escape"])
def test_pending_coordinates_and_revert_preserve_completed_scientific_state(fitted_viewer, revert):
    widget = fitted_viewer
    revision = widget._scientific_revision
    preparation, result = widget.fit_preparation, widget.fit_result
    model = widget.comparison._current
    paths = tuple(widget.comparison.paths)
    peaks = widget.peaks
    field = widget.inspector.x_field
    field.setText(str(float(field.text()) + 0.001))
    assert widget.inspector.pending
    assert "pending" in widget.workspace.edit.text()
    assert not widget.inspector.delete.isEnabled()
    assert widget._scientific_revision == revision
    assert widget.fit_preparation is preparation
    assert widget.comparison.active
    # A blocked preparation request must leave the current comparison intact too.
    widget.preview_fit_group()
    assert widget.comparison._current is model
    if revert == "button":
        widget.inspector.revert.click()
    else:
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
def test_applying_unchanged_coordinates_keeps_comparison(fitted_viewer, equivalent_text):
    widget = fitted_viewer
    revision = widget._scientific_revision
    model = widget.comparison._current
    preparation = widget.fit_preparation
    if equivalent_text:
        widget.inspector.x_field.setText(widget.inspector.x_field.text() + " ")
        assert widget.inspector.pending
    widget.inspector.apply.click()
    assert not widget.inspector.pending
    assert widget._scientific_revision == revision
    assert widget.fit_preparation is preparation
    assert widget.comparison._current is model
    assert widget.comparison.active
    assert not widget.edits.dirty
    assert not widget.edits.can_undo


def test_apply_invalidates_only_after_a_valid_coordinate_change(fitted_viewer):
    widget = fitted_viewer
    revision = widget._scientific_revision
    result = widget.fit_result
    coordinate = widget.peaks[0].positions[1]
    widget.inspector.x_field.setText("nan")
    widget.inspector.apply.click()
    assert widget.inspector.pending
    assert widget._scientific_revision == revision
    assert widget.comparison.active
    widget.inspector.x_field.setText(str(coordinate + 0.001))
    widget.inspector.apply.click()
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
    widget.inspector.y_field.setText(str(float(widget.inspector.y_field.text()) + 0.001))
    assert widget.fit_preparation is prepared
    assert widget._scientific_revision == revision

    def unexpected(*_args):
        pytest.fail("Pending fields must prevent fitting")

    monkeypatch.setattr(widget._fit_process, "start", unexpected)
    widget.run_local_fit()
    widget.preview_fit_group()
    widget.inspector.revert.click()
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


def test_pending_fields_block_hiding_selection_and_implicit_commit(viewer):
    widget, _prepared = viewer
    original = widget.peaks
    widget.workspace.edit.click()
    widget.inspector.name.setText("provisional name")
    widget.workspace.close_button.click()
    assert widget.workspace.details.isVisible()
    widget.workspace.profile.click()
    assert widget.workspace.pages.currentIndex() == 0
    widget.select_peak(original[1].id)
    assert widget.selected_id == original[0].id
    widget.preview_fit_group()
    assert widget.fit_preparation is None
    assert widget.peaks == original
    assert widget.inspector.pending
    assert "pending" in widget.workspace.edit.text()
    widget.inspector.revert.click()
    assert not widget.inspector.pending
    widget.inspector.name.setText("explicitly applied")
    widget.inspector.apply.click()
    assert widget.peaks[0].name == "explicitly applied"
    assert not widget.inspector.pending
    widget.close_details()
    assert widget.workspace.details.isHidden()
    widget.select_peak(original[1].id)
    assert widget.selected_id == original[1].id


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
