"""Workspace interactions preserve scientific state and require explicit edits."""

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from test_gui_fit_execution import _example, _wait

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
