"""The shared spectral view: native slices, scales, lanes, dividers and saved state."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from builders import two_cluster_dataset
from gui_helpers import wait_until
from PySide6 import QtCore, QtGui, QtTest

import peakfit.gui.editor.workspace as workspace_module
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.editor.spectral_geometry import SLICE_COLLAPSE, SliceSizes
from peakfit.gui.editor.spectral_view import LANE_SIGMA
from peakfit.gui.editor.workspace import EditorWorkspace
from peakfit.gui.spectrum import Dataset

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def scientific():
    spectra, peaks, config, prepared = two_cluster_dataset()
    result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    return spectra, peaks, config, result


def _workspace(qapp, scientific):
    spectra, peaks, config, _ = scientific
    dataset = Dataset(Path("synthetic.ft2"), spectra, peaks, (), True, 5.0, 0.01)
    widget = EditorWorkspace(dataset)
    widget.fitting.configuration = config.model_copy(deep=True)
    widget.fitting.setup_seen = True
    widget.resize(1280, 750)
    widget.show()
    qapp.processEvents()
    return widget


@pytest.fixture
def fitted(qapp, scientific):
    widget = _workspace(qapp, scientific)
    result = scientific[-1]
    fitting = widget.fitting
    fitting._submitted_revision = widget.edits.revision
    fitting.process.state = "completed"
    fitting._finished(0, result)
    wait_until(lambda: not fitting.busy)
    widget.select_peak(widget.peaks[0].id)
    widget.set_mode(1)
    wait_until(lambda: not widget.canvas.contour_pending)
    yield widget
    widget.stop()
    widget.close()
    widget.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    qapp.processEvents()


def test_slices_show_the_model_plane_and_the_display_residual_at_native_samples(fitted):
    c = fitted.canvas
    model = c._model.values
    noise = c.comparison_result.outcome.noise
    data = c.spectra.data[c.plane]
    np.testing.assert_array_equal(c.model_x.getData()[1], model[c.iy, :])
    np.testing.assert_array_equal(c.model_y.getData()[0], model[:, c.ix])
    # Lines through native samples: no sample markers, no off-grid points.
    np.testing.assert_array_equal(c.curve_x.getData()[0], c.ppm_x)
    assert c.curve_x.opts["symbol"] is None
    assert c.model_x.opts["symbol"] is None
    c.set_lanes(True)
    np.testing.assert_allclose(c.lane_x.getData()[1], (data[c.iy, :] - model[c.iy, :]) / noise)
    np.testing.assert_allclose(c.lane_y.getData()[0], (data[:, c.ix] - model[:, c.ix]) / noise)
    assert "model" in c.readout.text()
    assert "residual" in c.readout.text()


def test_residual_mode_shows_residuals_in_the_slices_and_keeps_the_lane_preference(fitted):
    c = fitted.canvas
    c.set_lanes(True)
    assert c.row_lane.isVisible()
    selected, ranges = fitted.selected_id, c.box.viewRange()
    fitted.set_mode(2)
    wait_until(lambda: not c.contour_pending)
    assert not c.row_lane.isVisible()
    assert not c.column_lane.isVisible()
    assert c.sizes.lanes  # the preference is kept
    noise = c.comparison_result.outcome.noise
    residual = (c.spectra.data[c.plane] - c._model.values) / noise
    np.testing.assert_array_equal(c.curve_x.getData()[1], residual[c.iy, :])
    np.testing.assert_allclose(c.row_box.viewRange()[1], [-LANE_SIGMA, LANE_SIGMA])
    fitted.set_mode(1)
    wait_until(lambda: not c.contour_pending)
    assert c.row_lane.isVisible()
    assert fitted.selected_id == selected
    assert c.box.viewRange() == ranges


def test_a_manual_scale_survives_hover_planes_and_zoom_until_reset(fitted):
    c = fitted.canvas
    region = c.slice_bounds()
    c.scale_intensity(120)  # wheel up: a narrower manual range
    manual = c.slice_bounds()
    assert manual != region
    assert c.manual_scale == manual
    assert "manual" in c.caption.text()
    c.sample(float(c.ppm_x[3]), float(c.ppm_y[4]))
    fitted.set_plane(2)
    (x0, x1), (y0, y1) = c.box.viewRange()
    c.set_ranges((x0, (x0 + x1) / 2), (y0, (y0 + y1) / 2))
    assert c.slice_bounds() == manual
    # Lanes scale on their own.
    c.scale_lanes(-120)
    assert c.lane_sigma == pytest.approx(LANE_SIGMA * 1.25)
    assert c.slice_bounds() == manual
    c.scale_intensity(None)
    assert c.manual_scale is None
    assert "manual" not in c.caption.text()


def test_the_region_scale_follows_deliberate_view_changes_only(fitted):
    c = fitted.canvas
    before = c.slice_bounds()
    c.sample(float(c.ppm_x[1]), float(c.ppm_y[1]))
    fitted.select_peak(fitted.peaks[1].id)  # may shift the view to keep it visible
    assert c.slice_bounds() == before
    (x0, x1), (y0, y1) = c.box.viewRange()
    c.set_ranges((x0, x0 + (x1 - x0) / 8), (y0, y0 + (y1 - y0) / 8))
    low, high = c.slice_bounds()
    assert low <= 0 <= high
    c.full_spectrum()
    assert c.slice_bounds() == before


def test_clipped_data_and_model_are_counted_separately(fitted):
    c = fitted.canvas
    for _ in range(12):
        c.scale_intensity(120)
    c.sample(*c.display(fitted.peaks[0]))
    data, model = c.clips["row"]
    assert data > 0
    assert model > 0
    assert "clipped" in c.caption.text()
    assert "model clipped" in c.caption.text()
    assert len(c.clip_x.data) == data


def _drag(qapp, viewport, start, end):
    QtTest.QTest.mousePress(viewport, QtCore.Qt.MouseButton.LeftButton, pos=start)
    for fraction in (0.25, 0.5, 0.75, 1.0):
        point = start + (end - start) * fraction
        qapp.sendEvent(
            viewport,
            QtGui.QMouseEvent(
                QtCore.QEvent.Type.MouseMove,
                QtCore.QPointF(point),
                QtCore.QPointF(viewport.mapToGlobal(point)),
                QtCore.Qt.MouseButton.NoButton,
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
            ),
        )
        qapp.processEvents()
    QtTest.QTest.mouseRelease(viewport, QtCore.Qt.MouseButton.LeftButton, pos=end)
    qapp.processEvents()


def test_dividers_resize_each_slice_independently_and_collapse(fitted, qapp):
    c = fitted.canvas
    viewport = c.view.viewport()
    changed = []
    c.sizes_changed.connect(lambda: changed.append(True))
    column = c.arrangement.sizes.column

    def center(divider):
        return c.view.mapFromScene(divider.sceneBoundingRect().center())

    start = center(c.row_divider)
    _drag(qapp, viewport, start, start + QtCore.QPoint(0, 40))
    assert c.sizes.row == SliceSizes().row + 40
    assert c.sizes.column == column
    start = center(c.column_divider)
    _drag(qapp, viewport, start, start + QtCore.QPoint(-30, 0))
    assert c.sizes.column == column + 30
    start = center(c.row_divider)
    _drag(qapp, viewport, start, start - QtCore.QPoint(0, 1000))
    assert c.sizes.row == 0
    assert not c.row_box.isVisible()
    start = center(c.row_divider)
    _drag(qapp, viewport, start, start + QtCore.QPoint(0, SLICE_COLLAPSE + 30))
    assert c.sizes.row == SLICE_COLLAPSE + 30
    assert changed
    # Every placement keeps the shared extents aligned.
    assert c.row_box.sceneBoundingRect().left() == c.box.sceneBoundingRect().left()
    assert c.row_box.sceneBoundingRect().width() == c.box.sceneBoundingRect().width()
    assert c.column_box.sceneBoundingRect().top() == c.box.sceneBoundingRect().top()
    assert c.column_box.sceneBoundingRect().height() == c.box.sceneBoundingRect().height()


def test_slice_sizes_lanes_and_scales_are_saved_and_restored(fitted, qapp, scientific, monkeypatch):
    # The synthetic spectrum has no source file; its header is not what this checks.
    monkeypatch.setattr(workspace_module, "dataset_header", lambda _dataset: {})
    c = fitted.canvas
    c.set_sizes(SliceSizes(row=120, column=90, lanes=True))
    c.scale_lanes(120)
    c.scale_intensity(120)
    saved = fitted.capture()
    state = c.slice_state()
    assert saved.view["slices"] == state
    json.dumps(saved.view, allow_nan=False)  # what the project writer stores
    other = _workspace(qapp, scientific)
    try:
        other.restore(saved)
        qapp.processEvents()
        assert other.canvas.slice_state() == state
        assert other.canvas.slice_bounds() == c.slice_bounds()
    finally:
        other.stop()
        other.close()
        other.deleteLater()
