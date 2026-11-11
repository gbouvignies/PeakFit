"""The shared spectral view: native slices, scales, lanes, dividers and saved state."""

import json
from copy import deepcopy
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from builders import two_cluster_dataset
from gui_helpers import wait_until
from PySide6 import QtCore, QtGui, QtTest

import peakfit.gui.editor.spectral_view as view_module
import peakfit.gui.editor.workspace as workspace_module
from peakfit.gui import formatting
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.editor import slices
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
    dataset = Dataset(Path("synthetic.ft2"), spectra, peaks, (), True, 0.01)
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


def _own_ranges(c):
    """Each slice's rule: its visible samples over every plane, zero included."""
    rows = slices.index_range(c.y_axis, c.box.viewRange()[1])
    columns = slices.index_range(c.x_axis, c.box.viewRange()[0])
    extremes = slices.plane_extremes(c.spectra.data)
    return (
        slices.intensity_range(extremes, (c.iy, c.iy), columns),
        slices.intensity_range(extremes, rows, (c.ix, c.ix)),
    )


def test_the_gain_survives_hover_planes_and_zoom_until_reset(fitted):
    c = fitted.canvas
    assert c.caption.text() == ""  # nothing to say at the default scale
    c.scale_intensity(120)  # wheel up: magnify both slices
    assert c.gain == pytest.approx(1.25)
    assert "Slices ×1.25" in c.caption.text()
    c.sample(float(c.ppm_x[3]), float(c.ppm_y[4]))
    fitted.set_plane(2)
    (x0, x1), (y0, y1) = c.box.viewRange()
    c.set_ranges((x0, (x0 + x1) / 2), (y0, (y0 + y1) / 2))
    assert c.gain == pytest.approx(1.25)
    expected = [np.divide(bounds, 1.25) for bounds in _own_ranges(c)]
    np.testing.assert_allclose(c.slice_bounds(), expected)
    # Lanes scale on their own.
    c.scale_lanes(-120)
    assert c.lane_sigma == pytest.approx(LANE_SIGMA * 1.25)
    assert c.gain == pytest.approx(1.25)
    c.scale_intensity(None)
    assert c.gain == 1
    assert "Slices ×" not in c.caption.text()


def test_each_slice_scales_to_its_own_visible_samples_over_all_planes(fitted):
    c = fitted.canvas
    c.sample(*c.display(fitted.peaks[0]))
    through_peak = c.slice_bounds()
    assert through_peak == _own_ranges(c)
    np.testing.assert_allclose(c.row_box.viewRange()[1], through_peak[0])
    np.testing.assert_allclose(c.column_box.viewRange()[0], through_peak[1])
    # Planes keep the range, so amplitudes compare from plane to plane.
    fitted.set_plane(2)
    assert c.slice_bounds() == through_peak
    # Hover follows the row and column shown: baseline gets its own, smaller range.
    c.sample(float(c.ppm_x[0]), float(c.ppm_y[0]))
    assert c.slice_bounds() == _own_ranges(c)
    assert c.slice_bounds()[0][1] < through_peak[0][1]
    # So does a deliberate view change.
    (x0, x1), (y0, y1) = c.box.viewRange()
    c.set_ranges((x0, x0 + (x1 - x0) / 8), (y0, y0 + (y1 - y0) / 8))
    assert c.slice_bounds() == _own_ranges(c)
    for low, high in c.slice_bounds():
        assert low <= 0 <= high


def test_clipped_data_and_model_are_counted_separately(fitted):
    c = fitted.canvas
    for _ in range(12):
        c.scale_intensity(120)
    c.sample(*c.display(fitted.peaks[0]))
    data, model = c.clips["row"]
    assert data > 0
    assert model > 0
    # The caption counts both slices.
    assert f"▲ {sum(d for d, _m in c.clips.values())} clipped" in c.caption.text()
    assert f"△ {sum(m for _d, m in c.clips.values())} model clipped" in c.caption.text()
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


def test_focus_keeps_its_own_slice_sizes_and_save_records_the_overview(fitted, monkeypatch):
    monkeypatch.setattr(workspace_module, "dataset_header", lambda _dataset: {})
    c = fitted.canvas
    overview, ranges = c.sizes, c.box.viewRange()
    fitted.open_focus(fitted.peaks[0].id)
    assert fitted.focus_open
    assert c.sizes.lanes
    c.resize_slice("row", 24)
    focused = c.sizes
    # Saved while in focus: the overview it returns to, and the focus sizes.
    saved = fitted.capture().view
    assert saved["ranges"] == ranges
    assert saved["slices"]["lanes"] == overview.lanes
    assert saved["focus_slices"]["row"] == focused.row
    fitted.close_focus()
    assert c.sizes == overview
    assert c.box.viewRange() == ranges
    fitted.open_focus(fitted.peaks[0].id)
    assert c.sizes == focused


def _at(c, item):
    return c.view.mapFromScene(item.sceneBoundingRect().center())


def test_double_clicks_reset_scales_and_sizes_with_real_events(fitted, qapp):
    c = fitted.canvas
    c.set_lanes(True)
    qapp.processEvents()
    viewport = c.view.viewport()
    c.scale_intensity(120)
    c.scale_lanes(120)
    QtTest.QTest.mouseDClick(viewport, QtCore.Qt.MouseButton.LeftButton, pos=_at(c, c.row_box))
    qapp.processEvents()
    assert c.gain == 1
    QtTest.QTest.mouseDClick(viewport, QtCore.Qt.MouseButton.LeftButton, pos=_at(c, c.row_lane))
    qapp.processEvents()
    assert c.lane_sigma == LANE_SIGMA
    c.set_sizes(SliceSizes(row=200, column=130, lanes=True))
    qapp.processEvents()
    QtTest.QTest.mouseDClick(viewport, QtCore.Qt.MouseButton.LeftButton, pos=_at(c, c.row_divider))
    qapp.processEvents()
    assert c.sizes.row == SliceSizes().row


def test_a_wheel_zoom_rescales_the_slices_to_the_new_view(fitted, qapp):
    c = fitted.canvas
    viewport = c.view.viewport()
    before = c.box.viewRange()
    point = _at(c, c.box)
    for _ in range(8):
        QtTest.QTest.qWait(10)
        qapp.sendEvent(
            viewport,
            QtGui.QWheelEvent(
                QtCore.QPointF(point),
                QtCore.QPointF(viewport.mapToGlobal(point)),
                QtCore.QPoint(),
                QtCore.QPoint(0, 120),
                QtCore.Qt.MouseButton.NoButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
                QtCore.Qt.ScrollPhase.NoScrollPhase,
                False,
            ),
        )
    assert c.box.viewRange() != before
    wait_until(lambda: not c._commit.isActive())
    assert c.slice_bounds() == _own_ranges(c)


def test_the_readout_shares_one_power_of_ten_and_the_status_bar_stays_quiet(fitted, qapp):
    c = fitted.canvas
    status = []
    c.status_changed.connect(status.append)
    c.sample(float(c.ppm_x[5]), float(c.ppm_y[6]))
    # One number style: intensity and model share the larger one's power of ten.
    value = float(c.spectra.data[c.plane, c.iy, c.ix])
    model = float(c._model.values[c.iy, c.ix]) if c._model is not None else 0.0
    power = formatting.exponent(max(abs(value), abs(model)))
    expected = f"{value * 10.0**-power:.6g}"
    assert f"I {expected}" in c.readout.text()
    assert "e+" not in c.readout.text()
    assert status[-1] == ""  # the readout shows it; the status bar does not repeat it
    assert "a.u." in c.view.accessibleDescription()
    # Without room for the readout, the status bar carries the sample instead.
    c.set_sizes(SliceSizes(row=150, column=0))
    qapp.processEvents()
    assert not c.readout.isVisible()
    c.sample(float(c.ppm_x[7]), float(c.ppm_y[6]))
    assert "ppm" in status[-1]


def test_run_identity_is_not_repeated_where_the_toolbar_already_names_it(fitted):
    run = fitted.fitting.result.run_id[:8]
    assert run in fitted.summary.text()
    assert run not in fitted.panel.status.text()
    assert run.upper() not in fitted.panel.fitted_title.text()
    fitted.open_focus(fitted.peaks[0].id)
    bar = fitted.focus_bar
    assert run not in bar.note.text()
    assert bar.note.text() == ""  # nothing differs from the toolbar for a current run
    # The run stays one hover away on the always-visible title.
    assert bar.title.isVisible()
    assert bar.title.toolTip() == f"Run {run} · Current"


def _shown_planes(c):
    """The plane every visible contour layer belongs to."""
    return {layer.key[2] for layer in c.paths}


def test_plane_steps_swap_measured_contours_without_a_gap_or_a_stale_layer(fitted):
    c = fitted.canvas
    assert c.comparison_mode == 1
    wait_until(lambda: not c.contour_pending)
    # The neighbouring plane's measured contours are prepared after the current ones.
    wait_until(lambda: c._experiment_key(c.plane + 1) in c._cache)
    fitted.set_plane(c.plane + 2)  # one-based: the next plane
    # Shown at once, in the same step that removed the previous plane's layers.
    assert [layer.role for layer in c.paths] == ["measured"]
    assert _shown_planes(c) == {c.plane}
    assert c._model is None or c.model_layers
    assert c.pending_label.toPlainText() in ("", "Experimental contours · model pending")
    wait_until(lambda: not c.contour_pending)
    assert [layer.role for layer in c.paths] == ["measured", "model"]
    assert _shown_planes(c) == {c.plane}
    assert not c.pending_label.isVisible()


def test_late_deliveries_never_show_on_another_plane(fitted):
    c = fitted.canvas
    wait_until(lambda: not c.contour_pending)
    old_plane, old_revision = c.plane, c._revision
    old_key = c._experiment_key()
    old_geometry = c._cache[old_key]
    old_model = c._model
    fitted.set_plane(old_plane + 2)
    # A model computed for the previous request is dropped; measured contours of the
    # previous plane are kept in the cache but not shown here.
    c._contours_ready("model", old_revision, (old_model, old_geometry))
    c._contours_ready("contours", c._revision, (old_key, old_geometry))
    assert _shown_planes(c) <= {c.plane}
    assert all(layer.role != "model" or layer.key[2] == c.plane for layer in c.paths)
    assert old_key in c._cache
    wait_until(lambda: not c.contour_pending)
    assert _shown_planes(c) == {c.plane}


def test_residual_and_mode_changes_keep_the_measured_contours_of_the_plane(fitted):
    c = fitted.canvas
    wait_until(lambda: not c.contour_pending)
    measured = c.experiment_layers[0]
    # Changing mode keeps the measured layer itself: nothing is redrawn or blinks.
    for mode in (2, 0, 1, 2):
        fitted.set_mode(mode)
        assert c.experiment_layers[0] is measured
    wait_until(lambda: not c.contour_pending)
    # Residual shows only the residual once it is in.
    assert [layer.role for layer in c.paths] == ["residual"]
    fitted.set_plane(c.plane + 2)
    # A new plane in Residual: its measured contours until its residual is ready.
    assert all(layer.key[2] == c.plane for layer in c.paths)
    if c.contour_pending:
        assert c.pending_label.toPlainText().startswith("Experimental contours · residual")
    wait_until(lambda: not c.contour_pending)
    assert [layer.role for layer in c.paths] == ["residual"]
    assert _shown_planes(c) == {c.plane}


def test_the_pending_label_waits_before_appearing_and_says_what_is_coming(fitted, monkeypatch):
    c = fitted.canvas
    wait_until(lambda: not c.contour_pending)
    release = Event()
    project = view_module.dataset_model_plane

    def held(*args, **kwargs):
        assert release.wait(5)
        return project(*args, **kwargs)

    monkeypatch.setattr(view_module, "dataset_model_plane", held)
    try:
        fitted.set_plane(c.plane + 2)
        # Said at once (and to assistive technology), shown only if the wait lasts.
        assert c.pending_label.toPlainText() == "Experimental contours · model pending"
        assert not c.pending_label.isVisible()
        wait_until(c.pending_label.isVisible, timeout=2)
        assert [layer.role for layer in c.paths] == ["measured"]
    finally:
        release.set()
    wait_until(lambda: not c.contour_pending)
    assert not c.pending_label.isVisible()


def test_measured_contours_take_their_own_pens_when_the_model_goes(fitted):
    c = fitted.canvas
    wait_until(lambda: not c.contour_pending)
    measured = c.experiment_layers[0]
    under_model = measured._pens[0].widthF()
    fitted.set_mode(0)
    # Same layer, restyled: under a model measured strokes widen; alone they do not.
    assert c.experiment_layers[0] is measured
    assert measured._pens[0].widthF() == pytest.approx(under_model / 1.5)
    fitted.set_mode(1)
    wait_until(lambda: not c.contour_pending)
    assert measured._pens[0].widthF() == pytest.approx(under_model)


def _evict(c, key):
    geometry = c._cache.pop(key)
    c.cache_bytes -= sum(p.nbytes + q.nbytes for p, q in geometry)


def _hold_measured(monkeypatch, c):
    """Block measured-contour geometry (of spectrum data) until released."""
    release, entered = Event(), Event()
    original = view_module.signed_geometry

    def held(data, axes, levels):
        if np.shares_memory(data, c.spectra.data):
            entered.set()
            assert release.wait(5)
        return original(data, axes, levels)

    monkeypatch.setattr(view_module, "signed_geometry", held)
    return release, entered


def _overlay_complete_or_measured_only(c):
    """Overlay shows measured contours alone, or measured with model, never model alone."""
    roles = [layer.role for layer in c.paths]
    assert roles in ([], ["measured"], ["measured", "model"]), roles
    assert all(layer.key[2] == c.plane for layer in c.paths)
    assert (c._model is not None) == ("model" in roles)


def _leave_and_evict(fitted, c):
    """Settle on plane 0, step to plane 1, and evict plane 0's measured geometry."""
    fitted.set_plane(1)
    wait_until(lambda: not c.contour_pending)
    model_key = c._model_key()
    fitted.set_plane(2)
    wait_until(lambda: not c.contour_pending)
    _evict(c, c._experiment_key(0))
    return model_key


def test_a_cached_model_waits_for_its_planes_measured_contours(fitted, monkeypatch):
    c = fitted.canvas
    model_key = _leave_and_evict(fitted, c)
    # Back on plane 0: its model geometry and model plane are cached, its measured
    # geometry is not, and computing it is held.
    assert model_key in c._cache
    release, entered = _hold_measured(monkeypatch, c)
    try:
        fitted.set_plane(1)
        wait_until(entered.is_set)
        _overlay_complete_or_measured_only(c)
        assert c.paths == []
        assert c._model is None
        assert c.pending_label.toPlainText() == "Preparing contours…"
    finally:
        release.set()
    wait_until(lambda: not c.contour_pending)
    assert [layer.role for layer in c.paths] == ["measured", "model"]
    assert _shown_planes(c) == {0}
    assert c._model is not None


def test_a_model_arriving_before_measured_contours_is_held(fitted, monkeypatch):
    c = fitted.canvas
    model_key = _leave_and_evict(fitted, c)
    geometry = c._cache.pop(model_key)  # nothing cached for plane 0: both computed
    c.cache_bytes -= sum(a.nbytes + b.nbytes for a, b in geometry)
    model = c.model_planes.get(model_key[0], 0)
    release, entered = _hold_measured(monkeypatch, c)
    try:
        fitted.set_plane(1)
        wait_until(entered.is_set)
        # The model for this very request arrives first (as a failed or slow
        # measured job would allow): it is held, not shown alone.
        c._contours_ready("model", c._revision, (model, geometry))
        _overlay_complete_or_measured_only(c)
        assert c.paths == []
        assert c._model is None
    finally:
        release.set()
    wait_until(lambda: not c.contour_pending)
    assert [layer.role for layer in c.paths] == ["measured", "model"]
    assert _shown_planes(c) == {0}
    assert c._model is not None


def test_a_missed_prefetch_on_reversal_shows_nothing_stale_or_partial(fitted, monkeypatch):
    c = fitted.canvas
    fitted.set_plane(2)  # the middle plane; the next one, forward, gets prepared
    wait_until(lambda: not c.contour_pending)
    wait_until(lambda: c._experiment_key(2) in c._cache)
    behind = c._experiment_key(0)
    if behind in c._cache:
        _evict(c, behind)
    release, entered = _hold_measured(monkeypatch, c)
    try:
        fitted.set_plane(1)  # reverse: plane 0 has no prepared measured contours
        wait_until(entered.is_set)
        _overlay_complete_or_measured_only(c)
        assert c.paths == []
        assert c.pending_label.toPlainText() == "Preparing contours…"
    finally:
        release.set()
    wait_until(lambda: not c.contour_pending)
    assert [layer.role for layer in c.paths] == ["measured", "model"]
    assert _shown_planes(c) == {0}
