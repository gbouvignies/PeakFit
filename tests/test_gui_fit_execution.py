"""Two integration contracts: scientific equivalence and the real spawned lifecycle."""

import multiprocessing
from copy import deepcopy
from dataclasses import replace
from time import perf_counter

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import FitConfig, PeakFitConfig
from peakfit.engine.domain.fit_steps import FitStep
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.fit.pipeline import run_pipeline
from peakfit.fit.simulation import project_final_outcome
from peakfit.gui.fit_display import local_model_plane, model_contours, model_slices
from peakfit.gui.fit_process import FitProcessResult, LocalFitProcess
from peakfit.gui.fitting import prepare_local_fit
from peakfit.gui.local_fit import execute_local_fit
from peakfit.gui.peaks import WorkingPeak
from peakfit.gui.viewer import SpectrumViewer

pytestmark = pytest.mark.gui


def _example():
    axes = [
        SpectralParameters(
            size=size,
            sw=500,
            obs=500,
            car=0,
            aq_time=0.03,
            apocode=0,
            apodq1=0,
            apodq2=0,
            apodq3=0,
            p180=not direct,
            direct=direct,
            ft=True,
        )
        for size, direct in ((3, False), (32, False), (32, True))
    ]
    spectra = Spectra(
        dic={},
        data=np.zeros((3, 32, 32)),
        z_values=np.array([2.0, 0.0, 2.0]),
        params=axes,
    )
    config = PeakFitConfig(
        noise_level=0.002,
        fitting=FitConfig(lineshape="gaussian", steps=[FitStep(fix=["*.*.lw"])]),
    )
    config.parameters.position_window = 0.04
    positions = [
        (axes[1].pts2ppm(14.2) + 1.0, axes[2].pts2ppm(14.4)),
        (axes[1].pts2ppm(15.5), axes[2].pts2ppm(15.7)),
    ]
    from peakfit.engine.domain.peaks import Peak  # noqa: PLC0415

    peaks = [
        Peak(
            f"p_{i}",
            np.array(pos),
            create_shapes(spectra, config.fitting, f"p_{i}", pos, ["gaussian", "gaussian"]),
        )
        for i, pos in enumerate(positions)
    ]
    indices = [index.ravel() for index in np.indices((32, 32))]
    shapes = Cluster.evaluate_peaks(peaks, Parameters.from_peaks(peaks), indices)
    amplitudes = np.array([[5.0, -3.0, 2.0], [3.0, -2.0, 4.0]])
    rng = np.random.default_rng(42)
    spectra.data[:] = (shapes.T @ amplitudes).T.reshape(3, 32, 32)
    spectra.data += rng.normal(0, 0.002, spectra.data.shape)
    working = tuple(
        WorkingPeak(str(i), "Duplicate", (pos[0] + 0.003, pos[1] - 0.003))
        for i, pos in enumerate(positions)
    )
    prepared = prepare_local_fit(spectra, working, working[0].id, config, contour_level=0.01)
    return spectra, working, prepared, config


def _wait(condition, timeout=10):
    started = perf_counter()
    while not condition() and perf_counter() - started < timeout:
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(10, loop.quit)
        loop.exec()
    assert condition(), "Timed out waiting for background work"


def test_prepared_execution_matches_one_pass_and_preserves_scientific_inputs(monkeypatch):  # noqa: PLR0915
    spectra, working, prepared, config = _example()
    assert prepared.cluster is not None
    assert prepared.parameters is not None
    assert len(prepared.identities) == 2
    before = [parameter.model_dump() for parameter in prepared.parameters.values()]
    reference_cluster = deepcopy(prepared.cluster)
    reference = run_pipeline(
        config,
        [reference_cluster],
        prepared.noise,
        prepared.parameters.copy(deep=True),
        reference_cluster.peaks,
    ).final_outcome

    # Starting optimization must neither segment again nor resolve parameters again.
    def unexpected(*_args, **_kwargs):
        pytest.fail("Execution repeated preparation")

    monkeypatch.setattr("peakfit.gui.fitting.segment_data", unexpected)
    monkeypatch.setattr("peakfit.fit.pipeline.prepare_cluster_parameters", unexpected)
    result = execute_local_fit(prepared)
    actual = result.outcome.clusters[0]
    expected = reference.clusters[0]
    assert actual.classification == expected.classification
    assert actual.peak_names == ("p_0", "p_1")
    assert result.inputs.snapshot.peaks == working
    assert [identity.working_id for identity in result.inputs.identities] == ["0", "1"]
    assert [parameter.model_dump() for parameter in prepared.parameters.values()] == before
    for final, other in zip(
        actual.final_nonlinear_parameters, expected.final_nonlinear_parameters, strict=True
    ):
        assert final.value == pytest.approx(other.value)
        start = prepared.parameters[final.name]
        assert (final.min, final.max, final.vary) == (start.min, start.max, start.vary)
        if not final.vary:
            assert final.value == start.value
    evaluation = actual.analytical_evaluation
    assert evaluation is not None
    assert expected.analytical_evaluation is not None
    np.testing.assert_allclose(evaluation.amplitudes, expected.analytical_evaluation.amplitudes)
    np.testing.assert_array_equal(np.sign(evaluation.amplitudes), [[1, -1, 1], [1, -1, 1]])
    np.testing.assert_allclose(evaluation.model_values, evaluation.shapes.T @ evaluation.amplitudes)
    np.testing.assert_allclose(
        evaluation.raw_residuals, prepared.cluster.data - evaluation.model_values
    )
    projected = project_final_outcome(
        result.outcome, result.simulation_snapshot, prepared.scope.grid_indices
    )
    assert projected is not None
    np.testing.assert_allclose(projected, evaluation.model_values, atol=1e-12)
    reordered = project_final_outcome(
        result.outcome, result.simulation_snapshot, prepared.scope.grid_indices, planes=[2, 0, 2]
    )
    assert reordered is not None
    np.testing.assert_allclose(reordered, evaluation.model_values[:, [2, 0, 2]], atol=1e-12)
    with pytest.raises(ValueError, match="paired"):
        project_final_outcome(result.outcome, result.simulation_snapshot, [np.array([np.nan])])
    # An aliased, irregular overlapping group: display predictions and residuals
    # share the authoritative fit samples, with undefined residuals kept as NaN.
    axes = spectra.spectral_params
    iy, ix = (int(i[0]) for i in prepared.scope.grid_indices)
    slices = model_slices(result, axes, iy, ix)
    sy, sx = prepared.scope.grid_indices
    np.testing.assert_allclose(slices.x[sx[sy == iy]], evaluation.model_values[sy == iy])
    np.testing.assert_allclose(slices.profile, slices.x[ix])
    np.testing.assert_array_equal(np.isfinite(slices.residual_x[:, 0]), slices.support_x)
    np.testing.assert_array_equal(np.isfinite(slices.residual_y[:, 0]), slices.support_y)
    assert slices.residual_profile is not None
    np.testing.assert_allclose(slices.residual_profile, evaluation.raw_residuals[0])
    outside = model_slices(result, axes, 0, 0)
    assert outside.residual_profile is None
    assert np.isnan(outside.residual_x).all()
    grid = local_model_plane(result, axes, 1, prepared.contour_level)
    assert np.isfinite(grid.values).all()
    assert grid.values.min() < 0  # Signed fitted amplitudes survive smooth projection.
    higher = model_contours(
        replace(grid, edge_visible=True),
        axes,
        prepared.scope.grid_indices,
        np.array([2 * np.abs(grid.values).max()]),
    )
    assert not higher.grid.edge_visible  # Reusing a grid cannot retain a stale clipping warning.
    assert grid.y.min() <= sy.min() <= sy.max() <= grid.y.max()
    assert grid.x.min() <= sx.min() <= sx.max() <= grid.x.max()
    assert not evaluation.model_values.flags.writeable
    with pytest.raises(ValueError, match="numerical inputs"):
        execute_local_fit(replace(prepared, cluster=None, parameters=None))
    fixed = deepcopy(prepared)
    for parameter in fixed.parameters.values():
        parameter.vary = False
    with pytest.raises(ValueError, match="variable nonlinear"):
        execute_local_fit(fixed)
    limited = config.model_copy(deep=True)
    limited.fitting.max_iterations = 1
    provisional = execute_local_fit(
        replace(
            prepared,
            snapshot=replace(prepared.snapshot, configuration=limited.model_dump_json()),
        )
    )
    assert provisional.outcome.clusters[0].classification == "usable_non_converged"
    from peakfit.gui import local_fit  # noqa: PLC0415

    real_fit = local_fit.fit_single_cluster_task

    def unusable_attempt(task):
        result = real_fit(task)
        result.cost = np.nan
        return result

    monkeypatch.setattr(local_fit, "fit_single_cluster_task", unusable_attempt)
    unusable = execute_local_fit(prepared).outcome.clusters[0]
    assert unusable.classification == "unusable"
    assert unusable.analytical_evaluation is None
    assert unusable.unusable_reason


def test_spawned_lifecycle_navigation_cancellation_failure_and_stale_results(monkeypatch):  # noqa: PLR0915
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    spectra, working, prepared, _ = _example()
    viewer = SpectrumViewer(spectra)
    viewer.set_peaks(working)
    viewer.select_peak(working[0].id)
    viewer._apply_fit_preparation(prepared)
    viewer.resize(1200, 800)
    viewer.show()
    app.processEvents()
    plots = (viewer.trace_x, viewer.trace_y, viewer.profile)
    positions = tuple((plot.mapTo(viewer, QtCore.QPoint()), plot.size()) for plot in plots)
    heartbeat = []
    timer = QtCore.QTimer(viewer)
    timer.setInterval(10)
    timer.timeout.connect(lambda: heartbeat.append(1))
    timer.start()
    initial_children = {child.pid for child in multiprocessing.active_children()}
    try:
        viewer.run_local_fit()
        assert viewer.fit_busy
        assert all(plot.isVisible() for plot in plots)
        assert tuple((p.mapTo(viewer, QtCore.QPoint()), p.size()) for p in plots) == positions
        assert not viewer.add_action.isEnabled()
        with pytest.raises(ValueError, match="running fit"):
            viewer.set_plane_values(np.array([1.0, 2.0, 3.0]))
        edits = viewer.edits
        revision = viewer._scientific_revision
        with pytest.raises(ValueError, match="running fit"):
            viewer.set_peaks(working[:1])
        assert viewer.edits is edits
        assert viewer.selected_id == working[0].id
        assert viewer._scientific_revision == revision
        viewer.dismiss_fit_preview()
        viewer.select_peak(None)
        viewer.set_plane(2)
        viewer.threshold.setValue(8)
        _wait(lambda: viewer.contour_status.text() == "")
        _wait(lambda: not viewer.fit_busy)
        assert len(heartbeat) >= 2
        assert viewer.plane == 1
        assert viewer.peaks == working
        assert viewer.workspaces.currentIndex() == 0
        assert isinstance(viewer.fit_result, FitProcessResult)
        retained = viewer.fit_result
        assert retained.input_bytes < spectra.data.nbytes * 4
        assert retained.scientific.inputs.contour_level == 0.01
        viewer.show_fit_result()
        app.processEvents()
        assert all(plot.isVisible() for plot in plots)
        assert tuple((p.mapTo(viewer, QtCore.QPoint()), p.size()) for p in plots) == positions
        assert "all planes / fitted samples" in viewer.fit_preview.quality.text()
        assert "no fit has been run" not in viewer.fit_preview._details
        assert "Function evaluations:" in viewer.fit_preview._details
        assert "Not saved" in viewer.result_notice.text()
        comparison = viewer.comparison
        assert comparison.active
        assert comparison.mode.currentText() == "Overlay"
        _wait(lambda: comparison._current is not None and comparison._slice is not None)
        assert any(path.isVisible() for path in comparison.paths)
        assert not viewer.fit_mask.isVisible()
        comparison.samples.setChecked(True)
        assert viewer.fit_mask.isVisible()
        comparison.samples.setChecked(False)
        viewport = viewer.spectrum_plot.getViewBox().viewRange()
        comparison.mode.setCurrentIndex(2)
        assert comparison.residual.isVisible()
        assert not any(path.isVisible() for path in comparison.paths)
        residual = comparison.residual.image
        sy, _sx = retained.scientific.inputs.scope.grid_indices
        assert residual is not None
        assert np.count_nonzero(residual[..., 3]) == len(sy)
        assert viewer.spectrum_plot.getViewBox().viewRange() == viewport
        comparison.mode.setCurrentIndex(1)
        previous_model = comparison._current
        previous_revision = comparison._revision
        viewer.set_plane(3)
        comparison._completed("model", previous_revision, ((1,), previous_model))
        assert comparison._current is None  # A previous plane can never install itself.
        viewer.set_plane(1)
        _wait(lambda: viewer._contour_applied_revision == viewer._contour_revision)
        _wait(lambda: comparison._current is not None and comparison._slice is not None)
        assert viewer._contour_plane == viewer.plane == 0
        comparison.profile_mode.setCurrentIndex(1)
        evaluation = retained.scientific.outcome.clusters[0].analytical_evaluation
        assert evaluation is not None
        np.testing.assert_allclose(
            comparison.profile_model.yData,
            evaluation.amplitudes[0],
        )
        assert not viewer.curve_profile.isVisible()
        comparison.profile_mode.setCurrentIndex(0)
        assert viewer.curve_profile.isVisible()
        retained_threshold = retained.scientific.inputs.contour_level
        viewer.threshold.setValue(9)
        assert retained.scientific.inputs.contour_level == retained_threshold
        viewer.fit_preview.details.click()
        assert viewer.fit_preview.detail_content.isVisible()
        viewer.dismiss_fit_preview()
        assert all(plot.isVisible() for plot in plots)
        viewer.show_fit_result()
        viewer.fit_preview.result_mode.setCurrentIndex(1)
        assert viewer.fit_preview.parameters.rowCount() == 2
        viewer.set_plane(3)
        item = viewer.fit_preview.parameters.item(0, 1)
        assert item is not None
        assert item.text() == "3"
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            lambda *_args: QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        viewer._apply_fit_preparation(prepared)
        viewer.run_local_fit()
        assert not viewer.fit_busy
        assert viewer.fit_result is retained
        # A canceled replacement, including cancellation during startup, keeps the old result.
        monkeypatch.setattr(viewer, "confirm_result_loss", lambda _action: True)
        viewer._apply_fit_preparation(prepared)
        viewer.run_local_fit()
        viewer._fit_process.cancel()
        _wait(lambda: not viewer.fit_busy)
        assert viewer._fit_process.state == "canceled"
        assert viewer.fit_result is retained
        # Deliberate internal revision change exercises the delivery freshness guard.
        viewer._apply_fit_preparation(prepared)
        viewer.run_local_fit()
        viewer._scientific_revision += 1
        _wait(lambda: not viewer.fit_busy)
        assert viewer.fit_result is retained
        assert "outdated" in viewer.fit_preview.status.text()
        # A genuine child-side numerical exception has no success-shaped placeholders.
        broken = deepcopy(prepared)
        broken.cluster.data[:] = np.nan
        viewer._apply_fit_preparation(broken)
        viewer.run_local_fit()
        _wait(lambda: not viewer.fit_busy)
        assert viewer._fit_process.state == "failed"
        assert viewer.fit_result is retained
        viewer._scientific_inputs_changed()
        assert not comparison.active
        assert not comparison.mode.isVisible()
        assert not comparison.residual.isVisible()
        assert not any(path.isVisible() for path in comparison.paths)
        viewer.show_fit_result()
        assert not comparison.active  # Outdated numerical inspection never restores graphics.
        # Close while a real spawned child is computing, then check it is reaped.
        process = LocalFitProcess(viewer)
        process.start(prepared)
        _wait(lambda: process.state == "running")
        process.cancel()
        _wait(lambda: not process.busy)
        assert process.state == "canceled"
        process.start(prepared)
        process.close()
        bridge = process._thread
        assert bridge is not None
        _wait(lambda: not bridge.is_alive())
    finally:
        viewer._fit_timer.stop()
        viewer._contour_timer.stop()
        _wait(lambda: not viewer._computation._busy)
        viewer.stop_preparation()
        viewer.close()
        viewer.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        app.processEvents()
    _wait(lambda: {child.pid for child in multiprocessing.active_children()} <= initial_children)
