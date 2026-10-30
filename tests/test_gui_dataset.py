"""Complete fitting, faithful total-model review and disposable-process lifecycle."""

import pickle
from concurrent.futures import CancelledError
from copy import deepcopy
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from PySide6 import QtCore, QtTest, QtWidgets
from test_gui_fit_execution import _example, _wait

from peakfit.engine.algorithms.evaluation import FitOutcomeClassification
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.fit.pipeline import run_pipeline
from peakfit.fit.simulation import project_final_outcome
from peakfit.fit.storage import read_scientific
from peakfit.gui.dataset_dialog import DatasetFitDialog
from peakfit.gui.dataset_fit import (
    DatasetFitResult,
    _completed_bytes,
    compute_dataset_fit,
    dataset_summary,
    finish_dataset_fit,
)
from peakfit.gui.fit_display import dataset_model_plane, dataset_model_slices
from peakfit.gui.fit_export import export_fit_result
from peakfit.gui.fit_process import FitProcessResult
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.peaks import WorkingPeak
from peakfit.gui.viewer import SpectrumViewer

pytestmark = pytest.mark.gui


def _dataset():
    spectra, peaks, _prepared, config = _example()
    pos = tuple(
        axis.pts2ppm(point)
        for axis, point in zip(
            spectra.spectral_params,
            (6.2, 25.4),
            strict=True,
        )
    )
    shape = Peak(
        "remote",
        np.array(pos),
        create_shapes(
            spectra,
            config.fitting,
            "remote",
            pos,
            ["gaussian", "gaussian"],
        ),
    )
    coordinates = [index.ravel() for index in np.indices((32, 32))]
    values = Cluster.evaluate_peaks([shape], Parameters.from_peaks([shape]), coordinates)[0]
    spectra.data += (np.array([4.0, -2.0, 3.0])[:, None] * values).reshape(3, 32, 32)
    peaks += (WorkingPeak("remote", "Duplicate", pos),)
    config.fitting.refine_iterations = 2
    config.fitting.steps = []
    config.clustering.contour_level = 0.1
    prepared = prepare_dataset_fit(spectra, peaks, config)
    assert len(prepared.clusters) > 1
    return spectra, peaks, config, prepared


@pytest.fixture
def dataset_viewer():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    spectra, peaks, config, prepared = _dataset()
    viewer = SpectrumViewer(spectra)
    viewer.set_peaks(peaks)
    viewer.resize(1280, 800)
    viewer.show()
    app.processEvents()
    yield viewer, config, prepared
    viewer.stop_preparation()
    viewer.close()
    viewer.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    app.processEvents()


def test_pipeline_parity_immutable_transfer_and_total_projection(tmp_path):
    spectra, peaks, config, prepared = _dataset()
    direct = deepcopy(prepared)
    expected = run_pipeline(
        config,
        direct.clusters,
        direct.noise,
        direct.parameters,
        [peak for cluster in direct.clusters for peak in cluster.peaks],
    )
    progress = []
    attempt = compute_dataset_fit(deepcopy(prepared), progress.append)
    transferred = pickle.loads(_completed_bytes(attempt))
    result = finish_dataset_fit(prepared, transferred)
    assert result.outcome.n_optimizer_passes == 2
    assert result.outcome.n_correction_updates == 1
    assert any(np.any(s.neighbor_corrections != 0) for s in result.completed_fit.samples)
    assert any("Pass 2" in message for message in progress)
    for actual, reference in zip(
        result.outcome.clusters, expected.final_outcome.clusters, strict=True
    ):
        np.testing.assert_array_equal(
            actual.analytical_evaluation.model_values, reference.analytical_evaluation.model_values
        )
        assert not actual.analytical_evaluation.amplitudes.flags.writeable
    with pytest.raises(TypeError):
        result.outcome.by_cluster_id[0] = result.outcome.clusters[0]  # ty: ignore[invalid-assignment]
    grid = tuple(index.ravel() for index in np.indices((32, 32)))
    total = project_final_outcome(
        result.outcome, result.simulation_snapshot, grid, planes=[2, 0, 2]
    )
    assert total is not None
    contributions = [
        project_final_outcome(
            result.outcome,
            result.simulation_snapshot,
            grid,
            cluster_id=c.cluster_id,
            planes=[2, 0, 2],
        )
        for c in result.outcome.clusters
    ]
    assert all(contribution is not None for contribution in contributions)
    np.testing.assert_allclose(
        total,
        np.sum(np.stack([c for c in contributions if c is not None]), axis=0),
        rtol=1e-14,
        atol=1e-14,
    )
    plane = dataset_model_plane(result, spectra.spectral_params, 2, cancel=Event())
    np.testing.assert_allclose(plane.values.ravel(), total[:, 0], rtol=1e-14, atol=1e-14)
    slices = dataset_model_slices(
        result, spectra.spectral_params, spectra.data, 14, 15, cancel=Event()
    )
    np.testing.assert_allclose(slices.x[:, 2], plane.values[14], rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(slices.residual_x, spectra.data[:, 14, :].T - slices.x)
    assert result.inputs.snapshot.plane_values == (2.0, 0.0, 2.0)
    saved = export_fit_result(result, tmp_path / "dataset")
    outcome, _samples, companion = read_scientific(saved / "scientific.npz")
    assert outcome.n_correction_updates == 1
    assert (outcome.n_correction_updates > 0) is True
    assert companion["plane_values"].tolist() == [2.0, 0.0, 2.0]
    assert [p["peak_id"] for p in companion["peak_identities"]] == ["0", "1", "remote"]
    config.fitting.refine_iterations = 1
    single = prepare_dataset_fit(spectra, peaks, config)
    single_result = finish_dataset_fit(single, compute_dataset_fit(deepcopy(single)))
    saved = export_fit_result(single_result, tmp_path / "one-pass")
    outcome, _samples, companion = read_scientific(saved / "scientific.npz")
    assert outcome.n_correction_updates == 0
    assert (outcome.n_correction_updates > 0) is False


def test_settings_are_reviewed_and_independent_of_display(dataset_viewer):
    viewer, config, _prepared = dataset_viewer
    viewer.threshold.setValue(20)
    config.clustering.contour_level = None
    dialog = DatasetFitDialog(viewer, config)
    dialog.show()
    try:
        _wait(lambda: dialog.prepared is not None)
        prepared = dialog.prepared
        assert prepared is not None
        assert dialog.passes.value() == 2
        assert prepared.contour_level == config.clustering.contour_factor * prepared.noise
        viewer.threshold.setValue(10)
        assert dialog.prepared is prepared
        QtTest.QTest.mouseClick(dialog.advanced_toggle, QtCore.Qt.MouseButton.LeftButton)
        assert dialog.parameters.isVisible()
        assert dialog.parameters.rowCount() == len(prepared.parameters)
        dialog.threshold.setValue(7)
        assert not dialog.run.isEnabled()
        _wait(lambda: dialog.prepared is not None)
        assert dialog.prepared is not None
        assert dialog.prepared.contour_level == 7 * dialog.prepared.noise
        assert dialog.run.isDefault()
        QtTest.QTest.keyClick(dialog.run, QtCore.Qt.Key.Key_Return)
        assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_complete_gui_review_navigation_cancel_failure_and_outdated(  # noqa: PLR0915
    dataset_viewer, monkeypatch
):
    viewer, _config, prepared = dataset_viewer
    viewer.run_dataset_fit(prepared)
    _wait(lambda: not viewer.fit_busy)
    assert viewer._fit_process.state == "completed", viewer.workspace.summary.toolTip()
    retained = viewer.fit_result
    assert isinstance(retained.scientific, DatasetFitResult)
    assert viewer.workspace.details.isHidden()
    comparison = viewer.comparison
    _wait(lambda: comparison._current is not None and comparison._slice is not None)
    assert comparison.mode.currentText() == "Overlay"
    assert "Complete model" in viewer.workspace.summary.text()
    assert any(path.isVisible() for path in comparison.paths)
    original_revision, original_model = comparison._revision, comparison._current
    for plane in (3, 1, 2):
        viewer.set_plane(plane)
    comparison._completed("model", original_revision, ((0,), original_model))
    assert comparison._current is None
    _wait(lambda: comparison._current is not None and comparison._slice is not None)
    direct = dataset_model_plane(
        retained.scientific, viewer.spectra.spectral_params, 1, cancel=Event()
    )
    np.testing.assert_array_equal(comparison._current.grid.values, direct.values)
    np.testing.assert_allclose(
        comparison.model_x.yData, direct.values[viewer.iy], rtol=1e-14, atol=1e-14
    )
    comparison.mode.setCurrentIndex(2)
    _wait(lambda: comparison.residual.isVisible() and comparison._slice is not None)
    np.testing.assert_allclose(
        comparison.model_x.yData,
        (viewer.spectra.data[1, viewer.iy] - direct.values[viewer.iy]) / prepared.noise,
    )
    assert comparison.residual.image.shape[:2] == direct.values.shape
    assert np.all(comparison.residual.image[..., 3] == 255)
    comparison.mode.setCurrentIndex(0)
    assert not any(path.isVisible() for path in comparison.paths)
    assert not comparison.residual.isVisible()
    monkeypatch.setattr(
        viewer, "_result_loss_choice", lambda _: QtWidgets.QMessageBox.StandardButton.Cancel
    )
    viewer.run_dataset_fit(prepared)
    assert not viewer.fit_busy
    assert viewer.fit_result is retained
    # A cancelled replacement and a real child failure preserve the original run.
    monkeypatch.setattr(viewer, "confirm_result_loss", lambda _: True)
    viewer.run_dataset_fit(prepared)
    viewer._fit_process.cancel()
    _wait(lambda: not viewer.fit_busy)
    assert viewer.fit_result is retained
    invalid = replace(prepared, parameters=Parameters())
    # Broken geometry reaches the real child; there is no invented partial model.
    invalid = deepcopy(invalid)
    invalid.clusters[0].data = np.full_like(invalid.clusters[0].data, np.nan)
    viewer.run_dataset_fit(invalid)
    _wait(lambda: not viewer.fit_busy)
    assert viewer._fit_process.state == "failed"
    assert viewer.fit_result is retained
    before = retained.scientific.completed_fit.samples[0].data.copy()
    viewer.edits.update(viewer.peaks[0].id, "Edited", viewer.peaks[0].positions)
    viewer.edits_changed.emit()
    assert not comparison.active
    assert "Outdated" in viewer.workspace.summary.text()
    viewer.undo()
    assert not comparison.active
    viewer.show_fit_result(activate=False)
    assert not comparison.active  # Undo does not recertify a retained result.
    np.testing.assert_array_equal(retained.scientific.completed_fit.samples[0].data, before)


def test_projection_cancellation_never_returns_partial_model():
    spectra, _peaks, _config, prepared = _dataset()
    result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    cancel = Event()
    cancel.set()
    with pytest.raises(CancelledError):
        dataset_model_plane(result, spectra.spectral_params, 0, cancel=cancel)
    assert "uncertainty" in dataset_summary(result)


def test_summary_separates_termination_from_usability_and_no_model(dataset_viewer):
    viewer, _config, prepared = dataset_viewer
    result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    outcomes = result.outcome.clusters
    assert len(outcomes) == 2
    # Inject terminal classifications at the presentation seam: termination and
    # analytical usability are independent, even when optimizer.success is true.
    usable = replace(
        outcomes[0],
        classification=FitOutcomeClassification.USABLE_NON_CONVERGED,
        optimizer_provenance=replace(outcomes[0].optimizer_provenance, converged=False),
    )
    failed = replace(
        outcomes[1],
        classification=FitOutcomeClassification.UNUSABLE,
        analytical_evaluation=None,
        unusable_reason="No finite analytical model",
        optimizer_provenance=replace(outcomes[1].optimizer_provenance, converged=True),
    )
    outcome = replace(
        result.outcome,
        clusters=(usable, failed),
        by_cluster_id={c.cluster_id: c for c in (usable, failed)},
    )
    mixed = replace(result, completed_fit=replace(result.completed_fit, outcome=outcome))
    assert "1/2 converged" in dataset_summary(mixed)
    assert "1/2 usable" in dataset_summary(mixed)
    assert "Incomplete model" in dataset_summary(mixed)
    failed_first = replace(
        usable, classification=FitOutcomeClassification.UNUSABLE, analytical_evaluation=None
    )
    outcome = replace(
        outcome,
        clusters=(failed_first, failed),
        by_cluster_id={c.cluster_id: c for c in (failed_first, failed)},
    )
    unusable = replace(result, completed_fit=replace(result.completed_fit, outcome=outcome))
    assert "No usable model" in dataset_summary(unusable)
    viewer.comparison.show_result(FitProcessResult(unusable, 0, 0, 0, 0), outdated=False)
    assert not viewer.comparison.active
    assert viewer.comparison._current is None
    assert not viewer.comparison.residual.isVisible()
