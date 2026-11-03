"""Complete fitting, faithful total-model review and disposable-process lifecycle."""

import pickle
from concurrent.futures import CancelledError
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from builders import two_cluster_dataset
from gui_helpers import wait_until
from PySide6 import QtCore

from peakfit.engine.algorithms.evaluation import FitOutcomeClassification
from peakfit.engine.scientific.simulation import project_final_outcome
from peakfit.fit.pipeline import run_pipeline
from peakfit.gui.child_process import encode
from peakfit.gui.dataset_fit import (
    compute_dataset_fit,
    dataset_summary,
    finish_dataset_fit,
)
from peakfit.gui.editor.workspace import EditorWorkspace
from peakfit.gui.fit_display import dataset_model_plane
from peakfit.gui.fit_export import export_fit_result
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.spectrum import Dataset
from peakfit.io.scientific import read_scientific

pytestmark = pytest.mark.gui


@pytest.fixture
def dataset_workspace(qapp):
    spectra, peaks, config, prepared = two_cluster_dataset()
    workspace = EditorWorkspace(Dataset(Path("synthetic.ft2"), spectra, peaks, (), True, 5.0, 0.01))
    workspace.resize(1280, 800)
    workspace.show()
    qapp.processEvents()
    yield workspace, config, prepared
    workspace.stop()
    workspace.close()
    workspace.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    qapp.processEvents()


def test_pipeline_parity_immutable_transfer_and_total_projection(tmp_path):
    spectra, peaks, config, prepared = two_cluster_dataset()
    direct = deepcopy(prepared)
    expected = run_pipeline(
        config,
        direct.clusters,
        direct.noise,
        direct.parameters,
    )
    progress = []
    attempt = compute_dataset_fit(deepcopy(prepared), progress.append)
    transferred = pickle.loads(encode(attempt))
    result = finish_dataset_fit(prepared, transferred)
    assert result.outcome.n_optimizer_passes == 2
    assert result.outcome.n_correction_updates == 1
    assert any(np.any(s.neighbor_corrections != 0) for s in result.completed_fit.samples)
    assert any("Pass 2" in message for message in progress)
    for actual, reference in zip(
        result.outcome.clusters, expected.completed_fit.outcome.clusters, strict=True
    ):
        np.testing.assert_array_equal(
            actual.analytical_evaluation.model_values, reference.analytical_evaluation.model_values
        )
        assert not actual.analytical_evaluation.amplitudes.flags.writeable
    with pytest.raises(TypeError):
        result.outcome.by_cluster_id[0] = result.outcome.clusters[0]  # ty: ignore[invalid-assignment]
    grid = tuple(index.ravel() for index in np.indices((32, 32)))
    total = project_final_outcome(
        result.outcome, result.completed_fit.model_snapshot, grid, planes=[2, 0, 2]
    )
    assert total is not None
    contributions = [
        project_final_outcome(
            result.outcome,
            result.completed_fit.model_snapshot,
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
    with pytest.raises(ValueError, match="paired"):
        project_final_outcome(
            result.outcome, result.completed_fit.model_snapshot, [np.array([np.nan])]
        )
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


def test_fit_setup_prepares_working_settings_independently_of_display(dataset_workspace):
    workspace, config, _prepared = dataset_workspace
    fitting, setup = workspace.fitting, workspace.panel.setup
    config.clustering.contour_level = None
    fitting.configure(config)
    wait_until(lambda: fitting.prepared is not None)
    prepared = fitting.prepared
    assert prepared is not None
    assert setup.passes.value() == 2
    assert prepared.contour_level == config.clustering.contour_factor * prepared.noise
    assert setup.preparation.text().startswith(f"{len(prepared.clusters)} clusters")
    workspace.canvas.set_plane(2)
    workspace.canvas.contour_threshold *= 3
    workspace.canvas.request_contours()
    assert fitting.prepared is prepared
    setup.threshold.setValue(7)
    assert fitting.prepared is None
    assert fitting.configuration.clustering.contour_factor == 7
    wait_until(lambda: fitting.prepared is not None)
    assert fitting.prepared.contour_level == 7 * fitting.prepared.noise


def test_projection_cancellation_never_returns_partial_model():
    spectra, _peaks, _config, prepared = two_cluster_dataset()
    result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    cancel = Event()
    cancel.set()
    with pytest.raises(CancelledError):
        dataset_model_plane(result, spectra.spectral_params, 0, cancel=cancel)
    assert "uncertainty" in dataset_summary(result)


def test_summary_separates_termination_from_usability_and_no_model():
    _spectra, _peaks, _config, prepared = two_cluster_dataset()
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
    )
    unusable = replace(result, completed_fit=replace(result.completed_fit, outcome=outcome))
    assert "No usable model" in dataset_summary(unusable)
