"""The per-axis full-grid model equals the pointwise model at every native point and plane."""

from concurrent.futures import CancelledError
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from builders import two_cluster_dataset

from peakfit.engine.domain.config import FitConfig, PeakFitConfig
from peakfit.engine.scientific.simulation import project_final_grid, project_final_outcome
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.fit_positions import continuation_source, dataset_completion
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.spectrum import read_dataset

DATA = Path(__file__).resolve().parents[1] / "examples/02-advanced-fitting/data"
# Different summation order only: agreement to a few units of rounding.
TOLERANCE = 1e-12


def _fit(spectra, peaks, config, source=None):
    prepared = prepare_dataset_fit(spectra, tuple(peaks), config, source)
    return finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))


def _pointwise(result, shape):
    grid = [index.ravel() for index in np.indices(shape)]
    model = project_final_outcome(result.outcome, result.completed_fit.model_snapshot, grid)
    assert model is not None
    return model.T.reshape((-1, *shape))


def _assert_same(result, shape):
    reference = _pointwise(result, shape)
    grid = project_final_grid(result.outcome, result.completed_fit.model_snapshot, shape)
    assert grid is not None
    assert grid.shape == reference.shape
    scale = np.max(np.abs(reference))
    assert np.max(np.abs(grid - reference)) <= TOLERANCE * scale


@pytest.mark.parametrize("lineshape", ["gaussian", "lorentzian", "pvoigt"])
def test_grid_model_equals_pointwise_model_with_a_folded_peak(lineshape):
    # p_0 lies one spectral width outside the window: its aliased tails wrap in.
    spectra, peaks, config, _ = two_cluster_dataset()
    config.fitting.lineshape = lineshape
    result = _fit(spectra, peaks, config)
    _assert_same(result, spectra.data.shape[1:])


@pytest.mark.slow
def test_grid_model_equals_pointwise_model_on_every_plane_of_real_data():
    # The bundled data use apodized SP2 lineshapes in both dimensions, 131 planes.
    dataset = read_dataset(DATA / "pseudo3d.ft2", DATA / "pseudo3d.list", DATA / "b1_offsets.txt")
    result = _fit(dataset.spectra, dataset.peaks, PeakFitConfig())
    _assert_same(result, dataset.spectra.data.shape[1:])


@pytest.mark.slow
def test_grid_model_equals_pointwise_model_through_cancelling_amplitudes():
    # The GUI assessment's correction: 147N-H moved onto a +3σ residual, then refit.
    # The pair collapses to amplitudes of about ±9×10¹² that cancel to a model of
    # about 2×10⁶, the worst case for a change of summation order. Observed: max
    # |Δ| 3.9×10⁻⁷ a.u., 3.3×10⁻¹¹ σ_stat, 1.5×10⁻¹³ of the largest model value.
    dataset = read_dataset(DATA / "pseudo3d.ft2", DATA / "pseudo3d.list", DATA / "b1_offsets.txt")
    settings = PeakFitConfig(fitting=FitConfig(refine_iterations=2))
    parent = _fit(dataset.spectra, dataset.peaks, settings)
    completion = dataset_completion(
        parent, parent.inputs.snapshot, submitted_revision=0, manual_revision=0
    )
    moved = tuple(
        replace(p, positions=(115.155, 8.3488)) if p.name == "147N-H" else p
        for p in completion.working_peaks
    )
    result = _fit(dataset.spectra, moved, settings, continuation_source(completion))
    names = {i.engine_name: i.assignment for i in result.inputs.identities}
    pair = next(
        c
        for c in result.outcome.clusters
        if {names[n] for n in c.peak_names} == {"53N-H", "147N-H"}
    )
    assert pair.analytical_evaluation is not None
    assert np.max(np.abs(pair.analytical_evaluation.amplitudes)) > 1e6 * result.outcome.noise
    _assert_same(result, dataset.spectra.data.shape[1:])


def test_grid_model_rejects_inconsistent_plane_counts_as_the_pointwise_model_does():
    spectra, peaks, config, _ = two_cluster_dataset()
    result = _fit(spectra, peaks, config)
    snapshot = result.completed_fit.model_snapshot
    first, *others = snapshot.clusters
    # Clusters disagreeing on the number of fitted planes: one more on the last one.
    last = replace(others[-1], n_series=first.n_series + 1)
    broken = replace(snapshot, clusters=(first, *others[:-1], last))
    shape = spectra.data.shape[1:]
    grid = [index.ravel() for index in np.indices(shape)]
    with pytest.raises(ValueError, match="series count") as pointwise:
        project_final_outcome(result.outcome, broken, grid)
    with pytest.raises(ValueError, match="series count") as per_axis:
        project_final_grid(result.outcome, broken, shape)
    assert str(per_axis.value) == str(pointwise.value)


def test_grid_model_keeps_plane_order_cancels_and_has_no_fabricated_zero(monkeypatch):
    spectra, peaks, config, _ = two_cluster_dataset()
    result = _fit(spectra, peaks, config)
    outcome, snapshot = result.outcome, result.completed_fit.model_snapshot
    shape = spectra.data.shape[1:]
    every = project_final_grid(outcome, snapshot, shape)
    assert every is not None
    chosen = project_final_grid(outcome, snapshot, shape, planes=[2, 0, 2])
    np.testing.assert_array_equal(chosen, every[[2, 0, 2]])
    with pytest.raises(CancelledError):
        project_final_grid(outcome, snapshot, shape, canceled=lambda: True)
    with pytest.raises(ValueError, match="grid shape"):
        project_final_grid(outcome, snapshot, (32,))
    with pytest.raises(ValueError, match="planes"):
        project_final_grid(outcome, snapshot, shape, planes=[3])
    monkeypatch.setattr(type(outcome.clusters[0]), "usable", property(lambda _self: False))
    assert project_final_grid(outcome, snapshot, shape) is None
