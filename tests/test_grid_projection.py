"""The per-axis full-grid model equals the pointwise model at every native point and plane."""

from concurrent.futures import CancelledError
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from builders import two_cluster_dataset

from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.scientific.simulation import project_final_grid, project_final_outcome
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.spectrum import read_dataset

DATA = Path(__file__).resolve().parents[1] / "examples/02-advanced-fitting/data"
# Different summation order only: agreement to a few units of rounding.
TOLERANCE = 1e-12


def _fit(spectra, peaks, config):
    prepared = prepare_dataset_fit(spectra, peaks, config)
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
