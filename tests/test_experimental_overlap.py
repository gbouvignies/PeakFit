"""The reported real-data collapses must be evaluated and flagged honestly."""

from pathlib import Path

import numpy as np
import pytest

from peakfit.engine.algorithms.evaluation import (
    AnalyticalModelEvaluation,
    evaluate_analytical_model,
)
from peakfit.engine.algorithms.varpro import fit_cluster
from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.fit.fitting import load_data

DATA = Path(__file__).resolve().parents[1] / "examples/02-advanced-fitting/data"


@pytest.fixture(scope="module")
def experimental_data():
    return load_data(
        DATA / "pseudo3d.ft2", DATA / "pseudo3d.list", DATA / "b1_offsets.txt", PeakFitConfig()
    )


@pytest.mark.parametrize(
    ("cluster_id", "n_points", "n_peaks"), [(269, 90, 3), (360, 106, 4), (424, 67, 2)]
)
def test_experimental_default_keeps_real_objective_and_flags_unreliable_errors(
    experimental_data, cluster_id, n_points, n_peaks
):
    cluster = next(c for c in experimental_data.clusters if c.cluster_id == cluster_id)
    assert cluster.data.shape == (n_points, 131)
    assert len(cluster.peaks) == n_peaks
    result = fit_cluster(Parameters.from_peaks(cluster.peaks), cluster, experimental_data.noise)
    evaluation = evaluate_analytical_model(cluster, result.params, experimental_data.noise)
    assert isinstance(evaluation, AnalyticalModelEvaluation)
    np.testing.assert_allclose(result.residual, evaluation.normalized_residuals, atol=1e-8)
    assert evaluation.uncertainty_warning is not None
    assert any(not np.isfinite(p.stderr) for p in result.params.values() if p.vary)


@pytest.mark.parametrize(
    ("cluster_id", "name", "expected_shift"),
    [(269, "36N-H", 119.8016505), (360, "5N-H", 117.7007364), (424, "53N-H", 115.3649824)],
)
def test_historical_settings_reproduce_released_local_solution(
    experimental_data, cluster_id, name, expected_shift
):
    cluster = next(c for c in experimental_data.clusters if c.cluster_id == cluster_id)
    params = Parameters.from_peaks(cluster.peaks)
    for parameter in params.values():
        if parameter.name.endswith(".lw"):
            parameter.value = 20 / np.pi
            parameter.min = 0.1 / np.pi
            parameter.max = 200 / np.pi
        else:
            parameter.min = parameter.value - (parameter.value - parameter.min) / 2
            parameter.max = parameter.value + (parameter.max - parameter.value) / 2
    result = fit_cluster(params, cluster, experimental_data.noise)
    assert result.params[f"{name}.F2.cs"].value == pytest.approx(expected_shift, abs=3e-5)
    evaluation = evaluate_analytical_model(cluster, result.params, experimental_data.noise)
    assert isinstance(evaluation, AnalyticalModelEvaluation)
    assert evaluation.uncertainty_warning is None
