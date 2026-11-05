"""The reported real-data collapses must be evaluated and flagged honestly."""

from pathlib import Path

import numpy as np
import pytest

from peakfit.engine.algorithms.evaluation import (
    AnalyticalModelEvaluation,
    evaluate_analytical_model,
)
from peakfit.engine.algorithms.noise_correlation import not_modelled
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


def _fit_from(cluster, noise, linewidth=None):
    """Fit one cluster from its default starts, or with every width starting at ``linewidth``."""
    params = Parameters.from_peaks(cluster.peaks)
    if linewidth is not None:
        for parameter in params.values():
            if parameter.name.endswith(".lw"):
                parameter.value = linewidth
    result = fit_cluster(params, cluster, noise, correlation=None)
    evaluation = evaluate_analytical_model(
        cluster, result.params, noise, noise_model=not_modelled("synthetic spectrum")
    )
    assert isinstance(evaluation, AnalyticalModelEvaluation)
    np.testing.assert_allclose(result.residual, evaluation.normalized_residuals, atol=1e-8)
    return result, evaluation


@pytest.mark.parametrize(
    ("cluster_id", "n_points", "n_peaks"), [(269, 90, 3), (360, 106, 4), (424, 67, 2)]
)
def test_a_broad_start_collapses_keeps_real_objective_and_flags_unreliable_errors(
    experimental_data, cluster_id, n_points, n_peaks
):
    # 20 Hz was the default start until #135.
    cluster = next(c for c in experimental_data.clusters if c.cluster_id == cluster_id)
    assert cluster.data.shape == (n_points, 131)
    assert len(cluster.peaks) == n_peaks
    result, evaluation = _fit_from(cluster, experimental_data.noise, linewidth=20.0)
    assert evaluation.uncertainty_warning is not None
    assert any(not np.isfinite(p.stderr) for p in result.params.values() if p.vary)


@pytest.mark.parametrize("cluster_id", [269, 360, 424])
def test_the_default_start_reaches_well_conditioned_solutions(experimental_data, cluster_id):
    # docs/validation/linewidth-start.md. For 269 and 360 the resolved solution also
    # has the lower chi-squared; for 424 (53N-H/147N-H) the collapsed limit is lower.
    cluster = next(c for c in experimental_data.clusters if c.cluster_id == cluster_id)
    result, evaluation = _fit_from(cluster, experimental_data.noise)
    assert evaluation.uncertainty_warning is None
    assert all(np.isfinite(p.stderr) for p in result.params.values() if p.vary)
    _, broad = _fit_from(cluster, experimental_data.noise, linewidth=20.0)
    lower = evaluation.statistics.chi_squared < broad.statistics.chi_squared
    assert lower is (cluster_id != 424)


# The local solutions the historical settings reach. The released values (119.8016505,
# 117.7007364, 115.3649824) were fitted with aq = APOD/SW; with the window over the
# acquired signal (docs/lineshapes.md) they move by +0.96, -0.02 and +0.09 Hz, and
# cluster 269, which holds the 36N-H/93N-H identifiability pair, now carries the
# uncertainty warning from either start.
@pytest.mark.parametrize(
    ("cluster_id", "name", "expected_shift", "warned"),
    [
        (269, "36N-H", 119.8135054, True),
        (360, "5N-H", 117.7004604, False),
        (424, "53N-H", 115.3660666, False),
    ],
)
def test_historical_settings_reach_their_local_solution(
    experimental_data, cluster_id, name, expected_shift, warned
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
    result = fit_cluster(params, cluster, experimental_data.noise, correlation=None)
    assert result.params[f"{name}.F2.cs"].value == pytest.approx(expected_shift, abs=3e-5)
    evaluation = evaluate_analytical_model(
        cluster,
        result.params,
        experimental_data.noise,
        noise_model=not_modelled("synthetic spectrum"),
    )
    assert isinstance(evaluation, AnalyticalModelEvaluation)
    assert (evaluation.uncertainty_warning is not None) is warned
