"""The posterior MCMC samples: docs/mcmc-formulation.md, validation V1-V3, V7-V9."""

from dataclasses import replace

import numpy as np
import pytest
from builders import completed_result, peak_pair

from peakfit.engine.algorithms.evaluation import FitOutcomeClassification
from peakfit.engine.domain.config import FitConfig, PeakFitConfig
from peakfit.engine.scientific.sampling import conditional_amplitudes, sampling_problem
from peakfit.fit.pipeline import run_pipeline
from peakfit.gui.fit_export import export_fit_result
from peakfit.mcmc.analysis import run_mcmc_analysis
from peakfit.shared.exceptions import ScientificIntegrityError


@pytest.fixture(scope="module")
def two_clusters():
    """A single-peak and a two-peak cluster, three planes, with retained observations."""
    return completed_result([], False, True)


@pytest.fixture(scope="module")
def phased():
    """A two-peak cluster whose fit varied a cluster-scoped phase."""
    cluster, params = peak_pair(6.0, noise=0.01, lineshape="no_apod", fit_phase=("F2",))
    config = PeakFitConfig(fitting=FitConfig(lineshape="no_apod", fit_phase=["F2"]))
    return run_pipeline(config, [cluster], 0.01, params).completed_fit


def _problems(two_clusters, phased):
    completed = two_clusters[2].completed_fit
    single, pair = (sampling_problem(completed, c.cluster_id) for c in completed.outcome.clusters)
    assert len(single.cluster.peaks) == 1
    assert len(pair.cluster.peaks) == 2
    return {
        "single peak": (completed, single),
        "multi-plane pair": (completed, pair),
        "phase-fitted": (phased, sampling_problem(phased, 1)),
    }


@pytest.mark.parametrize("case", ["single peak", "multi-plane pair", "phase-fitted"])
def test_v1_the_likelihood_at_the_fit_is_the_recorded_chi_squared(two_clusters, phased, case):
    completed, problem = _problems(two_clusters, phased)[case]
    recorded = completed.outcome.cluster(problem.cluster.cluster_id)
    chi2 = recorded.analytical_evaluation.statistics.chi_squared
    np.testing.assert_allclose(-2.0 * problem.log_likelihood(problem.start), chi2, rtol=1e-10)
    assert problem.data.shape[1] == problem.cluster.n_series > 1


def test_v1_a_likelihood_that_disagrees_with_the_record_is_refused(two_clusters):
    completed = two_clusters[2].completed_fit
    cluster = completed.outcome.clusters[0]
    evaluation = cluster.analytical_evaluation
    altered = replace(
        cluster,
        analytical_evaluation=replace(
            evaluation,
            statistics=replace(
                evaluation.statistics, chi_squared=evaluation.statistics.chi_squared * 1.01
            ),
        ),
    )
    outcome = replace(
        completed.outcome,
        clusters=tuple(altered if c is cluster else c for c in completed.outcome.clusters),
    )
    with pytest.raises(ScientificIntegrityError, match="recorded chi-squared"):
        sampling_problem(replace(completed, outcome=outcome), cluster.cluster_id)


def test_v9_a_fitted_phase_is_sampled(phased):
    problem = sampling_problem(phased, 1)
    assert "cluster_1.F2.phase" in problem.names
    theta = problem.start.copy()
    theta[problem.names.index("cluster_1.F2.phase")] += 5.0
    assert problem.log_likelihood(theta) < problem.log_likelihood(problem.start)


def test_only_the_target_cluster_must_be_usable(two_clusters):
    completed = two_clusters[2].completed_fit
    unusable, usable = completed.outcome.clusters
    outcome = replace(
        completed.outcome,
        clusters=(
            replace(
                unusable,
                classification=FitOutcomeClassification.UNUSABLE,
                unusable_reason="fixture",
            ),
            usable,
        ),
    )
    modified = replace(completed, outcome=outcome)
    assert np.isfinite(
        sampling_problem(modified, usable.cluster_id).log_likelihood(
            sampling_problem(modified, usable.cluster_id).start
        )
    )
    with pytest.raises(ValueError, match="unusable"):
        sampling_problem(modified, unusable.cluster_id)


def test_v8_the_fit_bounds_are_the_prior_support(two_clusters, phased):
    _, problem = _problems(two_clusters, phased)["multi-plane pair"]
    finite = np.flatnonzero(np.isfinite(problem.upper))
    assert finite.size
    theta = problem.start.copy()
    theta[finite[0]] = problem.upper[finite[0]]
    assert np.isfinite(problem.log_likelihood(theta))
    theta[finite[0]] = np.nextafter(problem.upper[finite[0]], np.inf)
    assert problem.log_likelihood(theta) == -np.inf
    assert problem.conditional(theta) is None


def test_v2_conditional_draws_match_the_closed_form(two_clusters, phased):
    _, problem = _problems(two_clusters, phased)["multi-plane pair"]
    conditional = problem.conditional(problem.start)
    assert conditional is not None
    assert conditional.determined.all()
    shapes = problem.cluster.evaluate(problem.parameters)
    expected_mean = np.linalg.lstsq(shapes.T, problem.data, rcond=None)[0]
    expected_cov = problem.noise**2 * np.linalg.inv(shapes @ shapes.T)
    n = 200_000
    draws = conditional.draw(np.random.default_rng(20261104), n)
    assert draws.shape == (n, 2, problem.data.shape[1])
    np.testing.assert_allclose(conditional.mean, expected_mean, rtol=1e-10, atol=0.0)
    sd = np.sqrt(np.diag(expected_cov))
    for plane in range(problem.data.shape[1]):
        sample = draws[:, :, plane]
        # Means within five standard errors.
        assert np.all(np.abs(sample.mean(axis=0) - expected_mean[:, plane]) < 5 * sd / np.sqrt(n))
        # Covariances within five standard errors of a Gaussian sample covariance.
        error = np.sqrt((np.outer(sd**2, sd**2) + expected_cov**2) / n)
        assert np.all(np.abs(np.cov(sample.T) - expected_cov) < 5 * error)


def test_v3_the_jeffreys_integral_is_the_profile_likelihood_up_to_a_constant(two_clusters, phased):
    _, problem = _problems(two_clusters, phased)["single peak"]
    y = problem.data[:, 0]
    name = next(n for n in problem.names if n.endswith(".lw"))
    ratios = []
    for factor in (0.8, 0.9, 1.0, 1.1, 1.25):
        parameters = problem.parameters.copy()
        parameters[name].value *= factor
        phi = problem.cluster.evaluate(parameters)[0]
        log_profile = conditional_amplitudes(phi[None, :], y[:, None], problem.noise).log_likelihood
        g = phi @ phi
        a_hat = phi @ y / g
        spread = problem.noise / np.sqrt(g)
        a = np.linspace(a_hat - 12 * spread, a_hat + 12 * spread, 40001)
        chi2 = np.sum((y[None, :] - a[:, None] * phi[None, :]) ** 2, axis=1) / problem.noise**2
        integral = np.trapezoid(np.exp(-0.5 * (chi2 + 2 * log_profile)) * np.sqrt(g), a)
        ratios.append(integral)
    np.testing.assert_allclose(ratios, np.sqrt(2 * np.pi) * problem.noise, rtol=1e-9)


def _gaussians(centers, width=3.0):
    x = np.arange(60.0)
    return np.exp(-0.5 * ((x[None, :] - np.asarray(centers)[:, None]) / width) ** 2)


def test_v7_a_duplicated_peak_is_undetermined_but_the_density_is_finite():
    shapes = _gaussians([20.0, 20.0, 40.0])
    data = (shapes.T @ np.array([[2.0], [1.0], [3.0]])) + 0.01 * np.random.default_rng(
        7
    ).standard_normal((60, 1))
    conditional = conditional_amplitudes(shapes, data, 0.01)
    assert np.isfinite(conditional.log_likelihood)
    assert conditional.determined.tolist() == [False, False, True]
    draws = conditional.draw(np.random.default_rng(1), 1000)
    assert np.isnan(draws[:, :2]).all()
    assert np.isfinite(draws[:, 2]).all()
    # The well-separated peak keeps its exact conditional distribution.
    np.testing.assert_allclose(draws[:, 2, 0].std(), 0.01 / np.linalg.norm(shapes[2]), rtol=0.1)


def test_v7_nearly_coincident_shapes_are_undetermined_although_full_rank():
    shapes = _gaussians([20.0, 20.0 + 1e-9, 40.0])
    singular = np.linalg.svd(shapes, compute_uv=False)
    eps = np.finfo(float).eps
    # Full numerical rank, but beyond the conditioning at which the fit withholds
    # its amplitude covariance.
    assert eps * 60 * singular[0] < singular[-1] < np.sqrt(eps) * singular[0]
    conditional = conditional_amplitudes(shapes, shapes.T @ np.ones((3, 2)), 0.01)
    assert conditional.determined.tolist() == [False, False, True]


def test_the_cli_analysis_samples_only_the_targeted_cluster(tmp_path, two_clusters):
    directory = export_fit_result(two_clusters[3], tmp_path / "results")
    completed = two_clusters[2].completed_fit
    target = completed.outcome.clusters[1]
    analysis = run_mcmc_analysis(
        directory,
        target_peaks=[target.peak_names[0]],
        n_walkers=20,
        n_steps=40,
        burn_in=10,
        auto_burnin=False,
    )
    (result,) = analysis.cluster_results
    assert result.cluster.cluster_id == target.cluster_id
    assert np.isfinite(result.result.values).all()
