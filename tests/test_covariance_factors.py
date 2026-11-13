"""First-order amplitude covariance from per-cluster factors (profile-uncertainty §3.1).

The reference is the OLS sandwich over every parameter, θ and all amplitudes, built
from the full Jacobian: (JᵀJ)⁻¹Jᵀ(σ²R ⊗ I)J(JᵀJ)⁻¹. It shares only the per-peak shape
derivatives with the factors, and those are checked against finite differences of the
shapes.
"""

import numpy as np
import pytest
from builders import peak_pair

from peakfit.engine.algorithms.covariance_factors import (
    covariance_between_planes,
    covariance_factors,
    peak_derivatives,
)
from peakfit.engine.algorithms.evaluation import (
    AnalyticalModelEvaluation,
    evaluate_analytical_model,
)
from peakfit.engine.algorithms.linear_algebra import calculate_amplitudes_with_uncertainty
from peakfit.engine.algorithms.noise_correlation import (
    AxisProcessing,
    NoiseModel,
    cluster_correlation,
)
from peakfit.engine.algorithms.varpro import fit_cluster
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.params_scalar import Parameters

NOISE = 0.3


def _correlation(n: int) -> np.ndarray:
    """A smooth positive definite correlation between the cluster's points."""
    lag = np.subtract.outer(np.arange(n), np.arange(n))
    return 0.6 * np.exp(-0.5 * (lag / 1.7) ** 2) + 0.4 * 0.5 ** np.abs(lag)


def _varied(cluster: Cluster, params: Parameters) -> list[str]:
    return [name for name in cluster.parameter_names(params) if params[name].vary]


def _full_sandwich(cluster, params, amplitudes, correlation):
    """θ block and amplitude block (L, S, L, S) of the full-Jacobian OLS sandwich."""
    names = _varied(cluster, params)
    shapes = cluster.evaluate(params)
    derivatives = peak_derivatives(cluster, params, names)  # (L, θ, n)
    n_peaks, n_points = shapes.shape
    n_planes = amplitudes.shape[1]
    n_theta = len(names)
    # Rows (point, plane), the plane fastest, as the residuals are ordered.
    jacobian = np.zeros((n_points, n_planes, n_theta + n_peaks * n_planes))
    jacobian[:, :, :n_theta] = np.einsum("ls,lpi->isp", amplitudes, derivatives)
    for peak in range(n_peaks):
        for plane in range(n_planes):
            jacobian[:, plane, n_theta + peak * n_planes + plane] = shapes[peak]
    jacobian = jacobian.reshape(n_points * n_planes, -1)
    bread = np.linalg.inv(jacobian.T @ jacobian)
    noise = NOISE**2 * np.kron(correlation, np.eye(n_planes))
    full = bread @ jacobian.T @ noise @ jacobian @ bread
    theta = full[:n_theta, :n_theta]
    blocks = full[n_theta:, n_theta:].reshape(n_peaks, n_planes, n_peaks, n_planes)
    return theta, blocks


def _factors(cluster, params, amplitudes, correlation):
    names = _varied(cluster, params)
    shapes = cluster.evaluate(params)
    _, _, conditional = calculate_amplitudes_with_uncertainty(
        shapes, cluster.corrected_data, NOISE, correlation
    )
    return covariance_factors(
        shapes,
        peak_derivatives(cluster, params, names),
        amplitudes,
        conditional,
        NOISE,
        correlation,
    )


def _relative(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.abs(actual - expected).max() / np.abs(expected).max())


def _cases():
    overlapping = peak_pair(3.0)
    isolated = peak_pair(30.0)
    shared = peak_pair(3.0, lineshape="no_apod", fit_phase=("F2",))
    shared[1]["cluster_1.F2.phase"].value = 4.0
    fixed = peak_pair(3.0)
    fixed[1][_varied(*fixed)[0]].vary = False
    all_fixed = peak_pair(3.0)
    for name in _varied(*all_fixed):
        all_fixed[1][name].vary = False
    return {
        "overlapping": overlapping,
        "isolated": isolated,
        "shared": shared,
        "fixed": fixed,
        "all fixed": all_fixed,
    }


@pytest.mark.parametrize("case", ["overlapping", "isolated", "shared", "fixed", "all fixed"])
def test_reconstructed_covariance_is_the_full_jacobian_sandwich(case) -> None:
    cluster, params = _cases()[case]
    amplitudes = np.linalg.lstsq(cluster.evaluate(params).T, cluster.data, rcond=None)[0]
    correlation = _correlation(cluster.n_points)
    factors = _factors(cluster, params, amplitudes, correlation)
    theta, expected = _full_sandwich(cluster, params, amplitudes, correlation)
    assert _relative(covariance_between_planes(factors, amplitudes), expected) < 1e-10
    assert factors.parameter_covariance is not None
    if theta.size:
        assert _relative(factors.parameter_covariance, theta) < 1e-10


@pytest.mark.parametrize("case", ["overlapping", "shared"])
def test_peak_derivatives_are_each_peaks_own_shape_derivatives(case) -> None:
    cluster, params = _cases()[case]
    names = _varied(cluster, params)
    if case == "shared":
        assert "cluster_1.F2.phase" in names
    analytical = peak_derivatives(cluster, params, names)
    for index, name in enumerate(names):
        step = 1e-6 if name.endswith(".cs") else 1e-4
        value = params[name].value
        params[name].value = value + step
        above = cluster.evaluate(params)
        params[name].value = value - step
        below = cluster.evaluate(params)
        params[name].value = value
        numerical = (above - below) / (2 * step)
        assert _relative(analytical[:, index], numerical) < 1e-6


def test_cross_terms_vanish_for_independent_points() -> None:
    cluster, params = peak_pair(3.0)
    amplitudes = np.linalg.lstsq(cluster.evaluate(params).T, cluster.data, rcond=None)[0]
    for correlation in (None, np.eye(cluster.n_points)):
        factors = _factors(cluster, params, amplitudes, correlation)
        assert factors.cross_gain is not None
        assert factors.parameter_covariance is not None
        gain = np.einsum("ms,mlp->slp", amplitudes, factors.cross_gain)
        lineshape = np.einsum(
            "ms,mlp,pq->slq", amplitudes, factors.shape_gain, factors.parameter_covariance
        )
        assert np.abs(gain).max() < 1e-12 * np.abs(lineshape).max()


def test_a_single_peak_lineshape_terms_are_a_common_scale() -> None:
    pair, _ = peak_pair(30.0)
    peak = pair.peaks[0]
    params = Parameters.from_peaks([peak])
    cluster = Cluster(1, [peak], pair.grid_indices, np.zeros((pair.n_points, pair.n_series)))
    amplitudes = np.array([[2.0, -1.0, 3.0, 0.5]])
    cluster.data = cluster.evaluate(params).T @ amplitudes
    factors = _factors(cluster, params, amplitudes, _correlation(cluster.n_points))
    covariance = covariance_between_planes(factors, amplitudes)[0, :, 0, :]
    lineshape = covariance - factors.conditional[0, 0] * np.eye(amplitudes.shape[1])
    scale = amplitudes[0] / amplitudes[0, 0]
    expected = lineshape[0, 0] * np.outer(scale, scale)
    assert abs(lineshape[0, 0]) > 1e-3 * factors.conditional[0, 0]
    assert _relative(lineshape, expected) < 1e-12


MODEL = NoiseModel(axes=(AxisProcessing(40, 128, "sine_bell", offset=0.4, end=0.98, power=2.0),))


def _fitted_evaluation(cluster, params) -> AnalyticalModelEvaluation:
    correlation = cluster_correlation(MODEL, cluster.grid_indices)
    fit_cluster(params, cluster, NOISE, correlation=correlation)
    evaluation = evaluate_analytical_model(cluster, params, NOISE, MODEL)
    assert isinstance(evaluation, AnalyticalModelEvaluation)
    return evaluation


def test_the_conditional_block_is_the_published_amplitude_covariance() -> None:
    cluster, params = peak_pair(3.0, noise=0.01)
    evaluation = _fitted_evaluation(cluster, params)
    factors = evaluation.covariance_factors
    assert factors is not None
    assert factors.available
    assert np.array_equal(factors.conditional, evaluation.amplitude_covariance)
    covariance = covariance_between_planes(factors, evaluation.amplitudes)
    for plane in range(cluster.n_series):
        assert np.all(np.diag(covariance[:, plane, :, plane]) >= np.diag(factors.conditional))


def test_a_parameter_at_an_active_bound_is_held_fixed() -> None:
    cluster, params = peak_pair(6.0, noise=0.01)
    name = params.get_vary_names()[0]
    # A lower bound above the true position: the fit is pressed against it.
    params[name].value += 0.002
    params[name].min = params[name].value
    evaluation = _fitted_evaluation(cluster, params)
    assert params[name].value == pytest.approx(params[name].min, abs=1e-12)
    assert not np.isfinite(params[name].stderr)
    factors = evaluation.covariance_factors
    assert factors is not None
    assert factors.available
    assert np.array_equal(factors.conditional, evaluation.amplitude_covariance)
    held = params.copy(deep=True)
    held[name].vary = False
    correlation = cluster_correlation(MODEL, cluster.grid_indices)
    _, expected = _full_sandwich(cluster, held, evaluation.amplitudes, correlation)
    covariance = covariance_between_planes(factors, evaluation.amplitudes)
    assert _relative(covariance, expected) < 1e-10


def test_unidentifiable_lineshape_parameters_withhold_the_lineshape_terms() -> None:
    cluster, params = peak_pair(6.0)
    names = _varied(cluster, params)
    shapes = cluster.evaluate(params)
    derivatives = peak_derivatives(cluster, params, names)
    derivatives[:, 1] = derivatives[:, 0]  # two parameters with one effect
    amplitudes, _, conditional = calculate_amplitudes_with_uncertainty(
        shapes, cluster.data, NOISE, None
    )
    factors = covariance_factors(shapes, derivatives, amplitudes, conditional, NOISE, None)
    assert not factors.available
    assert np.array_equal(factors.conditional, conditional)
    assert np.isnan(covariance_between_planes(factors, amplitudes)).all()


def test_unidentifiable_amplitudes_stay_withheld() -> None:
    cluster, params = peak_pair(0.0, noise=0.01)
    evaluation = _fitted_evaluation(cluster, params)
    assert np.isnan(evaluation.amplitude_covariance).all()
    factors = evaluation.covariance_factors
    assert factors is not None
    assert not factors.available
    assert np.isnan(factors.conditional).all()
