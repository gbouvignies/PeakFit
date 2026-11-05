"""Sandwich covariances under correlated noise: docs/correlated-noise.md §5.

R = I is the exact regression test of today's σ-only errors. The statistical
validation with PeakFit's fitting engine is ``tools/validate_sandwich_errors.py``.
"""

import importlib
from pathlib import Path

import numpy as np
import pytest

from peakfit.engine.algorithms.linear_algebra import (
    calculate_amplitudes_with_uncertainty,
    correlated_amplitude_covariance,
    factorize_shapes,
)
from peakfit.engine.algorithms.noise_correlation import (
    AxisProcessing,
    NoiseModel,
    cluster_correlation,
    not_modelled,
)
from peakfit.engine.algorithms.varpro import (
    VarProOptimizer,
    _calculate_errors,
    identifiable_covariance,
)
from peakfit.engine.domain.config import FitConfig, PeakFitConfig
from peakfit.engine.scientific.sampling import sampling_problem
from peakfit.fit.pipeline import run_pipeline
from peakfit.io.readers.spectrum import read_noise_model


@pytest.fixture(scope="module")
def tool():
    path = str(Path(__file__).resolve().parents[1] / "tools")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(path)
        yield importlib.import_module("validate_sandwich_errors")


def _correlation(points: int) -> np.ndarray:
    axis = AxisProcessing(24, 96, "sine_bell", offset=0.4, end=0.98, power=2.0)
    return cluster_correlation(NoiseModel(axes=(axis,)), [np.arange(points)])


def _shapes(points: int, centres, width: float = 3.0) -> np.ndarray:
    x = np.arange(points)
    return np.array([np.exp(-0.5 * ((x - c) / width) ** 2) for c in centres])


def test_r_equal_i_reproduces_todays_amplitude_covariance():
    projection = factorize_shapes(_shapes(40, [15.0, 21.0, 27.0]))
    sandwich = correlated_amplitude_covariance(projection, np.eye(40))
    np.testing.assert_allclose(sandwich, projection.covariance, rtol=1e-12, atol=0)
    _, errors, _ = calculate_amplitudes_with_uncertainty(
        _shapes(40, [15.0, 21.0, 27.0]), np.zeros((40, 2)), 0.3
    )
    np.testing.assert_allclose(0.3 * np.sqrt(np.diag(sandwich)), errors, rtol=1e-12)


def test_the_amplitude_sandwich_is_g_inverse_phi_r_phi_t_g_inverse():
    shapes = _shapes(40, [15.0, 21.0, 27.0])
    correlation = _correlation(40)
    inverse = np.linalg.inv(shapes @ shapes.T)
    expected = inverse @ shapes @ correlation @ shapes.T @ inverse
    sandwich = correlated_amplitude_covariance(factorize_shapes(shapes), correlation)
    np.testing.assert_allclose(sandwich, expected, rtol=1e-9)


@pytest.mark.parametrize("second", [20.0, 20.0 + 1e-9])  # duplicated, then weak
def test_the_amplitude_sandwich_is_withheld_exactly_where_todays_is(second):
    projection = factorize_shapes(_shapes(40, [20.0, second]))
    assert np.isnan(projection.covariance).all()
    assert np.isnan(correlated_amplitude_covariance(projection, _correlation(40))).all()


@pytest.fixture(scope="module")
def fitted(tool):
    """An overlapping 2D pair with processed noise, fitted by PeakFit's pipeline."""
    spectra, cluster, truth, _ = tool.build("overlapping pair", np.random.default_rng(7))
    config = PeakFitConfig(fitting=FitConfig(lineshape="sp2"))
    completed = run_pipeline(
        config, [cluster], tool.SIGMA, truth.copy(), noise_model=not_modelled("synthetic spectrum")
    ).completed_fit
    problem = sampling_problem(completed, 1)
    optimizer = VarProOptimizer(
        problem.cluster, list(problem.names), problem.parameters, tool.SIGMA
    )
    model = read_noise_model(spectra.dic, spectra.data.shape)
    correlation = cluster_correlation(model, problem.cluster.grid_indices)
    return problem, optimizer, correlation


def test_r_equal_i_reproduces_todays_nonlinear_errors(fitted):
    problem, optimizer, correlation = fitted
    jacobian = optimizer.compute_jacobian(problem.start)
    identity = np.eye(correlation.shape[0])
    np.testing.assert_allclose(
        identifiable_covariance(jacobian, identity), identifiable_covariance(jacobian), rtol=1e-10
    )
    published = np.array([problem.parameters[n].stderr for n in problem.names])
    np.testing.assert_allclose(
        _calculate_errors(optimizer, problem.start, identity), published, rtol=1e-10
    )


def test_the_nonlinear_sandwich_is_j_plus_r_kron_i_j_plus_t(fitted):
    problem, optimizer, correlation = fitted
    jacobian = optimizer.compute_jacobian(problem.start)
    planes = jacobian.shape[0] // correlation.shape[0]
    pseudoinverse = np.linalg.pinv(jacobian)
    # Residual rows are ordered (point, plane), the plane fastest.
    expected = pseudoinverse @ np.kron(correlation, np.eye(planes)) @ pseudoinverse.T
    np.testing.assert_allclose(identifiable_covariance(jacobian, correlation), expected, rtol=1e-8)
    # The processed noise makes every nonlinear error larger here.
    assert np.all(
        np.diag(identifiable_covariance(jacobian, correlation))
        > np.diag(identifiable_covariance(jacobian))
    )


def test_the_nonlinear_sandwich_is_withheld_exactly_where_todays_is(fitted):
    problem, optimizer, correlation = fitted
    jacobian = optimizer.compute_jacobian(problem.start)
    collinear = jacobian.copy()
    collinear[:, 1] = collinear[:, 0]
    assert np.isnan(identifiable_covariance(collinear)).all()
    assert np.isnan(identifiable_covariance(collinear, correlation)).all()
