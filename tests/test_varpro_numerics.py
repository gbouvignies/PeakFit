"""Scientific consistency at the variable-projection seam."""

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from peakfit.engine.algorithms.linear_algebra import (
    calculate_amplitudes_with_uncertainty,
    factorize_shapes,
)
from peakfit.engine.algorithms.varpro import (
    VarProOptimizer,
    _calculate_errors,
    _inject_shared_aliases_into_map,
    fit_cluster,
    identifiable_covariance,
)
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import FitConfig
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.params_vector import FitParameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters
from peakfit.engine.lineshapes.create import create_shapes


def test_duplicate_shapes_cannot_fit_a_nonexistent_direction() -> None:
    grid = np.linspace(-3, 3, 51)
    shape = np.exp(-(grid**2))
    shapes = np.array([shape, shape])
    projection = factorize_shapes(shapes)
    # The second QR direction has no counterpart in the model.
    data = np.linalg.qr(shapes.T)[0][:, 1:2]
    amplitudes = projection.solve(data)
    reference = np.linalg.lstsq(shapes.T, data, rcond=None)[0]
    np.testing.assert_allclose(amplitudes, reference, atol=1e-12)
    np.testing.assert_allclose(
        data - projection.basis @ (projection.basis.T @ data), data - shapes.T @ amplitudes
    )
    np.testing.assert_allclose(projection.pinv, np.linalg.pinv(shapes.T), atol=1e-12)


def test_duplicate_amplitude_uncertainties_are_unavailable() -> None:
    shapes = np.ones((2, 5))
    amplitudes, errors, covariance = calculate_amplitudes_with_uncertainty(
        shapes, np.ones((5, 2)), 1.0
    )
    np.testing.assert_allclose(shapes.T @ amplitudes, 1.0)
    assert np.isnan(errors).all()
    assert np.isnan(covariance).all()


def _gaussian_group(separation: float, *, noise: float = 0.0):
    def axis(size, direct):
        return SpectralParameters(
            size=size,
            sw=500,
            obs=500,
            car=0,
            aq_time=0.01,
            apocode=0,
            apodq1=0,
            apodq2=0,
            apodq3=0,
            p180=False,
            direct=direct,
            ft=True,
        )

    spectra = Spectra(
        dic={},
        data=np.zeros((4, 101)),
        z_values=np.arange(4.0),
        params=[axis(4, False), axis(101, True)],
    )
    peaks = []
    for name, point in zip(("A", "B"), (50 - separation / 2, 50 + separation / 2), strict=True):
        center = float(spectra.spectral_params[0].pts2ppm(point))
        peak = Peak(
            name=name,
            positions=np.array([center]),
            shapes=create_shapes(
                spectra, FitConfig(lineshape="gaussian"), name, [center], ["gaussian"]
            ),
        )
        peak.set_cluster_id(1)
        peaks.append(peak)
    params = Parameters.from_peaks(peaks)
    cluster = Cluster(1, peaks, [np.arange(101)], np.zeros((101, 4)))
    amplitudes = np.array([[2.0, -1.0, 3.0, 0.5], [-1.0, 3.0, 0.5, 2.0]])
    cluster.data = cluster.evaluate(params).T @ amplitudes
    cluster.data += noise * np.random.default_rng(59).normal(size=cluster.data.shape)
    return cluster, params


def _optimizer(cluster, params, noise=1.0):
    return VarProOptimizer(
        cluster,
        params.get_vary_names(),
        params,
        FitParameters.from_parameters(params, cluster.peaks),
        noise,
    )


@pytest.mark.parametrize("separation", [0.0, 1e-10, 1.5, 6.0, 30.0])
def test_actual_model_and_projection_agree(separation) -> None:
    cluster, params = _gaussian_group(separation, noise=0.01)
    optimizer = _optimizer(cluster, params)
    x = params.get_vary_values()
    residual = optimizer.compute_residuals(x)
    shapes = cluster.evaluate(params)
    reference = np.linalg.lstsq(shapes.T, cluster.data, rcond=None)[0]
    np.testing.assert_allclose(residual, (cluster.data - shapes.T @ reference).ravel(), atol=3e-5)
    np.testing.assert_allclose(
        residual,
        (cluster.data - optimizer._shapes.T @ optimizer.get_optimized_amplitudes()).ravel(),
        atol=1e-12,
    )


@pytest.mark.parametrize("separation", [1.5, 6.0, 30.0])
def test_jacobian_differentiates_the_actual_objective(separation) -> None:
    cluster, params = _gaussian_group(separation, noise=0.01)
    optimizer = _optimizer(cluster, params, noise=0.01)
    x = params.get_vary_values()
    analytical = optimizer.compute_jacobian(x)
    steps = np.array([1e-7 if name.endswith(".cs") else 1e-3 for name in optimizer.names])
    numerical = approx_derivative(optimizer.compute_residuals, x, method="3-point", abs_step=steps)
    np.testing.assert_allclose(analytical, numerical, rtol=2e-5, atol=1e-5)
    residual = optimizer.compute_residuals(x)
    np.testing.assert_allclose(analytical.T @ residual, numerical.T @ residual, rtol=1e-4)


@pytest.mark.parametrize("separation", [1.5, 6.0, 30.0])
def test_identifiable_overlapping_profiles_recover_truth(separation) -> None:
    cluster, truth = _gaussian_group(separation)
    params = truth.copy(deep=True)
    for name, parameter in params.items():
        parameter.value += 0.0002 if name.endswith(".cs") else 0.5
    result = fit_cluster(params, cluster, noise=0.01)
    assert result.success
    for name in truth:
        assert result.params[name].value == pytest.approx(truth[name].value, abs=2e-6)
        assert np.isfinite(result.params[name].stderr)
    assert np.linalg.norm(result.residual) < 1e-6


def test_rank_loss_has_no_trustworthy_joint_nonlinear_covariance() -> None:
    cluster, params = _gaussian_group(0.0)
    optimizer = _optimizer(cluster, params)
    errors, amplitude_errors = _calculate_errors(optimizer, params.get_vary_values(), 1.0)
    assert np.isnan(errors).all()
    assert np.isnan(amplitude_errors).all()


@pytest.mark.parametrize("epsilon", [0.0, 1e-12])
def test_covariance_withholds_unidentifiable_directions(epsilon) -> None:
    jacobian = np.array([[1.0, 1.0], [0.0, epsilon], [0.0, 0.0]])
    assert np.isnan(identifiable_covariance(jacobian)).all()


def test_covariance_respects_parameter_units_and_known_noise() -> None:
    jacobian = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    scales = np.array([1e-9, 1e9])
    expected = np.linalg.inv(jacobian.T @ jacobian)
    np.testing.assert_allclose(identifiable_covariance(jacobian), expected)
    np.testing.assert_allclose(
        identifiable_covariance(jacobian * scales) * scales[:, None] * scales[None, :],
        expected,
    )
    cluster, params = _gaussian_group(6.0)
    small = _optimizer(cluster, params.copy(deep=True), 0.01)
    large = _optimizer(cluster, params.copy(deep=True), 0.02)
    errors, _ = _calculate_errors(small, params.get_vary_values(), 0.01)
    doubled, _ = _calculate_errors(large, params.get_vary_values(), 0.02)
    np.testing.assert_allclose(doubled, 2 * errors)


def test_active_bound_uncertainty_is_unavailable() -> None:
    cluster, params = _gaussian_group(6.0)
    name = params.get_vary_names()[0]
    params[name].min = params[name].value
    optimizer = _optimizer(cluster, params, 0.01)
    errors, _ = _calculate_errors(optimizer, params.get_vary_values(), 0.01)
    assert np.isnan(errors[0])


def test_shared_linewidth_uses_each_peaks_signed_amplitude() -> None:
    cluster, params = _gaussian_group(6.0, noise=0.01)
    aliases = {"B.F2.lw": "A.F2.lw"}
    # The synthetic single spectral dimension is labelled F2.
    assert "B.F2.lw" in params
    params["B.F2.lw"].vary = False
    fit_params = FitParameters.from_parameters(params, cluster.peaks)
    _inject_shared_aliases_into_map(fit_params, params, aliases)
    optimizer = VarProOptimizer(cluster, params.get_vary_names(), params, fit_params, 1.0, aliases)
    x = params.get_vary_values()
    jacobian = optimizer.compute_jacobian(x)
    steps = [1e-7 if name.endswith(".cs") else 1e-3 for name in optimizer.names]
    numerical = approx_derivative(optimizer.compute_residuals, x, method="3-point", abs_step=steps)
    np.testing.assert_allclose(jacobian, numerical, atol=1e-7, rtol=2e-5)
    result = fit_cluster(params.copy(deep=True), cluster, 0.01, shared_aliases=aliases)
    assert result.params["A.F2.lw"].stderr > 0
    assert result.params["B.F2.lw"].stderr == result.params["A.F2.lw"].stderr


def test_noisy_identifiable_synthetic_positions_have_calibrated_local_errors() -> None:
    covered = []
    for seed in range(20):
        cluster, truth = _gaussian_group(6.0)
        cluster.data += np.random.default_rng(seed).normal(scale=0.005, size=cluster.data.shape)
        result = fit_cluster(truth.copy(deep=True), cluster, 0.005)
        for name in truth:
            if name.endswith(".cs"):
                error = result.params[name].stderr
                assert np.isfinite(error)
                assert error > 0
                covered.append(abs(result.params[name].value - truth[name].value) < 2 * error)
    assert np.mean(covered) >= 0.85
