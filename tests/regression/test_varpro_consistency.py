"""Regression tests for VarProOptimizer consistency.

This module formalizes the logic from `tools/validation/reproduce_jacobian.py`
to ensure the Variable Projection algorithm remains numerically stable
and consistent with finite difference approximations.
"""

import numpy as np
import pytest

from peakfit.core.domain.cluster import Cluster
from peakfit.core.domain.peaks import Peak
from peakfit.core.fitting.computation import calculate_shape_heights
from peakfit.core.algorithms.varpro import VarProOptimizer
from peakfit.core.fitting.parameters import Parameters
from peakfit.core.lineshapes import LorentzianEvaluator


class MockPeak(Peak):
    """Minimal Peak subclass to test Jacobian reproduction."""

    model_config = {"extra": "allow"}

    def __init__(self, name, x0, fwhm):
        # Initialize with dummy values to satisfy Pydantic
        super().__init__(name=name, positions=np.array([]), shapes=[])
        self.x0 = x0
        self.fwhm = fwhm
        self.evaluator = LorentzianEvaluator()

    def evaluate(self, grid, params):
        # Grid is expected to be a list of arrays, we take the first one
        x = grid[0] if isinstance(grid, (list, tuple)) else grid

        p_x0 = params[f"{self.name}.x0"].value
        p_fwhm = params[f"{self.name}.fwhm"].value
        dx = x - p_x0
        return self.evaluator.evaluate(dx, p_fwhm)

    def evaluate_derivatives(self, grid, params):
        x = grid[0] if isinstance(grid, (list, tuple)) else grid

        p_x0 = params[f"{self.name}.x0"].value
        p_fwhm = params[f"{self.name}.fwhm"].value
        dx = x - p_x0
        # Unpack 4 values: val, d_dx, d_fwhm, d_j
        val, d_dx, d_fwhm, d_j = self.evaluator.evaluate_derivatives(dx, p_fwhm)

        # d_model/dx0 = -d_model/d_dx
        mapped_derivs = {f"{self.name}.x0": -d_dx, f"{self.name}.fwhm": d_fwhm}
        return val, mapped_derivs


@pytest.fixture
def varpro_test_case():
    """Setup a standard test case for VarPro validation."""
    # Create data
    x = np.linspace(0, 10, 100)

    # True parameters
    true_x0 = 5.0
    true_fwhm = 1.0
    true_amp = 10.0

    # Generate data
    peak = MockPeak("p1", true_x0, true_fwhm)

    # Manually compute shape
    shape = 1 / (1 + ((x - true_x0) / (0.5 * true_fwhm)) ** 2)
    data = true_amp * shape

    # Create Cluster
    cluster = Cluster(cluster_id=1, peaks=[peak], positions=[x], data=data)

    # Create Parameters (started slightly off)
    params = Parameters()
    params.add("p1.x0", value=5.1, vary=True)
    params.add("p1.fwhm", value=1.2, vary=True)

    return params, cluster, x


def finite_difference_jacobian(params, cluster, noise=1.0, epsilon=1e-8):
    """Compute Jacobian via finite differences."""
    x0 = params.get_vary_values()
    names = params.get_vary_names()
    n_params = len(x0)
    n_points = len(cluster.positions[0])

    jac = np.zeros((n_points, n_params))

    # Initialize optimizer for base residuals
    opt = VarProOptimizer(cluster, names, params, noise)
    f0 = opt.compute_residuals(x0)

    for i in range(n_params):
        x_plus = x0.copy()
        x_plus[i] += epsilon
        f_plus = opt.compute_residuals(x_plus)
        jac[:, i] = (f_plus - f0) / epsilon

    # Reset params
    for i, name in enumerate(names):
        params[name].value = x0[i]

    return jac


def test_orthogonality(varpro_test_case):
    """Verify that residuals are orthogonal to the subspace spanned by shapes.

    This is a core property of Variable Projection:
    The residual vector must be orthogonal to the basis functions at the optimum
    for linear parameters (projected out).
    """
    params, cluster, _ = varpro_test_case
    noise = 1.0

    # Initialize optimizer
    names = params.get_vary_names()
    x0 = params.get_vary_values()
    opt = VarProOptimizer(cluster, names, params, noise)

    # Run residual computation to populate state
    _ = opt.compute_residuals(x0)

    # Get shapes and amplitudes using the external computation helper for verification
    shapes, amplitudes = calculate_shape_heights(params, cluster)
    if amplitudes.ndim == 1:
        amplitudes = amplitudes[:, np.newaxis]

    data = cluster.corrected_data
    if data.ndim == 1:
        data = data[:, np.newaxis]

    residuals = data - shapes.T @ amplitudes

    # Check orthogonality: S^T @ r ≈ 0
    orthogonality = shapes @ residuals
    max_ortho = np.max(np.abs(orthogonality))

    # Expected to be very close to machine epsilon
    assert max_ortho < 1e-14, f"Orthogonality violation: {max_ortho}"


def test_jacobian_analytical_consistency(varpro_test_case):
    """Verify that the analytical Jacobian matches finite differences.

    This ensures that the VarPro correction terms (accounting for implicit
    amplitude dependence) are correctly implemented in VarProOptimizer.
    """
    params, cluster, _ = varpro_test_case
    noise = 1.0
    names = params.get_vary_names()
    x0 = params.get_vary_values()

    # 1. Analytical Jacobian
    opt = VarProOptimizer(cluster, names, params, noise)
    jac_analytical = opt.compute_jacobian(x0)

    # 2. Finite Difference Jacobian
    jac_fd = finite_difference_jacobian(params, cluster, noise)

    # 3. Compare
    diff = np.abs(jac_analytical - jac_fd)
    max_diff = np.max(diff)

    # Threshold: Finite differences are approximate (1e-6 to 1e-7 usually)
    # We expect close agreement.
    assert max_diff < 1e-6, f"Jacobian mismatch: {max_diff}"

    # Also check shapes match
    assert jac_analytical.shape == jac_fd.shape
