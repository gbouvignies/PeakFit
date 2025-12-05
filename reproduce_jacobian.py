import numpy as np

from peakfit.core.domain.cluster import Cluster
from peakfit.core.domain.peaks import Peak
from peakfit.core.fitting.computation import calculate_shape_heights
from peakfit.core.fitting.jacobian import compute_jacobian
from peakfit.core.fitting.optimizer import compute_residuals
from peakfit.core.fitting.parameters import Parameters
from peakfit.core.lineshapes import LorentzianEvaluator


# Mock Peak and Cluster for testing
class MockPeak(Peak):
    model_config = {"extra": "allow"}

    def __init__(self, name, x0, fwhm):
        # Initialize with dummy values to satisfy Pydantic
        super().__init__(name=name, positions=np.array([]), shapes=[])
        self.x0 = x0
        self.fwhm = fwhm
        self.evaluator = LorentzianEvaluator()

    def evaluate(self, grid, params):
        # Simple Lorentzian
        # grid is expected to be a list of arrays, we take the first one
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

        # d_dx is derivative w.r.t dx.
        # dx = x - x0. d(dx)/dx0 = -1.
        # d_model/dx0 = d_model/d_dx * d_dx/dx0 = d_dx * (-1) = -d_dx.

        mapped_derivs = {f"{self.name}.x0": -d_dx, f"{self.name}.fwhm": d_fwhm}
        return val, mapped_derivs


def setup_test_case():
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
    # positions should be a list of arrays
    cluster = Cluster(cluster_id=1, peaks=[peak], positions=[x], data=data)
    # cluster.corrected_data is a property, no need to set it. corrections are 0 by default.

    # Create Parameters
    params = Parameters()
    params.add("p1.x0", value=5.1, vary=True)  # Slightly off
    params.add("p1.fwhm", value=1.2, vary=True)  # Slightly off

    return params, cluster, x


def finite_difference_jacobian(params, cluster, noise=1.0, epsilon=1e-8):
    x0 = params.get_vary_values()
    names = params.get_vary_names()
    n_params = len(x0)
    n_points = len(cluster.positions[0])

    jac = np.zeros((n_points, n_params))

    f0 = compute_residuals(x0, names, params, cluster, noise)

    for i in range(n_params):
        x_plus = x0.copy()
        x_plus[i] += epsilon
        f_plus = compute_residuals(x_plus, names, params, cluster, noise)

        jac[:, i] = (f_plus - f0) / epsilon

    return jac


def main():
    params, cluster, x = setup_test_case()
    noise = 1.0

    x0 = params.get_vary_values()
    names = params.get_vary_names()
    lower, upper = params.get_vary_bounds()

    # Analytical Jacobian (Current)
    jac_analytical = compute_jacobian(x0, names, params, cluster, noise)

    # Check orthogonality
    shapes, amplitudes = calculate_shape_heights(params, cluster)
    if amplitudes.ndim == 1:
        amplitudes = amplitudes[:, np.newaxis]

    # Ensure data is reshaped if needed
    data = cluster.corrected_data
    if data.ndim == 1:
        data = data[:, np.newaxis]

    residuals = data - shapes.T @ amplitudes
    orthogonality = shapes @ residuals
    print(f"Orthogonality check (S^T r): max abs = {np.max(np.abs(orthogonality))}")

    # Finite Difference Jacobian
    jac_fd = finite_difference_jacobian(params, cluster, noise)

    print("Jacobian Comparison:")
    print(f"Shape: {jac_analytical.shape}")

    diff = np.abs(jac_analytical - jac_fd)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Max Difference: {max_diff}")
    print(f"Mean Difference: {mean_diff}")

    # Check specific columns
    for i, name in enumerate(names):
        col_diff = np.max(np.abs(jac_analytical[:, i] - jac_fd[:, i]))
        print(f"Param {name} max diff: {col_diff}")

    # Verify derivatives of the peak itself
    print("\nVerifying Peak Derivatives:")
    peak = cluster.peaks[0]
    p_x0 = params[f"{peak.name}.x0"].value
    p_fwhm = params[f"{peak.name}.fwhm"].value
    x = cluster.positions[0]

    val, derivs = peak.evaluate_derivatives([x], params)

    # FD for x0
    eps = 1e-8
    params[f"{peak.name}.x0"].value = p_x0 + eps
    val_plus = peak.evaluate([x], params)
    d_x0_fd = (val_plus - val) / eps
    params[f"{peak.name}.x0"].value = p_x0  # Reset

    print(f"d/dx0 max diff: {np.max(np.abs(derivs[f'{peak.name}.x0'] - d_x0_fd))}")

    # FD for fwhm
    params[f"{peak.name}.fwhm"].value = p_fwhm + eps
    val_plus = peak.evaluate([x], params)
    d_fwhm_fd = (val_plus - val) / eps
    params[f"{peak.name}.fwhm"].value = p_fwhm  # Reset

    print(f"d/dfwhm max diff: {np.max(np.abs(derivs[f'{peak.name}.fwhm'] - d_fwhm_fd))}")


if __name__ == "__main__":
    main()
