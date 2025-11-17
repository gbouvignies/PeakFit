"""JAX-accelerated fitting with automatic differentiation.

This module provides JAX-based residual computation and exact gradient
calculation for NMR peak fitting. It enables:
- GPU-accelerated residual computation
- Exact gradients via autodiff (not finite differences)
- Newton-based optimization with exact Hessians

The key insight is that JAX can provide analytical gradients to scipy.optimize,
combining JAX's autodiff with scipy's robust bounded optimization algorithms.
"""

from typing import TYPE_CHECKING, Any

import numpy as np

from peakfit.typing import FloatArray

if TYPE_CHECKING:
    from peakfit.clustering import Cluster
    from peakfit.core.fitting import Parameters

# Lazy import JAX
_jax_available = False
_jnp = None
_jit = None
_jacobian = None
_value_and_grad = None

try:
    import jax
    import jax.numpy as jnp
    from jax import jacobian, jit, value_and_grad

    _jax_available = True
    _jnp = jnp
    _jit = jit
    _jacobian = jacobian
    _value_and_grad = value_and_grad

    # Enable 64-bit precision for NMR fitting
    jax.config.update("jax_enable_x64", True)
except ImportError:
    pass


def is_jax_fitting_available() -> bool:
    """Check if JAX fitting is available."""
    return _jax_available


def _require_jax() -> None:
    """Raise error if JAX is not available."""
    if not _jax_available:
        msg = "JAX not available. Install with: pip install peakfit[jax]"
        raise ImportError(msg)


# JAX-compiled lineshape functions for residual computation
if _jax_available:

    @_jit
    def _gaussian_jax_pure(dx: "jnp.ndarray", fwhm: "jnp.ndarray") -> "jnp.ndarray":
        """JAX Gaussian lineshape."""
        c = 4.0 * _jnp.log(2.0) / (fwhm * fwhm)
        return _jnp.exp(-dx * dx * c)

    @_jit
    def _lorentzian_jax_pure(dx: "jnp.ndarray", fwhm: "jnp.ndarray") -> "jnp.ndarray":
        """JAX Lorentzian lineshape."""
        half_width_sq = (0.5 * fwhm) ** 2
        return half_width_sq / (dx * dx + half_width_sq)

    @_jit
    def _pvoigt_jax_pure(
        dx: "jnp.ndarray", fwhm: "jnp.ndarray", eta: "jnp.ndarray"
    ) -> "jnp.ndarray":
        """JAX Pseudo-Voigt lineshape."""
        g = _gaussian_jax_pure(dx, fwhm)
        l = _lorentzian_jax_pure(dx, fwhm)
        return (1.0 - eta) * g + eta * l

    @_jit
    def _compute_pvoigt_shapes(
        params_array: "jnp.ndarray",
        positions: "jnp.ndarray",
        centers: "jnp.ndarray",
        n_peaks: int,
        n_dims: int,
    ) -> "jnp.ndarray":
        """Compute all peak shapes for Pseudo-Voigt lineshape.

        Args:
            params_array: Flat array of parameters
                          For PVoigt: [x0_1, fwhm_1, eta_1, ..., x0_n, fwhm_n, eta_n]
            positions: Grid positions, shape (n_dims, n_points)
            centers: Initial peak centers, shape (n_peaks, n_dims)
            n_peaks: Number of peaks
            n_dims: Number of spectral dimensions

        Returns:
            Shape array (n_peaks, n_points)
        """
        n_params_per_peak = 3 * n_dims  # x0, fwhm, eta for each dimension
        n_points = positions.shape[1]

        shapes = _jnp.ones((n_peaks, n_points))

        for peak_idx in range(n_peaks):
            peak_start = peak_idx * n_params_per_peak

            for dim in range(n_dims):
                dim_start = peak_start + dim * 3
                x0 = params_array[dim_start]  # position offset from center
                fwhm = params_array[dim_start + 1]
                eta = params_array[dim_start + 2]

                # Calculate frequency offset from peak center
                center = centers[peak_idx, dim]
                dx = positions[dim] - (center + x0)

                # Compute lineshape for this dimension
                shape_1d = _pvoigt_jax_pure(dx, fwhm, eta)

                # Multiply dimensions (separable product)
                shapes = shapes.at[peak_idx].set(shapes[peak_idx] * shape_1d)

        return shapes

    @_jit
    def _compute_amplitudes_jax(
        shapes: "jnp.ndarray", data: "jnp.ndarray"
    ) -> "jnp.ndarray":
        """Compute optimal amplitudes via linear least squares.

        Args:
            shapes: Peak shapes (n_peaks, n_points)
            data: Data to fit (n_planes, n_points) or (n_points,)

        Returns:
            Optimal amplitudes
        """
        # Solve shapes.T @ amplitudes = data
        return _jnp.linalg.lstsq(shapes.T, data, rcond=None)[0]

    @_jit
    def _compute_residuals_jax(
        params_array: "jnp.ndarray",
        positions: "jnp.ndarray",
        centers: "jnp.ndarray",
        data: "jnp.ndarray",
        noise: float,
        n_peaks: int,
        n_dims: int,
    ) -> "jnp.ndarray":
        """Compute residuals using JAX.

        Args:
            params_array: Flat parameter array
            positions: Grid positions
            centers: Peak centers
            data: Data to fit
            noise: Noise level
            n_peaks: Number of peaks
            n_dims: Number of dimensions

        Returns:
            Flat residual array
        """
        shapes = _compute_pvoigt_shapes(params_array, positions, centers, n_peaks, n_dims)
        amplitudes = _compute_amplitudes_jax(shapes, data)
        residuals = (data - shapes.T @ amplitudes) / noise
        return residuals.ravel()

    @_jit
    def _compute_chi_squared_jax(
        params_array: "jnp.ndarray",
        positions: "jnp.ndarray",
        centers: "jnp.ndarray",
        data: "jnp.ndarray",
        noise: float,
        n_peaks: int,
        n_dims: int,
    ) -> float:
        """Compute chi-squared statistic."""
        residuals = _compute_residuals_jax(
            params_array, positions, centers, data, noise, n_peaks, n_dims
        )
        return _jnp.sum(residuals**2)

    # Create JIT-compiled Jacobian function
    _residuals_jacobian = _jit(_jacobian(_compute_residuals_jax, argnums=0))


def create_jax_residual_function(
    cluster: "Cluster",
    noise: float,
    param_indices: dict[str, int],
) -> tuple[Any, Any]:
    """Create JAX-accelerated residual and Jacobian functions.

    This creates closures over cluster data for efficient repeated evaluation
    during optimization.

    Args:
        cluster: Cluster to fit
        noise: Noise level
        param_indices: Mapping from parameter names to array indices

    Returns:
        Tuple of (residual_func, jacobian_func) that take parameter array
        and return residuals and Jacobian matrix
    """
    _require_jax()

    # Extract cluster data as JAX arrays
    positions = _jnp.array(np.array(cluster.positions))
    data = _jnp.array(cluster.corrected_data)
    n_peaks = len(cluster.peaks)
    n_dims = len(cluster.positions)

    # Extract centers from peaks
    centers = _jnp.array(np.array([peak.positions for peak in cluster.peaks]))

    def residual_func(params_array: FloatArray) -> FloatArray:
        """Compute residuals."""
        params_jax = _jnp.array(params_array)
        residuals = _compute_residuals_jax(
            params_jax, positions, centers, data, noise, n_peaks, n_dims
        )
        return np.asarray(residuals)

    def jacobian_func(params_array: FloatArray) -> FloatArray:
        """Compute Jacobian of residuals w.r.t. parameters."""
        params_jax = _jnp.array(params_array)
        jac = _residuals_jacobian(
            params_jax, positions, centers, data, noise, n_peaks, n_dims
        )
        return np.asarray(jac)

    return residual_func, jacobian_func


def fit_with_jax_gradients(
    params: "Parameters",
    cluster: "Cluster",
    noise: float,
    max_nfev: int = 1000,
    ftol: float = 1e-8,
    xtol: float = 1e-8,
    gtol: float = 1e-8,
    verbose: int = 0,
) -> dict[str, Any]:
    """Fit cluster using JAX autodiff gradients.

    This provides exact analytical gradients to scipy.optimize.least_squares,
    which can improve convergence and accuracy compared to finite differences.

    Args:
        params: Initial parameters
        cluster: Cluster to fit
        noise: Noise level
        max_nfev: Maximum function evaluations
        ftol: Function tolerance
        xtol: Parameter tolerance
        gtol: Gradient tolerance
        verbose: Verbosity level

    Returns:
        Dictionary with optimization results
    """
    from scipy.optimize import least_squares

    _require_jax()

    # Get varying parameter info
    vary_names = params.get_vary_names()
    x0 = np.array(params.get_vary_values())
    bounds_lower = np.array([params[name].min for name in vary_names])
    bounds_upper = np.array([params[name].max for name in vary_names])

    # Create parameter index mapping
    param_indices = {name: i for i, name in enumerate(vary_names)}

    # Create JAX functions
    residual_func, jacobian_func = create_jax_residual_function(
        cluster, noise, param_indices
    )

    # Run optimization with analytical Jacobian
    result = least_squares(
        residual_func,
        x0,
        jac=jacobian_func,  # Use JAX autodiff Jacobian
        bounds=(bounds_lower, bounds_upper),
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        max_nfev=max_nfev,
        verbose=verbose,
    )

    # Update parameters with optimized values
    for i, name in enumerate(vary_names):
        params[name].value = result.x[i]

    return {
        "success": result.success,
        "cost": result.cost,
        "optimality": result.optimality,
        "nfev": result.nfev,
        "njev": result.njev if hasattr(result, "njev") else 0,
        "x": result.x,
        "message": result.message,
    }


def compute_hessian_at_minimum(
    params: "Parameters",
    cluster: "Cluster",
    noise: float,
) -> FloatArray:
    """Compute Hessian of chi-squared at current parameter values.

    The Hessian at the minimum can be used to estimate parameter uncertainties
    via the covariance matrix: Cov ≈ 2 * (Hessian)^(-1)

    Args:
        params: Current parameters (should be at minimum)
        cluster: Cluster being fitted
        noise: Noise level

    Returns:
        Hessian matrix (n_params x n_params)
    """
    _require_jax()

    from jax import hessian

    # Get current parameter values
    vary_names = params.get_vary_names()
    x = _jnp.array(params.get_vary_values())

    # Extract cluster data
    positions = _jnp.array(np.array(cluster.positions))
    data = _jnp.array(cluster.corrected_data)
    n_peaks = len(cluster.peaks)
    n_dims = len(cluster.positions)
    centers = _jnp.array(np.array([peak.positions for peak in cluster.peaks]))

    # Compute Hessian of chi-squared
    hess_func = hessian(_compute_chi_squared_jax)
    hess_matrix = hess_func(x, positions, centers, data, noise, n_peaks, n_dims)

    return np.asarray(hess_matrix)


def estimate_parameter_errors(
    params: "Parameters",
    cluster: "Cluster",
    noise: float,
) -> dict[str, float]:
    """Estimate parameter uncertainties using JAX Hessian.

    Uses the Hessian of chi-squared at the minimum to compute
    parameter standard errors.

    Args:
        params: Fitted parameters
        cluster: Cluster that was fitted
        noise: Noise level

    Returns:
        Dictionary mapping parameter names to standard errors
    """
    _require_jax()

    hessian_matrix = compute_hessian_at_minimum(params, cluster, noise)

    try:
        # Covariance matrix is approximately 2 * H^(-1)
        covariance = 2.0 * np.linalg.inv(hessian_matrix)
        # Standard errors are square roots of diagonal elements
        std_errors = np.sqrt(np.diag(covariance))

        vary_names = params.get_vary_names()
        return {name: std_errors[i] for i, name in enumerate(vary_names)}
    except np.linalg.LinAlgError:
        # Singular Hessian - return NaN for all errors
        vary_names = params.get_vary_names()
        return {name: np.nan for name in vary_names}
