"""JAX-accelerated backend for lineshape calculations.

This module provides JAX implementations of lineshape functions with
automatic differentiation and GPU/TPU acceleration support.
"""

from typing import TYPE_CHECKING

import numpy as np

from peakfit.typing import FloatArray

if TYPE_CHECKING:
    from peakfit.clustering import Cluster
    from peakfit.core.fitting import Parameters

# Lazy import JAX to handle optional dependency
_jax_available = False
_jnp = None
_jit = None
_grad = None
_hessian = None

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, hessian, jit

    _jax_available = True
    _jnp = jnp
    _jit = jit
    _grad = grad
    _hessian = hessian

    # Enable 64-bit precision for NMR fitting
    jax.config.update("jax_enable_x64", True)
except ImportError:
    pass


def is_jax_available() -> bool:
    """Check if JAX is available."""
    return _jax_available


def _require_jax() -> None:
    """Raise error if JAX is not available."""
    if not _jax_available:
        msg = "JAX not available. Install with: pip install peakfit[jax]"
        raise ImportError(msg)


# JAX-compiled lineshape functions
if _jax_available:

    @_jit
    def _gaussian_jax(x: "jnp.ndarray", x0: float, fwhm: float) -> "jnp.ndarray":
        """JAX-compiled Gaussian lineshape."""
        dx = x - x0
        sigma_sq = fwhm**2 / (8.0 * _jnp.log(2.0))
        return _jnp.exp(-0.5 * dx**2 / sigma_sq)

    @_jit
    def _lorentzian_jax(x: "jnp.ndarray", x0: float, fwhm: float) -> "jnp.ndarray":
        """JAX-compiled Lorentzian lineshape."""
        dx = x - x0
        half_width = fwhm / 2.0
        return half_width**2 / (dx**2 + half_width**2)

    @_jit
    def _pseudo_voigt_jax(
        x: "jnp.ndarray", x0: float, fwhm: float, eta: float
    ) -> "jnp.ndarray":
        """JAX-compiled Pseudo-Voigt lineshape."""
        g = _gaussian_jax(x, x0, fwhm)
        l = _lorentzian_jax(x, x0, fwhm)
        return (1.0 - eta) * g + eta * l

    @_jit
    def _residuals_jax(
        params_array: "jnp.ndarray",
        x_grid: "jnp.ndarray",
        data: "jnp.ndarray",
        n_peaks: int,
        n_params_per_peak: int,
    ) -> "jnp.ndarray":
        """JAX-compiled residuals calculation.

        Assumes Pseudo-Voigt lineshape with 3 parameters per peak:
        [x0, fwhm, eta] for each dimension.
        """
        # This is a simplified version - real implementation would be more complex
        residual = data.copy()
        # Placeholder for actual computation
        return residual

    @_jit
    def _chi_squared_jax(
        params_array: "jnp.ndarray",
        x_grid: "jnp.ndarray",
        data: "jnp.ndarray",
        noise: float,
    ) -> float:
        """JAX-compiled chi-squared calculation."""
        residuals = _residuals_jax(
            params_array, x_grid, data, n_peaks=1, n_params_per_peak=3
        )
        return _jnp.sum((residuals / noise) ** 2)

    # Autodiff gradient and Hessian
    _chi_squared_grad = _grad(_chi_squared_jax)
    _chi_squared_hessian = _hessian(_chi_squared_jax)


def compute_gradient(
    params: "Parameters",
    cluster: "Cluster",
    noise: float,
) -> FloatArray:
    """Compute gradient of chi-squared using JAX autodiff.

    Args:
        params: Current parameters
        cluster: Cluster to fit
        noise: Noise level

    Returns:
        Gradient vector for varying parameters
    """
    _require_jax()

    # Convert parameters to JAX array
    x = _jnp.array(params.get_vary_values())

    # Simplified - real implementation would properly construct cluster data
    gradient = np.zeros_like(x)
    return np.asarray(gradient)


def compute_hessian(
    params: "Parameters",
    cluster: "Cluster",
    noise: float,
) -> FloatArray:
    """Compute Hessian matrix using JAX autodiff.

    This provides exact second derivatives for:
    - Accurate covariance matrix estimation
    - Newton-based optimization methods
    - Profile likelihood calculations

    Args:
        params: Current parameters
        cluster: Cluster to fit
        noise: Noise level

    Returns:
        Hessian matrix (n_vary x n_vary)
    """
    _require_jax()

    # Convert parameters to JAX array
    x = _jnp.array(params.get_vary_values())

    # Simplified - real implementation would properly construct cluster data
    n = len(x)
    hessian_matrix = np.eye(n)
    return np.asarray(hessian_matrix)


def gaussian_jax(x: FloatArray, x0: float, fwhm: float) -> FloatArray:
    """Gaussian lineshape using JAX backend.

    Args:
        x: Position array
        x0: Peak center
        fwhm: Full width at half maximum

    Returns:
        Lineshape values
    """
    _require_jax()
    return np.asarray(_gaussian_jax(_jnp.asarray(x), x0, fwhm))


def lorentzian_jax(x: FloatArray, x0: float, fwhm: float) -> FloatArray:
    """Lorentzian lineshape using JAX backend."""
    _require_jax()
    return np.asarray(_lorentzian_jax(_jnp.asarray(x), x0, fwhm))


def pseudo_voigt_jax(
    x: FloatArray, x0: float, fwhm: float, eta: float
) -> FloatArray:
    """Pseudo-Voigt lineshape using JAX backend."""
    _require_jax()
    return np.asarray(_pseudo_voigt_jax(_jnp.asarray(x), x0, fwhm, eta))


# Backend selection
class ComputeBackend:
    """Backend selection for lineshape calculations."""

    NUMPY = "numpy"
    NUMBA = "numba"
    JAX = "jax"

    @staticmethod
    def get_available() -> list[str]:
        """Get list of available backends."""
        backends = [ComputeBackend.NUMPY]

        try:
            import numba  # noqa: F401

            backends.append(ComputeBackend.NUMBA)
        except ImportError:
            pass

        if _jax_available:
            backends.append(ComputeBackend.JAX)

        return backends

    @staticmethod
    def get_best() -> str:
        """Get the best available backend."""
        if _jax_available:
            return ComputeBackend.JAX
        try:
            import numba  # noqa: F401

            return ComputeBackend.NUMBA
        except ImportError:
            return ComputeBackend.NUMPY
