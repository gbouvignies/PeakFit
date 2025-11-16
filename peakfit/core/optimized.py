"""Optimized lineshape functions with optional Numba JIT compilation.

This module provides performance-optimized versions of core lineshape functions.
If numba is available, functions are JIT-compiled for better performance.
Otherwise, pure NumPy implementations are used.
"""

import numpy as np

from peakfit.typing import FloatArray

# Try to import numba for JIT compilation
try:
    from numba import jit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    # Fallback decorator that does nothing
    def jit(*args, **kwargs):  # noqa: ARG001
        def decorator(func):
            return func

        return decorator


@jit(nopython=True, cache=True, fastmath=True)
def gaussian_jit(dx: np.ndarray, fwhm: float) -> np.ndarray:
    """JIT-optimized Gaussian lineshape.

    Args:
        dx: Frequency offset array (Hz)
        fwhm: Full width at half maximum (Hz)

    Returns:
        Gaussian profile values
    """
    c = 4.0 * np.log(2.0) / (fwhm * fwhm)
    return np.exp(-dx * dx * c)


@jit(nopython=True, cache=True, fastmath=True)
def lorentzian_jit(dx: np.ndarray, fwhm: float) -> np.ndarray:
    """JIT-optimized Lorentzian lineshape.

    Args:
        dx: Frequency offset array (Hz)
        fwhm: Full width at half maximum (Hz)

    Returns:
        Lorentzian profile values
    """
    half_width_sq = (0.5 * fwhm) ** 2
    return half_width_sq / (dx * dx + half_width_sq)


@jit(nopython=True, cache=True, fastmath=True)
def pvoigt_jit(dx: np.ndarray, fwhm: float, eta: float) -> np.ndarray:
    """JIT-optimized Pseudo-Voigt lineshape.

    Args:
        dx: Frequency offset array (Hz)
        fwhm: Full width at half maximum (Hz)
        eta: Lorentzian fraction (0=pure Gaussian, 1=pure Lorentzian)

    Returns:
        Pseudo-Voigt profile values
    """
    c_gauss = 4.0 * np.log(2.0) / (fwhm * fwhm)
    gauss = np.exp(-dx * dx * c_gauss)

    half_width_sq = (0.5 * fwhm) ** 2
    lorentz = half_width_sq / (dx * dx + half_width_sq)

    return (1.0 - eta) * gauss + eta * lorentz


@jit(nopython=True, cache=True)
def calculate_lstsq_amplitude(shapes: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Fast linear least squares for amplitude calculation.

    Uses direct normal equations solution: A^T A x = A^T b

    Args:
        shapes: Shape matrix (n_peaks x n_points)
        data: Data vector (n_points,)

    Returns:
        Amplitude coefficients
    """
    # shapes.T @ shapes
    ata = np.dot(shapes, shapes.T)
    # shapes.T @ data
    atb = np.dot(shapes, data)
    # Solve using Cholesky decomposition would be ideal but not in numba
    # Use simple solve
    return np.linalg.solve(ata, atb)


def get_optimized_gaussian():
    """Get the optimized Gaussian function.

    Returns:
        Function: Gaussian lineshape function (JIT if available)
    """
    return gaussian_jit


def get_optimized_lorentzian():
    """Get the optimized Lorentzian function.

    Returns:
        Function: Lorentzian lineshape function (JIT if available)
    """
    return lorentzian_jit


def get_optimized_pvoigt():
    """Get the optimized Pseudo-Voigt function.

    Returns:
        Function: Pseudo-Voigt lineshape function (JIT if available)
    """
    return pvoigt_jit


# Vectorized operations for batch processing
def evaluate_peaks_batch(
    positions: list[np.ndarray],
    centers: np.ndarray,
    fwhms: np.ndarray,
    shape_func,
) -> np.ndarray:
    """Evaluate multiple peaks at once for better cache utilization.

    Args:
        positions: List of position arrays per dimension
        centers: Peak centers (n_peaks,)
        fwhms: Peak widths (n_peaks,)
        shape_func: Lineshape function to use

    Returns:
        Shape values for all peaks (n_peaks x n_points)
    """
    n_peaks = len(centers)
    n_points = len(positions[0])
    result = np.zeros((n_peaks, n_points))

    for i in range(n_peaks):
        dx = positions[0] - centers[i]
        result[i] = shape_func(dx, fwhms[i])

    return result


def check_numba_available() -> bool:
    """Check if Numba is available for optimization.

    Returns:
        bool: True if Numba is available
    """
    return HAS_NUMBA


def get_optimization_info() -> dict:
    """Get information about available optimizations.

    Returns:
        dict: Optimization status information
    """
    return {
        "numba_available": HAS_NUMBA,
        "jit_enabled": HAS_NUMBA,
        "optimizations": ["gaussian_jit", "lorentzian_jit", "pvoigt_jit"]
        if HAS_NUMBA
        else ["numpy_vectorized"],
    }
