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
def no_apod_jit(
    dx: np.ndarray, r2: float, aq: float, phase: float = 0.0
) -> np.ndarray:
    """JIT-optimized non-apodized lineshape.

    Args:
        dx: Frequency offset array (Hz)
        r2: R2 relaxation rate (Hz)
        aq: Acquisition time (s)
        phase: Phase correction (degrees)

    Returns:
        NoApod profile values (real part)
    """
    n = len(dx)
    result = np.empty(n, dtype=np.float64)

    phase_rad = np.deg2rad(phase)
    cos_phase = np.cos(phase_rad)
    sin_phase = np.sin(phase_rad)

    for i in range(n):
        # z1 = aq * (1j * dx[i] + r2)
        z1_real = aq * r2
        z1_imag = aq * dx[i]

        # exp(-z1)
        exp_neg_z1_real = np.exp(-z1_real) * np.cos(-z1_imag)
        exp_neg_z1_imag = np.exp(-z1_real) * np.sin(-z1_imag)

        # 1 - exp(-z1)
        num_real = 1.0 - exp_neg_z1_real
        num_imag = -exp_neg_z1_imag

        # Division by z1 (complex)
        z1_mag_sq = z1_real * z1_real + z1_imag * z1_imag
        div_real = (num_real * z1_real + num_imag * z1_imag) / z1_mag_sq
        div_imag = (num_imag * z1_real - num_real * z1_imag) / z1_mag_sq

        # Multiply by aq
        spec_real = aq * div_real
        spec_imag = aq * div_imag

        # Apply phase correction and take real part
        result[i] = spec_real * cos_phase - spec_imag * sin_phase

    return result


@jit(nopython=True, cache=True)
def sp1_jit(
    dx: np.ndarray, r2: float, aq: float, end: float, off: float, phase: float = 0.0
) -> np.ndarray:
    """JIT-optimized SP1 (sine bell) lineshape.

    Args:
        dx: Frequency offset array (Hz)
        r2: R2 relaxation rate (Hz)
        aq: Acquisition time (s)
        end: End parameter for sine bell
        off: Offset parameter for sine bell
        phase: Phase correction (degrees)

    Returns:
        SP1 profile values (real part)
    """
    n = len(dx)
    result = np.empty(n, dtype=np.float64)

    phase_rad = np.deg2rad(phase)
    cos_phase = np.cos(phase_rad)
    sin_phase = np.sin(phase_rad)

    # Precompute f1 and f2 (imaginary)
    f1_imag = off * np.pi
    f2_imag = (end - off) * np.pi

    for i in range(n):
        # z1 = aq * (1j * dx[i] + r2)
        z1_real = aq * r2
        z1_imag = aq * dx[i]

        # exp(f2) where f2 = 1j * f2_imag
        exp_f2_real = np.cos(f2_imag)
        exp_f2_imag = np.sin(f2_imag)

        # exp(-f2)
        exp_neg_f2_real = np.cos(-f2_imag)
        exp_neg_f2_imag = np.sin(-f2_imag)

        # exp(z1)
        exp_z1_real = np.exp(z1_real) * np.cos(z1_imag)
        exp_z1_imag = np.exp(z1_real) * np.sin(z1_imag)

        # exp(-z1)
        exp_neg_z1_real = np.exp(-z1_real) * np.cos(-z1_imag)
        exp_neg_z1_imag = np.exp(-z1_real) * np.sin(-z1_imag)

        # exp(f1) where f1 = 1j * f1_imag
        exp_f1_real = np.cos(f1_imag)
        exp_f1_imag = np.sin(f1_imag)

        # exp(-f1)
        exp_neg_f1_real = np.cos(-f1_imag)
        exp_neg_f1_imag = np.sin(-f1_imag)

        # a1 numerator: (exp(f2) - exp(z1)) * exp(-z1 + f1)
        num1_real = exp_f2_real - exp_z1_real
        num1_imag = exp_f2_imag - exp_z1_imag

        # exp(-z1 + f1) = exp(-z1) * exp(f1)
        exp_neg_z1_f1_real = exp_neg_z1_real * exp_f1_real - exp_neg_z1_imag * exp_f1_imag
        exp_neg_z1_f1_imag = exp_neg_z1_real * exp_f1_imag + exp_neg_z1_imag * exp_f1_real

        a1_num_real = num1_real * exp_neg_z1_f1_real - num1_imag * exp_neg_z1_f1_imag
        a1_num_imag = num1_real * exp_neg_z1_f1_imag + num1_imag * exp_neg_z1_f1_real

        # a1 denominator: 2 * (z1 - f2), where f2 is purely imaginary
        denom1_real = 2.0 * z1_real
        denom1_imag = 2.0 * (z1_imag - f2_imag)
        denom1_mag_sq = denom1_real * denom1_real + denom1_imag * denom1_imag

        a1_real = (a1_num_real * denom1_real + a1_num_imag * denom1_imag) / denom1_mag_sq
        a1_imag = (a1_num_imag * denom1_real - a1_num_real * denom1_imag) / denom1_mag_sq

        # a2 numerator: (exp(z1) - exp(-f2)) * exp(-z1 - f1)
        num2_real = exp_z1_real - exp_neg_f2_real
        num2_imag = exp_z1_imag - exp_neg_f2_imag

        # exp(-z1 - f1) = exp(-z1) * exp(-f1)
        exp_neg_z1_neg_f1_real = (
            exp_neg_z1_real * exp_neg_f1_real - exp_neg_z1_imag * exp_neg_f1_imag
        )
        exp_neg_z1_neg_f1_imag = (
            exp_neg_z1_real * exp_neg_f1_imag + exp_neg_z1_imag * exp_neg_f1_real
        )

        a2_num_real = num2_real * exp_neg_z1_neg_f1_real - num2_imag * exp_neg_z1_neg_f1_imag
        a2_num_imag = num2_real * exp_neg_z1_neg_f1_imag + num2_imag * exp_neg_z1_neg_f1_real

        # a2 denominator: 2 * (z1 + f2)
        denom2_real = 2.0 * z1_real
        denom2_imag = 2.0 * (z1_imag + f2_imag)
        denom2_mag_sq = denom2_real * denom2_real + denom2_imag * denom2_imag

        a2_real = (a2_num_real * denom2_real + a2_num_imag * denom2_imag) / denom2_mag_sq
        a2_imag = (a2_num_imag * denom2_real - a2_num_real * denom2_imag) / denom2_mag_sq

        # spec = 1j * aq * (a1 + a2)
        sum_real = a1_real + a2_real
        sum_imag = a1_imag + a2_imag
        spec_real = -aq * sum_imag  # 1j * (a + bj) = -b + aj
        spec_imag = aq * sum_real

        # Apply phase correction and take real part
        result[i] = spec_real * cos_phase - spec_imag * sin_phase

    return result


@jit(nopython=True, cache=True)
def sp2_jit(
    dx: np.ndarray, r2: float, aq: float, end: float, off: float, phase: float = 0.0
) -> np.ndarray:
    """JIT-optimized SP2 (sine squared bell) lineshape.

    Args:
        dx: Frequency offset array (Hz)
        r2: R2 relaxation rate (Hz)
        aq: Acquisition time (s)
        end: End parameter for sine bell
        off: Offset parameter for sine bell
        phase: Phase correction (degrees)

    Returns:
        SP2 profile values (real part)
    """
    n = len(dx)
    result = np.empty(n, dtype=np.float64)

    phase_rad = np.deg2rad(phase)
    cos_phase = np.cos(phase_rad)
    sin_phase = np.sin(phase_rad)

    # Precompute f1 and f2 (imaginary)
    f1_imag = off * np.pi
    f2_imag = (end - off) * np.pi

    for i in range(n):
        # z1 = aq * (1j * dx[i] + r2)
        z1_real = aq * r2
        z1_imag = aq * dx[i]

        # exp(2*f2) where f2 is purely imaginary
        exp_2f2_real = np.cos(2.0 * f2_imag)
        exp_2f2_imag = np.sin(2.0 * f2_imag)

        # exp(-2*f2)
        exp_neg_2f2_real = np.cos(-2.0 * f2_imag)
        exp_neg_2f2_imag = np.sin(-2.0 * f2_imag)

        # exp(z1)
        exp_z1_real = np.exp(z1_real) * np.cos(z1_imag)
        exp_z1_imag = np.exp(z1_real) * np.sin(z1_imag)

        # exp(-z1)
        exp_neg_z1_real = np.exp(-z1_real) * np.cos(-z1_imag)
        exp_neg_z1_imag = np.exp(-z1_real) * np.sin(-z1_imag)

        # exp(2*f1) where f1 is purely imaginary
        exp_2f1_real = np.cos(2.0 * f1_imag)
        exp_2f1_imag = np.sin(2.0 * f1_imag)

        # exp(-2*f1)
        exp_neg_2f1_real = np.cos(-2.0 * f1_imag)
        exp_neg_2f1_imag = np.sin(-2.0 * f1_imag)

        # a1: (exp(2*f2) - exp(z1)) * exp(-z1 + 2*f1) / (4 * (z1 - 2*f2))
        num1_real = exp_2f2_real - exp_z1_real
        num1_imag = exp_2f2_imag - exp_z1_imag

        exp_neg_z1_2f1_real = exp_neg_z1_real * exp_2f1_real - exp_neg_z1_imag * exp_2f1_imag
        exp_neg_z1_2f1_imag = exp_neg_z1_real * exp_2f1_imag + exp_neg_z1_imag * exp_2f1_real

        a1_num_real = num1_real * exp_neg_z1_2f1_real - num1_imag * exp_neg_z1_2f1_imag
        a1_num_imag = num1_real * exp_neg_z1_2f1_imag + num1_imag * exp_neg_z1_2f1_real

        denom1_real = 4.0 * z1_real
        denom1_imag = 4.0 * (z1_imag - 2.0 * f2_imag)
        denom1_mag_sq = denom1_real * denom1_real + denom1_imag * denom1_imag

        a1_real = (a1_num_real * denom1_real + a1_num_imag * denom1_imag) / denom1_mag_sq
        a1_imag = (a1_num_imag * denom1_real - a1_num_real * denom1_imag) / denom1_mag_sq

        # a2: (exp(-2*f2) - exp(z1)) * exp(-z1 - 2*f1) / (4 * (z1 + 2*f2))
        num2_real = exp_neg_2f2_real - exp_z1_real
        num2_imag = exp_neg_2f2_imag - exp_z1_imag

        exp_neg_z1_neg_2f1_real = (
            exp_neg_z1_real * exp_neg_2f1_real - exp_neg_z1_imag * exp_neg_2f1_imag
        )
        exp_neg_z1_neg_2f1_imag = (
            exp_neg_z1_real * exp_neg_2f1_imag + exp_neg_z1_imag * exp_neg_2f1_real
        )

        a2_num_real = num2_real * exp_neg_z1_neg_2f1_real - num2_imag * exp_neg_z1_neg_2f1_imag
        a2_num_imag = num2_real * exp_neg_z1_neg_2f1_imag + num2_imag * exp_neg_z1_neg_2f1_real

        denom2_real = 4.0 * z1_real
        denom2_imag = 4.0 * (z1_imag + 2.0 * f2_imag)
        denom2_mag_sq = denom2_real * denom2_real + denom2_imag * denom2_imag

        a2_real = (a2_num_real * denom2_real + a2_num_imag * denom2_imag) / denom2_mag_sq
        a2_imag = (a2_num_imag * denom2_real - a2_num_real * denom2_imag) / denom2_mag_sq

        # a3: (1 - exp(-z1)) / (2 * z1)
        num3_real = 1.0 - exp_neg_z1_real
        num3_imag = -exp_neg_z1_imag

        denom3_real = 2.0 * z1_real
        denom3_imag = 2.0 * z1_imag
        denom3_mag_sq = denom3_real * denom3_real + denom3_imag * denom3_imag

        a3_real = (num3_real * denom3_real + num3_imag * denom3_imag) / denom3_mag_sq
        a3_imag = (num3_imag * denom3_real - num3_real * denom3_imag) / denom3_mag_sq

        # spec = aq * (a1 + a2 + a3)
        sum_real = a1_real + a2_real + a3_real
        sum_imag = a1_imag + a2_imag + a3_imag
        spec_real = aq * sum_real
        spec_imag = aq * sum_imag

        # Apply phase correction and take real part
        result[i] = spec_real * cos_phase - spec_imag * sin_phase

    return result


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
        "optimizations": [
            "gaussian_jit",
            "lorentzian_jit",
            "pvoigt_jit",
            "no_apod_jit",
            "sp1_jit",
            "sp2_jit",
        ]
        if HAS_NUMBA
        else ["numpy_vectorized"],
    }
