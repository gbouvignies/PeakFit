"""Numba-accelerated lineshape functions for NMR peak fitting.

Functional composition architecture:
- Core kernels: Pure mathematical lineshape functions
- Factories: Generate J-coupled and batch versions dynamically
- Registry: Single source of truth for all shape computations
"""

import numba as nb
import numpy as np

from peakfit.typing import FloatArray

# =============================================================================
# Core Kernels (inline for maximum Numba optimization)
# =============================================================================


@nb.njit(cache=True, fastmath=True, inline="always")
def _kernel_gaussian(dx: np.ndarray, fwhm: float) -> np.ndarray:
    """Gaussian lineshape kernel."""
    c = 4.0 * np.log(2.0) / (fwhm * fwhm)
    return np.exp(-dx * dx * c)


@nb.njit(cache=True, fastmath=True, inline="always")
def _kernel_lorentzian(dx: np.ndarray, fwhm: float) -> np.ndarray:
    """Lorentzian lineshape kernel."""
    hw2 = (0.5 * fwhm) ** 2
    return hw2 / (dx * dx + hw2)


@nb.njit(cache=True, fastmath=True, inline="always")
def _kernel_pvoigt(dx: np.ndarray, fwhm: float, eta: float) -> np.ndarray:
    """Pseudo-Voigt lineshape kernel."""
    c = 4.0 * np.log(2.0) / (fwhm * fwhm)
    hw2 = (0.5 * fwhm) ** 2
    gauss = np.exp(-dx * dx * c)
    lorentz = hw2 / (dx * dx + hw2)
    return (1.0 - eta) * gauss + eta * lorentz


@nb.njit(cache=True, fastmath=True, inline="always")
def _kernel_no_apod(dx: np.ndarray, r2: float, aq: float, phase: float) -> np.ndarray:
    """Non-apodized FID-based frequency-domain lineshape kernel."""
    phase_rad = phase * np.pi / 180.0
    result = np.empty_like(dx)
    for i in range(len(dx)):
        z = aq * (r2 + 1j * dx[i])
        spec = aq * (1.0 - np.exp(-z)) / z
        result[i] = (spec * np.exp(1j * phase_rad)).real
    return result


@nb.njit(cache=True, fastmath=True, inline="always")
def _kernel_sp1(
    dx: np.ndarray, r2: float, aq: float, end: float, off: float, phase: float
) -> np.ndarray:
    """SP1 apodization FID-based frequency-domain lineshape kernel."""
    phase_rad = phase * np.pi / 180.0
    f1 = 1j * off * np.pi
    f2 = 1j * (end - off) * np.pi
    result = np.empty_like(dx)
    for i in range(len(dx)):
        z = aq * (r2 + 1j * dx[i])
        a1 = (np.exp(f2) - np.exp(z)) * np.exp(-z + f1) / (2 * (z - f2))
        a2 = (np.exp(z) - np.exp(-f2)) * np.exp(-z - f1) / (2 * (z + f2))
        spec = 1j * aq * (a1 + a2)
        result[i] = (spec * np.exp(1j * phase_rad)).real
    return result


@nb.njit(cache=True, fastmath=True, inline="always")
def _kernel_sp2(
    dx: np.ndarray, r2: float, aq: float, end: float, off: float, phase: float
) -> np.ndarray:
    """SP2 apodization FID-based frequency-domain lineshape kernel."""
    phase_rad = phase * np.pi / 180.0
    f1 = 1j * off * np.pi
    f2 = 1j * (end - off) * np.pi
    result = np.empty_like(dx)
    for i in range(len(dx)):
        z = aq * (r2 + 1j * dx[i])
        a1 = (np.exp(2 * f2) - np.exp(z)) * np.exp(-z + 2 * f1) / (4 * (z - 2 * f2))
        a2 = (np.exp(-2 * f2) - np.exp(z)) * np.exp(-z - 2 * f1) / (4 * (z + 2 * f2))
        a3 = (1.0 - np.exp(-z)) / (2 * z)
        spec = aq * (a1 + a2 + a3)
        result[i] = (spec * np.exp(1j * phase_rad)).real
    return result


# =============================================================================
# J-Coupling: Explicit implementations (Numba doesn't support *args in closures)
# =============================================================================


@nb.njit(cache=True, fastmath=True, inline="always")
def _apply_jcoupling_gaussian(dx: np.ndarray, j_hz: float, fwhm: float) -> np.ndarray:
    """Apply J-coupling to gaussian shape."""
    if j_hz == 0.0:
        return _kernel_gaussian(dx, fwhm)

    offset = j_hz * np.pi
    plus = _kernel_gaussian(dx + offset, fwhm)
    minus = _kernel_gaussian(dx - offset, fwhm)

    tmp = np.empty(1, dtype=dx.dtype)
    tmp[0] = offset
    n_plus = _kernel_gaussian(tmp, fwhm)[0]
    tmp[0] = -offset
    n_minus = _kernel_gaussian(tmp, fwhm)[0]
    norm = n_plus + n_minus
    return (plus + minus) / norm


@nb.njit(cache=True, fastmath=True, inline="always")
def _apply_jcoupling_lorentzian(dx: np.ndarray, j_hz: float, fwhm: float) -> np.ndarray:
    """Apply J-coupling to lorentzian shape."""
    if j_hz == 0.0:
        return _kernel_lorentzian(dx, fwhm)

    offset = j_hz * np.pi
    plus = _kernel_lorentzian(dx + offset, fwhm)
    minus = _kernel_lorentzian(dx - offset, fwhm)

    tmp = np.empty(1, dtype=dx.dtype)
    tmp[0] = offset
    n_plus = _kernel_lorentzian(tmp, fwhm)[0]
    tmp[0] = -offset
    n_minus = _kernel_lorentzian(tmp, fwhm)[0]
    norm = n_plus + n_minus
    return (plus + minus) / norm


@nb.njit(cache=True, fastmath=True, inline="always")
def _apply_jcoupling_pvoigt(dx: np.ndarray, j_hz: float, fwhm: float, eta: float) -> np.ndarray:
    """Apply J-coupling to pvoigt shape."""
    if j_hz == 0.0:
        return _kernel_pvoigt(dx, fwhm, eta)

    offset = j_hz * np.pi
    plus = _kernel_pvoigt(dx + offset, fwhm, eta)
    minus = _kernel_pvoigt(dx - offset, fwhm, eta)

    tmp = np.empty(1, dtype=dx.dtype)
    tmp[0] = offset
    n_plus = _kernel_pvoigt(tmp, fwhm, eta)[0]
    tmp[0] = -offset
    n_minus = _kernel_pvoigt(tmp, fwhm, eta)[0]
    norm = n_plus + n_minus
    return (plus + minus) / norm


@nb.njit(cache=True, fastmath=True, inline="always")
def _apply_jcoupling_no_apod(
    dx: np.ndarray, j_hz: float, r2: float, aq: float, phase: float
) -> np.ndarray:
    """Apply J-coupling to no_apod shape."""
    if j_hz == 0.0:
        return _kernel_no_apod(dx, r2, aq, phase)

    offset = j_hz * np.pi
    plus = _kernel_no_apod(dx + offset, r2, aq, phase)
    minus = _kernel_no_apod(dx - offset, r2, aq, phase)

    tmp = np.empty(1, dtype=dx.dtype)
    tmp[0] = offset
    n_plus = _kernel_no_apod(tmp, r2, aq, phase)[0]
    tmp[0] = -offset
    n_minus = _kernel_no_apod(tmp, r2, aq, phase)[0]
    norm = n_plus + n_minus
    return (plus + minus) / norm


@nb.njit(cache=True, fastmath=True, inline="always")
def _apply_jcoupling_sp1(
    dx: np.ndarray, j_hz: float, r2: float, aq: float, end: float, off: float, phase: float
) -> np.ndarray:
    """Apply J-coupling to sp1 shape."""
    if j_hz == 0.0:
        return _kernel_sp1(dx, r2, aq, end, off, phase)

    offset = j_hz * np.pi
    plus = _kernel_sp1(dx + offset, r2, aq, end, off, phase)
    minus = _kernel_sp1(dx - offset, r2, aq, end, off, phase)

    tmp = np.empty(1, dtype=dx.dtype)
    tmp[0] = offset
    n_plus = _kernel_sp1(tmp, r2, aq, end, off, phase)[0]
    tmp[0] = -offset
    n_minus = _kernel_sp1(tmp, r2, aq, end, off, phase)[0]
    norm = n_plus + n_minus
    return (plus + minus) / norm


@nb.njit(cache=True, fastmath=True, inline="always")
def _apply_jcoupling_sp2(
    dx: np.ndarray, j_hz: float, r2: float, aq: float, end: float, off: float, phase: float
) -> np.ndarray:
    """Apply J-coupling to sp2 shape."""
    if j_hz == 0.0:
        return _kernel_sp2(dx, r2, aq, end, off, phase)

    offset = j_hz * np.pi
    plus = _kernel_sp2(dx + offset, r2, aq, end, off, phase)
    minus = _kernel_sp2(dx - offset, r2, aq, end, off, phase)

    tmp = np.empty(1, dtype=dx.dtype)
    tmp[0] = offset
    n_plus = _kernel_sp2(tmp, r2, aq, end, off, phase)[0]
    tmp[0] = -offset
    n_minus = _kernel_sp2(tmp, r2, aq, end, off, phase)[0]
    norm = n_plus + n_minus
    return (plus + minus) / norm


def make_batch_evaluator(j_coupled_func):
    """Factory for parallel batch evaluators.

    Returns a @njit(parallel=True) function with signature:
    (positions, centers, j_couplings, *param_arrays)

    Each element of param_arrays must be a 1D array with length == n_peaks.
    """

    @nb.njit(cache=True, fastmath=True, parallel=True)
    def batch(positions: np.ndarray, centers: np.ndarray, j_couplings: np.ndarray, *param_arrays):
        n_peaks = centers.shape[0]
        n_points = positions.shape[0]
        result = np.empty((n_peaks, n_points), dtype=np.float64)

        n_params = len(param_arrays)
        for i in nb.prange(n_peaks):
            dx = positions - centers[i]
            if n_params == 0:
                result[i] = j_coupled_func(dx, j_couplings[i])
            elif n_params == 1:
                result[i] = j_coupled_func(dx, j_couplings[i], param_arrays[0][i])
            elif n_params == 2:
                result[i] = j_coupled_func(
                    dx, j_couplings[i], param_arrays[0][i], param_arrays[1][i]
                )
            elif n_params == 3:
                result[i] = j_coupled_func(
                    dx,
                    j_couplings[i],
                    param_arrays[0][i],
                    param_arrays[1][i],
                    param_arrays[2][i],
                )
            elif n_params == 4:
                result[i] = j_coupled_func(
                    dx,
                    j_couplings[i],
                    param_arrays[0][i],
                    param_arrays[1][i],
                    param_arrays[2][i],
                    param_arrays[3][i],
                )
            else:
                # support up to 5 params explicitly
                result[i] = j_coupled_func(
                    dx,
                    j_couplings[i],
                    param_arrays[0][i],
                    param_arrays[1][i],
                    param_arrays[2][i],
                    param_arrays[3][i],
                    param_arrays[4][i],
                )

        return result

    return batch


# =============================================================================
# Registry and Batch Evaluator Factory
# =============================================================================

# Registry: shape name -> J-coupled kernel function
KERNEL_REGISTRY = {
    "gaussian": _apply_jcoupling_gaussian,
    "lorentzian": _apply_jcoupling_lorentzian,
    "pvoigt": _apply_jcoupling_pvoigt,
    "no_apod": _apply_jcoupling_no_apod,
    "sp1": _apply_jcoupling_sp1,
    "sp2": _apply_jcoupling_sp2,
}

# =============================================================================
# Public API functions (with 2D array compatibility for models)
# =============================================================================


def gaussian(dx: FloatArray, fwhm: float) -> FloatArray:
    """Gaussian lineshape.

    If dx is 2D (for external J-coupling from models), compute for each row.
    """
    dx_array = np.atleast_1d(dx)
    if dx_array.ndim == 2:
        return np.array([_kernel_gaussian(row, fwhm) for row in dx_array])
    return _kernel_gaussian(dx_array, fwhm)


def lorentzian(dx: FloatArray, fwhm: float) -> FloatArray:
    """Lorentzian lineshape.

    If dx is 2D (for external J-coupling from models), compute for each row.
    """
    dx_array = np.atleast_1d(dx)
    if dx_array.ndim == 2:
        return np.array([_kernel_lorentzian(row, fwhm) for row in dx_array])
    return _kernel_lorentzian(dx_array, fwhm)


def pvoigt(dx: FloatArray, fwhm: float, eta: float) -> FloatArray:
    """Pseudo-Voigt lineshape.

    If dx is 2D (for external J-coupling from models), compute for each row.
    """
    dx_array = np.atleast_1d(dx)
    if dx_array.ndim == 2:
        return np.array([_kernel_pvoigt(row, fwhm, eta) for row in dx_array])
    return _kernel_pvoigt(dx_array, fwhm, eta)


def no_apod(dx: FloatArray, r2: float, aq: float, phase: float = 0.0) -> FloatArray:
    """Non-apodized FID-based frequency-domain lineshape.

    If dx is 2D (for external J-coupling from models), compute for each row.
    """
    dx_array = np.atleast_1d(dx)
    if dx_array.ndim == 2:
        return np.array([_kernel_no_apod(row, r2, aq, phase) for row in dx_array])
    return _kernel_no_apod(dx_array, r2, aq, phase)


def sp1(
    dx: FloatArray, r2: float, aq: float, end: float, off: float, phase: float = 0.0
) -> FloatArray:
    """SP1 apodization FID-based frequency-domain lineshape.

    If dx is 2D (for external J-coupling from models), compute for each row.
    """
    dx_array = np.atleast_1d(dx)
    if dx_array.ndim == 2:
        return np.array([_kernel_sp1(row, r2, aq, end, off, phase) for row in dx_array])
    return _kernel_sp1(dx_array, r2, aq, end, off, phase)


def sp2(
    dx: FloatArray, r2: float, aq: float, end: float, off: float, phase: float = 0.0
) -> FloatArray:
    """SP2 apodization FID-based frequency-domain lineshape.

    If dx is 2D (for external J-coupling from models), compute for each row.
    """
    dx_array = np.atleast_1d(dx)
    if dx_array.ndim == 2:
        return np.array([_kernel_sp2(row, r2, aq, end, off, phase) for row in dx_array])
    return _kernel_sp2(dx_array, r2, aq, end, off, phase)


# =============================================================================
# Batch evaluation functions (via factories)
# =============================================================================

# Compiled batch evaluators (standardized signature inside wrappers)
_batch_gaussian = make_batch_evaluator(_apply_jcoupling_gaussian)
_batch_lorentzian = make_batch_evaluator(_apply_jcoupling_lorentzian)
_batch_pvoigt = make_batch_evaluator(_apply_jcoupling_pvoigt)
_batch_no_apod = make_batch_evaluator(_apply_jcoupling_no_apod)
_batch_sp1 = make_batch_evaluator(_apply_jcoupling_sp1)
_batch_sp2 = make_batch_evaluator(_apply_jcoupling_sp2)


def compute_all_gaussian_shapes(
    positions: np.ndarray,
    centers: np.ndarray,
    fwhms: np.ndarray,
    j_couplings: np.ndarray | None = None,
) -> np.ndarray:
    if j_couplings is None:
        j_couplings = np.zeros(centers.shape[0], dtype=np.float64)
    return _batch_gaussian(positions, centers, j_couplings, fwhms)


def compute_all_lorentzian_shapes(
    positions: np.ndarray,
    centers: np.ndarray,
    fwhms: np.ndarray,
    j_couplings: np.ndarray | None = None,
) -> np.ndarray:
    if j_couplings is None:
        j_couplings = np.zeros(centers.shape[0], dtype=np.float64)
    return _batch_lorentzian(positions, centers, j_couplings, fwhms)


def compute_all_pvoigt_shapes(
    positions: np.ndarray,
    centers: np.ndarray,
    fwhms: np.ndarray,
    etas: np.ndarray,
    j_couplings: np.ndarray | None = None,
) -> np.ndarray:
    if j_couplings is None:
        j_couplings = np.zeros(centers.shape[0], dtype=np.float64)
    return _batch_pvoigt(positions, centers, j_couplings, fwhms, etas)


def compute_all_no_apod_shapes(
    positions: np.ndarray,
    centers: np.ndarray,
    r2s: np.ndarray,
    aq: float | np.ndarray,
    phases: np.ndarray,
    j_couplings: np.ndarray | None = None,
) -> np.ndarray:
    if j_couplings is None:
        j_couplings = np.zeros(centers.shape[0], dtype=np.float64)
    if np.ndim(aq) == 0:
        aqs = np.full_like(r2s, float(aq))
    else:
        aqs = aq  # type: ignore[assignment]
    return _batch_no_apod(positions, centers, j_couplings, r2s, aqs, phases)


def compute_all_sp1_shapes(
    positions: np.ndarray,
    centers: np.ndarray,
    r2s: np.ndarray,
    aq: float | np.ndarray,
    end: float | np.ndarray,
    off: float | np.ndarray,
    phases: np.ndarray,
    j_couplings: np.ndarray | None = None,
) -> np.ndarray:
    if j_couplings is None:
        j_couplings = np.zeros(centers.shape[0], dtype=np.float64)
    if np.ndim(aq) == 0:
        aqs = np.full_like(r2s, float(aq))
    else:
        aqs = aq  # type: ignore[assignment]
    if np.ndim(end) == 0:
        ends = np.full_like(r2s, float(end))
    else:
        ends = end  # type: ignore[assignment]
    if np.ndim(off) == 0:
        offs = np.full_like(r2s, float(off))
    else:
        offs = off  # type: ignore[assignment]
    return _batch_sp1(positions, centers, j_couplings, r2s, aqs, ends, offs, phases)


def compute_all_sp2_shapes(
    positions: np.ndarray,
    centers: np.ndarray,
    r2s: np.ndarray,
    aq: float | np.ndarray,
    end: float | np.ndarray,
    off: float | np.ndarray,
    phases: np.ndarray,
    j_couplings: np.ndarray | None = None,
) -> np.ndarray:
    if j_couplings is None:
        j_couplings = np.zeros(centers.shape[0], dtype=np.float64)
    if np.ndim(aq) == 0:
        aqs = np.full_like(r2s, float(aq))
    else:
        aqs = aq  # type: ignore[assignment]
    if np.ndim(end) == 0:
        ends = np.full_like(r2s, float(end))
    else:
        ends = end  # type: ignore[assignment]
    if np.ndim(off) == 0:
        offs = np.full_like(r2s, float(off))
    else:
        offs = off  # type: ignore[assignment]
    return _batch_sp2(positions, centers, j_couplings, r2s, aqs, ends, offs, phases)


# =============================================================================
# Matrix operations for least-squares fitting
# =============================================================================


@nb.njit(cache=True, fastmath=True, parallel=True)
def compute_ata_symmetric(shapes: np.ndarray) -> np.ndarray:
    """Compute A^T A exploiting symmetry."""
    n_peaks = shapes.shape[0]
    result = np.empty((n_peaks, n_peaks), dtype=np.float64)

    for i in nb.prange(n_peaks):
        for j in range(i, n_peaks):
            val = np.dot(shapes[i], shapes[j])
            result[i, j] = val
            result[j, i] = val

    return result


@nb.njit(cache=True, fastmath=True)
def compute_atb(shapes: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Compute A^T b."""
    return shapes @ data


# =============================================================================
# Backward compatibility and utilities
# =============================================================================

# Legacy names (backward compatibility)
evaluate_apod_shape_no_apod = no_apod
evaluate_apod_shape_sp1 = sp1
evaluate_apod_shape_sp2 = sp2

# Aliases for old kernel names (backward compatibility)
_gaussian = _kernel_gaussian
_lorentzian = _kernel_lorentzian
_pvoigt = _kernel_pvoigt
_no_apod = _kernel_no_apod
_sp1 = _kernel_sp1
_sp2 = _kernel_sp2


@nb.njit(cache=True, fastmath=True)
def evaluate_apod_shape(
    dx_rads: FloatArray,
    r2: float,
    aq: float,
    end: float,
    off: float,
    phase: float,
    shape_type: int,
) -> FloatArray:
    """Runtime dispatch for apodization shapes."""
    if shape_type == 0:
        return _kernel_no_apod(dx_rads, r2, aq, phase)
    if shape_type == 1:
        return _kernel_sp1(dx_rads, r2, aq, end, off, phase)
    return _kernel_sp2(dx_rads, r2, aq, end, off, phase)


def warm_numba_cache() -> dict[str, float]:
    """Pre-compile all Numba functions."""
    import time

    times = {}
    dx = np.linspace(-100.0, 100.0, 512, dtype=np.float64)
    positions = dx.copy()
    centers = np.array([0.0, 10.0], dtype=np.float64)
    fwhms = np.array([20.0, 25.0], dtype=np.float64)
    etas = np.array([0.5, 0.5], dtype=np.float64)
    r2s = np.array([10.0, 15.0], dtype=np.float64)
    phases = np.array([0.0, 0.0], dtype=np.float64)
    j_couplings = np.array([0.0, 0.0], dtype=np.float64)
    aq = 0.05
    end = 2.0
    off = 0.5

    # Compile individual shapes
    start = time.perf_counter()
    _ = _kernel_gaussian(dx, 20.0)
    times["_kernel_gaussian"] = time.perf_counter() - start

    start = time.perf_counter()
    _ = _kernel_lorentzian(dx, 20.0)
    times["_kernel_lorentzian"] = time.perf_counter() - start

    start = time.perf_counter()
    _ = _kernel_pvoigt(dx, 20.0, 0.5)
    times["_kernel_pvoigt"] = time.perf_counter() - start

    start = time.perf_counter()
    _ = _kernel_no_apod(dx, 10.0, aq, 0.0)
    times["_kernel_no_apod"] = time.perf_counter() - start

    start = time.perf_counter()
    _ = _kernel_sp1(dx, 10.0, aq, end, off, 0.0)
    times["_kernel_sp1"] = time.perf_counter() - start

    start = time.perf_counter()
    _ = _kernel_sp2(dx, 10.0, aq, end, off, 0.0)
    times["_kernel_sp2"] = time.perf_counter() - start

    # Compile J-coupling wrappers
    start = time.perf_counter()
    _ = _apply_jcoupling_gaussian(dx, 0.0, 20.0)
    times["_apply_jcoupling_gaussian"] = time.perf_counter() - start

    start = time.perf_counter()
    _ = _apply_jcoupling_lorentzian(dx, 0.0, 20.0)
    times["_apply_jcoupling_lorentzian"] = time.perf_counter() - start

    start = time.perf_counter()
    _ = _apply_jcoupling_pvoigt(dx, 0.0, 20.0, 0.5)
    times["_apply_jcoupling_pvoigt"] = time.perf_counter() - start

    start = time.perf_counter()
    _ = _apply_jcoupling_no_apod(dx, 0.0, 10.0, aq, 0.0)
    times["_apply_jcoupling_no_apod"] = time.perf_counter() - start

    start = time.perf_counter()
    _ = _apply_jcoupling_sp1(dx, 0.0, 10.0, aq, end, off, 0.0)
    times["_apply_jcoupling_sp1"] = time.perf_counter() - start

    start = time.perf_counter()
    _ = _apply_jcoupling_sp2(dx, 0.0, 10.0, aq, end, off, 0.0)
    times["_apply_jcoupling_sp2"] = time.perf_counter() - start

    # Compile batch functions
    start = time.perf_counter()
    _ = compute_all_gaussian_shapes(positions, centers, fwhms, j_couplings)
    times["compute_all_gaussian_shapes"] = time.perf_counter() - start

    start = time.perf_counter()
    _ = compute_all_lorentzian_shapes(positions, centers, fwhms, j_couplings)
    times["compute_all_lorentzian_shapes"] = time.perf_counter() - start

    start = time.perf_counter()
    _ = compute_all_pvoigt_shapes(positions, centers, fwhms, etas, j_couplings)
    times["compute_all_pvoigt_shapes"] = time.perf_counter() - start

    start = time.perf_counter()
    _ = compute_all_no_apod_shapes(positions, centers, r2s, aq, phases, j_couplings)
    times["compute_all_no_apod_shapes"] = time.perf_counter() - start

    start = time.perf_counter()
    _ = compute_all_sp1_shapes(positions, centers, r2s, aq, end, off, phases, j_couplings)
    times["compute_all_sp1_shapes"] = time.perf_counter() - start

    start = time.perf_counter()
    _ = compute_all_sp2_shapes(positions, centers, r2s, aq, end, off, phases, j_couplings)
    times["compute_all_sp2_shapes"] = time.perf_counter() - start

    # Compile matrix operations
    rng = np.random.default_rng(1337)
    shapes_mat = rng.standard_normal((4, 512))
    data_vec = rng.standard_normal(512)

    start = time.perf_counter()
    _ = compute_ata_symmetric(shapes_mat)
    times["compute_ata_symmetric"] = time.perf_counter() - start

    start = time.perf_counter()
    _ = compute_atb(shapes_mat, data_vec)
    times["compute_atb"] = time.perf_counter() - start

    return times
