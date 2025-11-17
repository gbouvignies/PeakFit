"""Backend registry for computation backends (NumPy, Numba, JAX).

This module provides a centralized way to select and use different computational
backends for lineshape calculations and fitting.
"""

from typing import Any

import numpy as np

from peakfit.typing import FloatArray

# Global backend state
_current_backend = "numpy"
_backend_functions: dict[str, dict[str, Any]] = {}


def set_backend(backend: str) -> None:
    """Set the global computational backend.

    Args:
        backend: One of "numpy", "numba", or "jax"

    Raises:
        ValueError: If backend is not available
    """
    global _current_backend

    available = get_available_backends()
    if backend not in available:
        msg = f"Backend '{backend}' not available. Available: {available}"
        raise ValueError(msg)

    _current_backend = backend
    _initialize_backend_functions(backend)


def get_backend() -> str:
    """Get current backend name."""
    return _current_backend


def get_available_backends() -> list[str]:
    """Get list of available backends."""
    backends = ["numpy"]

    try:
        import numba  # noqa: F401

        backends.append("numba")
    except ImportError:
        pass

    try:
        import jax  # noqa: F401

        backends.append("jax")
    except ImportError:
        pass

    return backends


def get_best_backend() -> str:
    """Get the best available backend.

    Prefers Numba for CPU-only execution (better performance with lower overhead).
    Only prefers JAX when mature GPU/TPU acceleration is available (CUDA, not Metal).
    """
    available = get_available_backends()

    # Check if JAX has stable GPU/TPU available (CUDA only, not Metal which is experimental)
    if "jax" in available:
        try:
            import jax

            devices = jax.devices()
            # Only prefer JAX if we have CUDA GPUs (not Metal which is experimental)
            has_cuda = any("cuda" in str(d).lower() for d in devices)
            if has_cuda:
                # Verify JAX CUDA actually works with a simple operation
                try:
                    import jax.numpy as jnp

                    test_result = jnp.array([1.0]) + jnp.array([1.0])
                    _ = float(test_result[0])  # Force computation
                    return "jax"
                except Exception:
                    pass  # JAX CUDA not working, fall through
        except Exception:
            pass

    # Prefer Numba for CPU-only execution (better performance, lower overhead)
    if "numba" in available:
        return "numba"

    # JAX on CPU is still faster than pure NumPy for some operations
    # but avoid JAX Metal (experimental) on macOS
    if "jax" in available:
        try:
            import jax

            devices = jax.devices()
            # Only use JAX CPU if no Metal devices (which are experimental)
            has_metal = any("metal" in str(d).lower() for d in devices)
            if not has_metal:
                return "jax"
        except Exception:
            pass

    return "numpy"


def auto_select_backend() -> str:
    """Automatically select and set the best backend."""
    backend = get_best_backend()
    set_backend(backend)
    return backend


def _initialize_backend_functions(backend: str) -> None:
    """Initialize backend-specific functions."""
    global _backend_functions

    if backend == "numpy":
        _backend_functions = {
            "gaussian": _gaussian_numpy,
            "lorentzian": _lorentzian_numpy,
            "pvoigt": _pvoigt_numpy,
            "no_apod": _no_apod_numpy,
            "sp1": _sp1_numpy,
            "sp2": _sp2_numpy,
        }
    elif backend == "numba":
        from peakfit.core.optimized import (
            gaussian_jit,
            lorentzian_jit,
            no_apod_jit,
            pvoigt_jit,
            sp1_jit,
            sp2_jit,
        )

        _backend_functions = {
            "gaussian": gaussian_jit,
            "lorentzian": lorentzian_jit,
            "pvoigt": pvoigt_jit,
            "no_apod": no_apod_jit,
            "sp1": sp1_jit,
            "sp2": sp2_jit,
        }
    elif backend == "jax":
        from peakfit.core.jax_backend import (
            gaussian_jax,
            lorentzian_jax,
            no_apod_jax,
            pseudo_voigt_jax,
            sp1_jax,
            sp2_jax,
        )

        _backend_functions = {
            "gaussian": gaussian_jax,
            "lorentzian": lorentzian_jax,
            "pvoigt": pseudo_voigt_jax,
            "no_apod": no_apod_jax,
            "sp1": sp1_jax,
            "sp2": sp2_jax,
        }


# NumPy implementations (fallback)
def _gaussian_numpy(dx: FloatArray, fwhm: float) -> FloatArray:
    """Pure NumPy Gaussian lineshape."""
    return np.exp(-(dx**2) * 4 * np.log(2) / (fwhm**2))


def _lorentzian_numpy(dx: FloatArray, fwhm: float) -> FloatArray:
    """Pure NumPy Lorentzian lineshape."""
    return (0.5 * fwhm) ** 2 / (dx**2 + (0.5 * fwhm) ** 2)


def _pvoigt_numpy(dx: FloatArray, fwhm: float, eta: float) -> FloatArray:
    """Pure NumPy Pseudo-Voigt lineshape."""
    return (1.0 - eta) * _gaussian_numpy(dx, fwhm) + eta * _lorentzian_numpy(dx, fwhm)


def _no_apod_numpy(
    dx: FloatArray, r2: float, aq: float, phase: float = 0.0
) -> FloatArray:
    """Pure NumPy non-apodized lineshape."""
    z1 = aq * (1j * dx + r2)
    spec = aq * (1.0 - np.exp(-z1)) / z1
    return (spec * np.exp(1j * np.deg2rad(phase))).real


def _sp1_numpy(
    dx: FloatArray, r2: float, aq: float, end: float, off: float, phase: float = 0.0
) -> FloatArray:
    """Pure NumPy SP1 apodization lineshape."""
    z1 = aq * (1j * dx + r2)
    f1, f2 = 1j * off * np.pi, 1j * (end - off) * np.pi
    a1 = (np.exp(+f2) - np.exp(+z1)) * np.exp(-z1 + f1) / (2 * (z1 - f2))
    a2 = (np.exp(+z1) - np.exp(-f2)) * np.exp(-z1 - f1) / (2 * (z1 + f2))
    spec = 1j * aq * (a1 + a2)
    return (spec * np.exp(1j * np.deg2rad(phase))).real


def _sp2_numpy(
    dx: FloatArray, r2: float, aq: float, end: float, off: float, phase: float = 0.0
) -> FloatArray:
    """Pure NumPy SP2 apodization lineshape."""
    z1 = aq * (1j * dx + r2)
    f1, f2 = 1j * off * np.pi, 1j * (end - off) * np.pi
    a1 = (np.exp(+2 * f2) - np.exp(z1)) * np.exp(-z1 + 2 * f1) / (4 * (z1 - 2 * f2))
    a2 = (np.exp(-2 * f2) - np.exp(z1)) * np.exp(-z1 - 2 * f1) / (4 * (z1 + 2 * f2))
    a3 = (1.0 - np.exp(-z1)) / (2 * z1)
    spec = aq * (a1 + a2 + a3)
    return (spec * np.exp(1j * np.deg2rad(phase))).real


# Public API for getting backend-specific functions
def get_gaussian_func():
    """Get current backend's Gaussian function."""
    return _backend_functions.get("gaussian", _gaussian_numpy)


def get_lorentzian_func():
    """Get current backend's Lorentzian function."""
    return _backend_functions.get("lorentzian", _lorentzian_numpy)


def get_pvoigt_func():
    """Get current backend's Pseudo-Voigt function."""
    return _backend_functions.get("pvoigt", _pvoigt_numpy)


def get_no_apod_func():
    """Get current backend's non-apodized lineshape function."""
    return _backend_functions.get("no_apod", _no_apod_numpy)


def get_sp1_func():
    """Get current backend's SP1 apodization function."""
    return _backend_functions.get("sp1", _sp1_numpy)


def get_sp2_func():
    """Get current backend's SP2 apodization function."""
    return _backend_functions.get("sp2", _sp2_numpy)


# Initialize with best available backend on import
_initialize_backend_functions("numpy")


# Convenience functions that use the current backend
def gaussian(dx: FloatArray, fwhm: float) -> FloatArray:
    """Compute Gaussian lineshape using current backend."""
    func = get_gaussian_func()
    return func(dx, fwhm)


def lorentzian(dx: FloatArray, fwhm: float) -> FloatArray:
    """Compute Lorentzian lineshape using current backend."""
    func = get_lorentzian_func()
    return func(dx, fwhm)


def pvoigt(dx: FloatArray, fwhm: float, eta: float) -> FloatArray:
    """Compute Pseudo-Voigt lineshape using current backend."""
    func = get_pvoigt_func()
    return func(dx, fwhm, eta)


# Backend info utilities
def print_backend_info() -> str:
    """Get formatted string with backend information."""
    lines = [
        "Computational Backend Information",
        "=" * 40,
        f"Current backend: {_current_backend}",
        f"Available backends: {get_available_backends()}",
        f"Best backend: {get_best_backend()}",
    ]

    # Check JAX devices if available
    if _current_backend == "jax":
        try:
            import jax

            devices = jax.devices()
            lines.append(f"JAX devices: {[str(d) for d in devices]}")
        except Exception:
            pass

    return "\n".join(lines)
