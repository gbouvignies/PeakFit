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
    """Get the best available backend (prefers JAX > Numba > NumPy)."""
    available = get_available_backends()
    if "jax" in available:
        return "jax"
    if "numba" in available:
        return "numba"
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
        }
    elif backend == "numba":
        from peakfit.core.optimized import gaussian_jit, lorentzian_jit, pvoigt_jit

        _backend_functions = {
            "gaussian": gaussian_jit,
            "lorentzian": lorentzian_jit,
            "pvoigt": pvoigt_jit,
        }
    elif backend == "jax":
        from peakfit.core.jax_backend import (
            gaussian_jax,
            lorentzian_jax,
            pseudo_voigt_jax,
        )

        _backend_functions = {
            "gaussian": gaussian_jax,
            "lorentzian": lorentzian_jax,
            "pvoigt": pseudo_voigt_jax,
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
