"""JAX-optimized lineshape functions.

This module provides JAX-JIT compiled versions of core lineshape functions.
If JAX is available, functions are JIT-compiled for better performance and
can run on CPU or GPU. JAX supports native complex arithmetic, resulting in
cleaner code compared to Numba implementations.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

# Try to import JAX for JIT compilation
try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

    # Fallback: jnp is just numpy when JAX is not available
    jnp = np  # type: ignore[assignment]

    # Fallback decorator that does nothing
    class jax:  # type: ignore[no-redef] # noqa: N801
        """Fallback JAX namespace when JAX is not installed."""

        @staticmethod
        def jit(*args: object, **kwargs: object) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            """Fallback JIT decorator that does nothing."""

            def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
                return func

            return decorator


# =============================================================================
# JAX JIT-compiled implementations
# =============================================================================


@jax.jit
def _gaussian_jax_impl(dx: jnp.ndarray, fwhm: float) -> jnp.ndarray:
    """JAX-optimized Gaussian lineshape.

    Args:
        dx: Frequency offset array (Hz)
        fwhm: Full width at half maximum (Hz)

    Returns:
        Gaussian lineshape values (normalized to 1 at center)

    Notes:
        Uses the original NumPy formula with native JAX operations.
        Formula: exp(-dx^2 * 4*ln(2) / fwhm^2)
    """
    c = 4.0 * jnp.log(2.0) / (fwhm * fwhm)
    return jnp.exp(-dx * dx * c)


@jax.jit
def _lorentzian_jax_impl(dx: jnp.ndarray, fwhm: float) -> jnp.ndarray:
    """JAX-optimized Lorentzian lineshape.

    Args:
        dx: Frequency offset array (Hz)
        fwhm: Full width at half maximum (Hz)

    Returns:
        Lorentzian lineshape values (normalized to 1 at center)

    Notes:
        Uses the original NumPy formula with native JAX operations.
        Formula: (fwhm/2)^2 / (dx^2 + (fwhm/2)^2)
    """
    half_width_sq = (0.5 * fwhm) ** 2
    return half_width_sq / (dx * dx + half_width_sq)


@jax.jit
def _pvoigt_jax_impl(dx: jnp.ndarray, fwhm: float, eta: float) -> jnp.ndarray:
    """JAX-optimized Pseudo-Voigt lineshape.

    Args:
        dx: Frequency offset array (Hz)
        fwhm: Full width at half maximum (Hz)
        eta: Mixing parameter (0 = pure Gaussian, 1 = pure Lorentzian)

    Returns:
        Pseudo-Voigt lineshape values (normalized to 1 at center)

    Notes:
        Uses the original NumPy formula with native JAX operations.
        Formula: (1-eta)*Gaussian + eta*Lorentzian
        Inline computation to avoid function call overhead.
    """
    # Gaussian component
    c_gauss = 4.0 * jnp.log(2.0) / (fwhm * fwhm)
    gauss = jnp.exp(-dx * dx * c_gauss)

    # Lorentzian component
    half_width_sq = (0.5 * fwhm) ** 2
    lorentz = half_width_sq / (dx * dx + half_width_sq)

    # Linear combination
    return (1.0 - eta) * gauss + eta * lorentz


@jax.jit
def _no_apod_jax_impl(dx: jnp.ndarray, r2: float, aq: float, phase: float = 0.0) -> jnp.ndarray:
    """JAX-optimized non-apodized lineshape.

    Args:
        dx: Frequency offset array (Hz)
        r2: Relaxation rate (1/s)
        aq: Acquisition time (s)
        phase: Phase rotation in degrees (default: 0.0)

    Returns:
        Real part of the non-apodized lineshape

    Notes:
        Uses the original NumPy formula with NATIVE COMPLEX ARITHMETIC.
        This is cleaner than Numba's manual real/imaginary splitting.
        Formula: Re[aq * (1 - exp(-z1)) / z1 * exp(i*phase)]
        where z1 = aq * (i*dx + r2)
    """
    z1 = aq * (1j * dx + r2)
    spec = aq * (1.0 - jnp.exp(-z1)) / z1
    return (spec * jnp.exp(1j * jnp.deg2rad(phase))).real


@jax.jit
def _sp1_jax_impl(
    dx: jnp.ndarray, r2: float, aq: float, end: float, off: float, phase: float = 0.0
) -> jnp.ndarray:
    """JAX-optimized SP1 sine bell apodization lineshape.

    Args:
        dx: Frequency offset array (Hz)
        r2: Relaxation rate (1/s)
        aq: Acquisition time (s)
        end: End parameter for sine bell
        off: Offset parameter for sine bell
        phase: Phase rotation in degrees (default: 0.0)

    Returns:
        Real part of the SP1 apodized lineshape

    Notes:
        Uses the original NumPy formula with NATIVE COMPLEX ARITHMETIC.
        This is much cleaner than Numba's manual real/imaginary splitting.
        Formula: Re[i*aq*(a1 + a2) * exp(i*phase)]
    """
    z1 = aq * (1j * dx + r2)
    f1, f2 = 1j * off * jnp.pi, 1j * (end - off) * jnp.pi

    a1 = (jnp.exp(+f2) - jnp.exp(+z1)) * jnp.exp(-z1 + f1) / (2 * (z1 - f2))
    a2 = (jnp.exp(+z1) - jnp.exp(-f2)) * jnp.exp(-z1 - f1) / (2 * (z1 + f2))

    spec = 1j * aq * (a1 + a2)
    return (spec * jnp.exp(1j * jnp.deg2rad(phase))).real


@jax.jit
def _sp2_jax_impl(
    dx: jnp.ndarray, r2: float, aq: float, end: float, off: float, phase: float = 0.0
) -> jnp.ndarray:
    """JAX-optimized SP2 sine squared bell apodization lineshape.

    Args:
        dx: Frequency offset array (Hz)
        r2: Relaxation rate (1/s)
        aq: Acquisition time (s)
        end: End parameter for sine squared bell
        off: Offset parameter for sine squared bell
        phase: Phase rotation in degrees (default: 0.0)

    Returns:
        Real part of the SP2 apodized lineshape

    Notes:
        Uses the original NumPy formula with NATIVE COMPLEX ARITHMETIC.
        This is much cleaner than Numba's manual real/imaginary splitting.
        Formula: Re[aq*(a1 + a2 + a3) * exp(i*phase)]
    """
    z1 = aq * (1j * dx + r2)
    f1, f2 = 1j * off * jnp.pi, 1j * (end - off) * jnp.pi

    a1 = (jnp.exp(+2 * f2) - jnp.exp(z1)) * jnp.exp(-z1 + 2 * f1) / (4 * (z1 - 2 * f2))
    a2 = (jnp.exp(-2 * f2) - jnp.exp(z1)) * jnp.exp(-z1 - 2 * f1) / (4 * (z1 + 2 * f2))
    a3 = (1.0 - jnp.exp(-z1)) / (2 * z1)

    spec = aq * (a1 + a2 + a3)
    return (spec * jnp.exp(1j * jnp.deg2rad(phase))).real


# =============================================================================
# Public exports with NumPy array conversion for compatibility
# =============================================================================


def _convert_to_numpy(func: Callable[..., Any]) -> Callable[..., np.ndarray]:
    """Wrapper that converts JAX array outputs to NumPy arrays.

    This ensures compatibility with code that expects NumPy arrays.
    """

    def wrapper(*args: Any, **kwargs: Any) -> np.ndarray:
        result = func(*args, **kwargs)
        # Convert JAX array to NumPy array
        return np.asarray(result)

    return wrapper


if HAS_JAX:
    # Wrap JAX functions to return NumPy arrays for compatibility
    gaussian_jax = _convert_to_numpy(_gaussian_jax_impl)
    lorentzian_jax = _convert_to_numpy(_lorentzian_jax_impl)
    pvoigt_jax = _convert_to_numpy(_pvoigt_jax_impl)
    no_apod_jax = _convert_to_numpy(_no_apod_jax_impl)
    sp1_jax = _convert_to_numpy(_sp1_jax_impl)
    sp2_jax = _convert_to_numpy(_sp2_jax_impl)
else:
    # Fallback to NumPy implementations if JAX is not available
    # These will be imported from optimized.py if needed
    gaussian_jax = None  # type: ignore[assignment]
    lorentzian_jax = None  # type: ignore[assignment]
    pvoigt_jax = None  # type: ignore[assignment]
    no_apod_jax = None  # type: ignore[assignment]
    sp1_jax = None  # type: ignore[assignment]
    sp2_jax = None  # type: ignore[assignment]


# =============================================================================
# Utility functions
# =============================================================================


def prewarm_jax_functions() -> None:
    """Pre-compile JAX functions with example inputs.

    This is useful to trigger compilation before multiprocessing,
    similar to the Numba pre-warming strategy.
    """
    if not HAS_JAX:
        return

    # Create example inputs for compilation
    dx = jnp.linspace(-100.0, 100.0, 100)
    fwhm = 10.0
    eta = 0.5
    r2, aq, end, off, phase = 5.0, 0.1, 1.0, 0.0, 0.0

    # Trigger compilation by calling each function once
    _ = gaussian_jax(dx, fwhm)
    _ = lorentzian_jax(dx, fwhm)
    _ = pvoigt_jax(dx, fwhm, eta)
    _ = no_apod_jax(dx, r2, aq, phase)
    _ = sp1_jax(dx, r2, aq, end, off, phase)
    _ = sp2_jax(dx, r2, aq, end, off, phase)
