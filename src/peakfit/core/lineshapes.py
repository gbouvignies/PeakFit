"""Pure JAX lineshape functions for NMR peak fitting.

This module provides high-performance lineshape functions using JAX for
automatic differentiation and JIT compilation. Falls back to NumPy for
basic lineshape evaluation if JAX is unavailable.

Key features:
- Native complex arithmetic (much cleaner than Numba's manual splitting)
- JIT compilation for performance
- 64-bit precision by default
- Automatic differentiation support
- GPU-ready (no code changes needed)

Performance note (Phase 1):
Current scipy-based optimizer calls lineshape functions in Python loops,
which triggers JAX recompilation overhead. Phase 2 will replace this with
a pure JAX optimizer (Optimistix) that vectorizes the entire computation,
providing 5-10x speedup over current JAX implementation and 2-3x over Numba.
For now, use Numba backend for best performance in production fits.
"""

from collections.abc import Callable
from typing import Any

# Try to import JAX for JIT compilation and autodiff
try:
    import jax
    import jax.numpy as jnp
    from jax import Array

    # Enable 64-bit precision for numerical accuracy in NMR fitting
    jax.config.update("jax_enable_x64", True)

    HAS_JAX = True
    ArrayType = Array
except ImportError:
    import numpy as jnp  # type: ignore[no-redef]

    HAS_JAX = False
    ArrayType = Any  # type: ignore[misc,assignment]

    # Fallback decorator that does nothing
    class jax:  # type: ignore[no-redef]
        @staticmethod
        def jit(func: Callable[..., Any]) -> Callable[..., Any]:
            """No-op decorator when JAX unavailable."""
            return func


# =============================================================================
# Core Lineshape Functions (work with both JAX and NumPy)
# =============================================================================


@jax.jit
def gaussian(dx: ArrayType, fwhm: float) -> ArrayType:
    """Gaussian lineshape function.

    Args:
        dx: Frequency offset array (Hz)
        fwhm: Full width at half maximum (Hz)

    Returns:
        Gaussian profile values (normalized to peak height = 1)

    Notes:
        Uses the formula: exp(-4*ln(2)*dx^2/fwhm^2)
    """
    c = 4.0 * jnp.log(2.0) / (fwhm * fwhm)
    return jnp.exp(-dx * dx * c)


@jax.jit
def lorentzian(dx: ArrayType, fwhm: float) -> ArrayType:
    """Lorentzian lineshape function.

    Args:
        dx: Frequency offset array (Hz)
        fwhm: Full width at half maximum (Hz)

    Returns:
        Lorentzian profile values (normalized to peak height = 1)

    Notes:
        Uses the formula: (fwhm/2)^2 / (dx^2 + (fwhm/2)^2)
    """
    half_width_sq = (0.5 * fwhm) ** 2
    return half_width_sq / (dx * dx + half_width_sq)


@jax.jit
def pvoigt(dx: ArrayType, fwhm: float, eta: float) -> ArrayType:
    """Pseudo-Voigt lineshape (linear combination of Gaussian and Lorentzian).

    Args:
        dx: Frequency offset array (Hz)
        fwhm: Full width at half maximum (Hz)
        eta: Lorentzian fraction (0=pure Gaussian, 1=pure Lorentzian)

    Returns:
        Pseudo-Voigt profile values (normalized to peak height = 1)

    Notes:
        Uses: (1-eta)*Gaussian + eta*Lorentzian
        This is an approximation to the true Voigt profile but much faster.
    """
    # Compute Gaussian component
    c_gauss = 4.0 * jnp.log(2.0) / (fwhm * fwhm)
    gauss = jnp.exp(-dx * dx * c_gauss)

    # Compute Lorentzian component
    half_width_sq = (0.5 * fwhm) ** 2
    lorentz = half_width_sq / (dx * dx + half_width_sq)

    # Linear combination
    return (1.0 - eta) * gauss + eta * lorentz


@jax.jit
def no_apod(dx: ArrayType, r2: float, aq: float, phase: float = 0.0) -> ArrayType:
    """Non-apodized NMR lineshape from Fourier transform of exponential decay.

    Args:
        dx: Frequency offset array (Hz)
        r2: R2 relaxation rate (Hz)
        aq: Acquisition time (s)
        phase: Phase correction (degrees)

    Returns:
        NoApod profile values (real part after phase correction)

    Notes:
        This is the Fourier transform of: exp(-R2*t) * [1 - exp(-t/aq)]
        where the second term accounts for finite acquisition time.

        JAX version uses native complex arithmetic - much cleaner than
        Numba's manual real/imag splitting (~30 lines vs ~40 lines).
    """
    # Complex frequency variable: z1 = aq * (i*dx + r2)
    z1 = aq * (1j * dx + r2)

    # Fourier transform formula: aq * (1 - exp(-z1)) / z1
    spec = aq * (1.0 - jnp.exp(-z1)) / z1

    # Apply phase correction and take real part
    return (spec * jnp.exp(1j * jnp.deg2rad(phase))).real


@jax.jit
def sp1(
    dx: ArrayType, r2: float, aq: float, end: float, off: float, phase: float = 0.0
) -> ArrayType:
    """SP1 (sine bell) apodized NMR lineshape.

    The SP1 apodization applies a sine bell window function to the FID:
        sin(π * (t/aq - off) / (end - off))

    Args:
        dx: Frequency offset array (Hz)
        r2: R2 relaxation rate (Hz)
        aq: Acquisition time (s)
        end: End parameter for sine bell (typically 1.0)
        off: Offset parameter for sine bell (typically 0.0)
        phase: Phase correction (degrees)

    Returns:
        SP1 profile values (real part after phase correction)

    Notes:
        JAX version uses native complex arithmetic.
        Original Numba version: ~100 lines of manual complex math
        This JAX version: ~15 lines with native complex operations
    """
    # Complex frequency variable
    z1 = aq * (1j * dx + r2)

    # Sine bell parameters (purely imaginary)
    f1 = 1j * off * jnp.pi
    f2 = 1j * (end - off) * jnp.pi

    # Fourier transform of windowed FID (complex formula)
    a1 = (jnp.exp(+f2) - jnp.exp(+z1)) * jnp.exp(-z1 + f1) / (2 * (z1 - f2))
    a2 = (jnp.exp(+z1) - jnp.exp(-f2)) * jnp.exp(-z1 - f1) / (2 * (z1 + f2))

    # Final spectrum (multiply by i*aq)
    spec = 1j * aq * (a1 + a2)

    # Apply phase correction and take real part
    return (spec * jnp.exp(1j * jnp.deg2rad(phase))).real


@jax.jit
def sp2(
    dx: ArrayType, r2: float, aq: float, end: float, off: float, phase: float = 0.0
) -> ArrayType:
    """SP2 (sine squared bell) apodized NMR lineshape.

    The SP2 apodization applies a sine squared bell window to the FID:
        sin^2(π * (t/aq - off) / (end - off))

    Args:
        dx: Frequency offset array (Hz)
        r2: R2 relaxation rate (Hz)
        aq: Acquisition time (s)
        end: End parameter for sine bell (typically 1.0)
        off: Offset parameter for sine bell (typically 0.0)
        phase: Phase correction (degrees)

    Returns:
        SP2 profile values (real part after phase correction)

    Notes:
        JAX version uses native complex arithmetic.
        Original Numba version: ~110 lines of manual complex math
        This JAX version: ~20 lines with native complex operations
    """
    # Complex frequency variable
    z1 = aq * (1j * dx + r2)

    # Sine squared bell parameters (purely imaginary)
    f1 = 1j * off * jnp.pi
    f2 = 1j * (end - off) * jnp.pi

    # Fourier transform of windowed FID (complex formula with 3 terms)
    a1 = (jnp.exp(+2 * f2) - jnp.exp(z1)) * jnp.exp(-z1 + 2 * f1) / (4 * (z1 - 2 * f2))
    a2 = (jnp.exp(-2 * f2) - jnp.exp(z1)) * jnp.exp(-z1 - 2 * f1) / (4 * (z1 + 2 * f2))
    a3 = (1.0 - jnp.exp(-z1)) / (2 * z1)

    # Final spectrum
    spec = aq * (a1 + a2 + a3)

    # Apply phase correction and take real part
    return (spec * jnp.exp(1j * jnp.deg2rad(phase))).real


# =============================================================================
# Linear Least Squares for Amplitude Calculation
# =============================================================================


@jax.jit
def calculate_lstsq_amplitude(shapes: ArrayType, data: ArrayType) -> ArrayType:
    """Fast linear least squares for amplitude calculation using normal equations.

    Solves: minimize ||shapes.T @ amplitudes - data||^2
    Using normal equations: (shapes @ shapes.T) @ amplitudes = shapes @ data

    Args:
        shapes: Shape matrix (n_peaks × n_points)
        data: Data vector (n_points,)

    Returns:
        Amplitude coefficients (n_peaks,)

    Notes:
        Uses JAX's linear algebra routines which are highly optimized.
        For GPU execution, this automatically uses cuBLAS/cuSOLVER.

        The normal equations approach is faster than QR decomposition for
        small to medium-sized problems typical in NMR peak fitting.
    """
    # Compute: A^T @ A and A^T @ b
    ata = jnp.dot(shapes, shapes.T)
    atb = jnp.dot(shapes, data)

    # Solve linear system: A^T @ A @ x = A^T @ b
    return jnp.linalg.solve(ata, atb)


# =============================================================================
# Utility Functions
# =============================================================================


def check_jax_available() -> bool:
    """Check if JAX is available for optimization.

    Returns:
        bool: True if JAX is available
    """
    return HAS_JAX


def get_backend_info() -> dict[str, Any]:
    """Get information about the active backend.

    Returns:
        dict: Backend status information including:
            - jax_available: Whether JAX is installed
            - backend: 'jax' or 'numpy'
            - jit_enabled: Whether JIT compilation is active
            - precision: Floating point precision (32 or 64 bit)
            - devices: Available compute devices (CPU, GPU, TPU)
    """
    info: dict[str, Any] = {
        "jax_available": HAS_JAX,
        "backend": "jax" if HAS_JAX else "numpy",
        "jit_enabled": HAS_JAX,
    }

    if HAS_JAX:
        import jax as real_jax

        # Get precision setting
        info["precision"] = 64 if real_jax.config.read("jax_enable_x64") else 32

        # Get available devices
        try:
            devices = real_jax.devices()
            info["devices"] = [
                {"type": d.device_kind, "id": d.id, "platform": d.platform} for d in devices
            ]
            info["default_device"] = devices[0].device_kind
        except Exception:
            info["devices"] = []
            info["default_device"] = "unknown"
    else:
        info["precision"] = 64  # NumPy defaults to float64
        info["devices"] = [{"type": "cpu", "id": 0, "platform": "numpy"}]
        info["default_device"] = "cpu"

    return info


def require_jax() -> None:
    """Raise an error if JAX is not available.

    Raises:
        RuntimeError: If JAX is not installed

    Notes:
        This should be called at the start of any function that requires
        JAX for optimization (as opposed to simple lineshape evaluation).
    """
    if not HAS_JAX:
        msg = (
            "JAX is required for peak fitting and optimization.\n\n"
            "Install JAX with:\n"
            "  pip install 'peakfit[performance]'\n\n"
            "Or install JAX directly:\n"
            "  pip install jax jaxlib  # CPU version\n"
            "  # For GPU support, see: https://github.com/google/jax#installation\n\n"
            "Note: NumPy fallback only supports basic lineshape evaluation,\n"
            "not optimization or uncertainty estimation."
        )
        raise RuntimeError(msg)


# =============================================================================
# Backend Registry (for compatibility with existing code)
# =============================================================================

# Export all lineshape functions for easy import
__all__ = [
    "gaussian",
    "lorentzian",
    "pvoigt",
    "no_apod",
    "sp1",
    "sp2",
    "calculate_lstsq_amplitude",
    "check_jax_available",
    "get_backend_info",
    "require_jax",
    "HAS_JAX",
]
