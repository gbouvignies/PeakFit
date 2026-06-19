"""Utilities for lineshape modules."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeGuard, overload

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Callable

    from peakfit.engine.lineshapes.grid import SpectralGrid
    from peakfit.shared.typing import ComplexArray, FloatArray

# =============================================================================
# Constants
# =============================================================================

_LN2 = np.log(2.0)
_SQRT_PI_4LN2 = np.sqrt(np.pi / (4.0 * _LN2))

_J_COUPLING_ABS_THRESHOLD_RAD_S = 1e-10


@dataclass(frozen=True, slots=True)
class LineshapeContext:
    """Optional context for lineshape evaluation and parameter defaults."""

    grid: SpectralGrid | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def get_axis_label(dim_index: int) -> str:
    """Get the axis label for a dimension using Bruker Topspin convention.

    For pseudo-3D experiments:
    - F1 = pseudo-dimension (intensities, CEST offsets, etc.)
    - F2 = first spectral dimension (indirect, e.g., 15N)
    - F3 = second spectral dimension (direct/acquisition, e.g., 1H)

    Args:
        dim_index: 1-based dimension index (1 = first spectral dim after pseudo).

    Returns:
        str: Axis label like "F2", "F3", "F4".
    """
    # Offset by 1: F1 is reserved for pseudo-dimension
    return f"F{dim_index + 1}"


# =============================================================================
# Cached Result Container
# =============================================================================


@dataclass(slots=True)
class CachedResult:
    """Container for cached lineshape evaluation results.

    Stores both values and derivatives from a single _core() call,
    allowing reuse when scipy calls fun() then jac() with same parameters.

    Cache is validated by storing both the array's id() and its bytes hash.
    This handles the case where different arrays happen to be allocated at
    the same memory address after garbage collection.
    """

    # Input parameters (for cache validation)
    dx_id: int = 0  # id() of dx array (fast check)
    dx_hash: int = 0  # Hash of dx.tobytes() (content check)
    params_hash: int = 0  # Hash of scalar parameters

    # Cached outputs
    value: FloatArray | None = field(default=None)
    d_dx: FloatArray | None = field(default=None)
    d_lw: FloatArray | None = field(default=None)
    d_j: FloatArray | None = field(default=None)
    d_eta: FloatArray | None = field(default=None)  # For PseudoVoigt
    d_r2: FloatArray | None = field(default=None)  # For apodization shapes
    d_phase: FloatArray | None = field(default=None)  # For apodization shapes

    def matches(self, dx: FloatArray, *params: float) -> bool:
        """Check if cache matches the given inputs.

        Uses both id() and content hash to handle array reallocation at same address.
        """
        # Fast path: same array object (same id and same content hash)
        if id(dx) == self.dx_id and hash(params) == self.params_hash:
            # Verify content hasn't changed (in case of in-place modification)
            return hash(dx.tobytes()) == self.dx_hash
        return False

    def update_key(self, dx: FloatArray, *params: float) -> None:
        """Update cache key for new inputs."""
        self.dx_id = id(dx)
        self.dx_hash = hash(dx.tobytes())
        self.params_hash = hash(params)


# =============================================================================
# Transform Helpers
# =============================================================================


@overload
def apply_j_coupling(
    kernel_func: Callable[..., FloatArray],
    dw: npt.ArrayLike,
    rate: npt.ArrayLike,
    j_coupling: npt.ArrayLike,
    extras: dict[str, FloatArray],
) -> FloatArray: ...


@overload
def apply_j_coupling(
    kernel_func: Callable[..., ComplexArray],
    dw: npt.ArrayLike,
    rate: npt.ArrayLike,
    j_coupling: npt.ArrayLike,
    extras: dict[str, FloatArray],
) -> ComplexArray: ...


def apply_j_coupling(
    kernel_func: Callable[..., FloatArray | ComplexArray],
    dw: npt.ArrayLike,
    rate: npt.ArrayLike,
    j_coupling: npt.ArrayLike,
    extras: dict[str, FloatArray],
) -> FloatArray | ComplexArray:
    """Apply J-coupling splitting to a lineshape kernel.

    Args:
        kernel_func: Function(dw, rate, extras) -> values
        dw: Angular frequency offset (rad/s), shape (N, K)
        rate: Decay rate (s⁻¹), shape (1, K)
        j_coupling: J-coupling (Hz), shape (K,)
        extras: Extra parameters for the kernel

    Returns:
        Splitted lineshape values, shape (N, K)
    """
    dw_arr = np.asarray(dw, dtype=np.float64)
    rate_arr = np.asarray(rate, dtype=np.float64)
    j_coupling_arr = np.asarray(j_coupling, dtype=np.float64)

    j_rads = np.pi * j_coupling_arr  # Hz -> rad/s
    has_j = np.abs(j_rads) > _J_COUPLING_ABS_THRESHOLD_RAD_S

    if not np.any(has_j):
        return kernel_func(dw_arr, rate_arr, extras)

    j_shift = 0.5 * j_rads[None, :]

    # Calculate components
    val_plus = kernel_func(dw_arr + j_shift, rate_arr, extras)
    val_minus = kernel_func(dw_arr - j_shift, rate_arr, extras)
    val_singlet = kernel_func(dw_arr, rate_arr, extras)

    return np.where(has_j[None, :], val_plus + val_minus, val_singlet)


@overload
def apply_j_coupling_with_derivs(
    kernel_func: Callable[..., tuple[FloatArray, dict[str, FloatArray]]],
    dw: npt.ArrayLike,
    rate: npt.ArrayLike,
    j_coupling: npt.ArrayLike,
    extras: dict[str, FloatArray],
) -> tuple[FloatArray, dict[str, FloatArray]]: ...


@overload
def apply_j_coupling_with_derivs(
    kernel_func: Callable[..., tuple[ComplexArray, dict[str, ComplexArray]]],
    dw: npt.ArrayLike,
    rate: npt.ArrayLike,
    j_coupling: npt.ArrayLike,
    extras: dict[str, FloatArray],
) -> tuple[ComplexArray, dict[str, ComplexArray]]: ...


def apply_j_coupling_with_derivs(
    kernel_func: Callable[
        ..., tuple[FloatArray | ComplexArray, dict[str, FloatArray | ComplexArray]]
    ],
    dw: npt.ArrayLike,
    rate: npt.ArrayLike,
    j_coupling: npt.ArrayLike,
    extras: dict[str, FloatArray],
) -> tuple[FloatArray | ComplexArray, dict[str, FloatArray | ComplexArray]]:
    """Apply J-coupling splitting with derivatives.

    Args:
        kernel_func: Function(dw, rate, extras) -> (values, derivatives)
        dw: Angular frequency offset (rad/s), shape (N, K)
        rate: Decay rate (s⁻¹), shape (1, K)
        j_coupling: J-coupling (Hz), shape (K,)
        extras: Extra parameters for the kernel

    Returns:
        (values, derivatives)
    """
    dw_arr = np.asarray(dw, dtype=np.float64)
    rate_arr = np.asarray(rate, dtype=np.float64)
    j_coupling_arr = np.asarray(j_coupling, dtype=np.float64)

    j_rads = np.pi * j_coupling_arr
    has_j = np.abs(j_rads) > _J_COUPLING_ABS_THRESHOLD_RAD_S

    if not np.any(has_j):
        return kernel_func(dw_arr, rate_arr, extras)

    j_shift = 0.5 * j_rads[None, :]

    # Calculate components
    v_p, d_p = kernel_func(dw_arr + j_shift, rate_arr, extras)
    v_m, d_m = kernel_func(dw_arr - j_shift, rate_arr, extras)
    v_s, d_s = kernel_func(dw_arr, rate_arr, extras)

    # Combine values
    values = np.where(has_j[None, :], v_p + v_m, v_s)

    # Combine derivatives
    derivs: dict[str, FloatArray | ComplexArray] = {}

    # Keys present in all results (e.g. "dw", "rate")
    for key in d_p:
        derivs[key] = np.where(has_j[None, :], d_p[key] + d_m[key], d_s[key])

    return values, derivs


def apply_phase(
    z_values: npt.ArrayLike,
    phases: npt.ArrayLike,
) -> FloatArray:
    """Apply zero-order phase correction and return real part.

    Args:
        z_values: Complex lineshape values, shape (N, K)
        phases: Phase in degrees, shape (K,)

    Returns:
        Real part of phased lineshape, shape (N, K)
    """
    # Convert phase to radians
    z_arr = np.asarray(z_values, dtype=np.complex128)
    phases_arr = np.asarray(phases, dtype=np.float64)
    phi = np.radians(phases_arr)[None, :]

    # Apply rotation: exp(i * phi) * z
    # We want Real part.
    # Re( (x+iy) * (cos+isin) ) = x*cos - y*sin

    # Or simply:
    phasor = np.exp(1j * phi)
    return np.real(z_arr * phasor)


def apply_phase_with_derivs(
    z_values: npt.ArrayLike,
    z_derivs: npt.ArrayLike | dict[str, ComplexArray],
    phases: npt.ArrayLike,
    aq: float,
) -> tuple[FloatArray, dict[str, FloatArray]]:
    """Apply phase correction with derivatives.

    Returns derivatives w.r.t phase and underlying parameters via chain rule.

    Chain Rule for dF/d_param where z = aq*(R2 + i*dw):
        dF/dz is provided.
        d_z/d_dw = i * aq
        d_z/d_r2 = aq

    Args:
        z_values: Complex values F(z)
        z_derivs: Either dF/dz (array) or a dict of partial derivatives dF/dX.
        phases: Phase in degrees
        aq: Acquisition time (for chain rule)

    Returns:
        values: Real part of phased lineshape
        derivs: Dict with keys "rate" (from r2), "dw", "phase"
    """

    def _is_complex_deriv_map(obj: object) -> TypeGuard[dict[str, ComplexArray]]:
        if not isinstance(obj, dict):
            return False
        return all(isinstance(k, str) for k in obj)

    z_arr = np.asarray(z_values, dtype=np.complex128)
    phases_arr = np.asarray(phases, dtype=np.float64)
    phi = np.radians(phases_arr)[None, :]
    phasor = np.exp(1j * phi)

    # 1. Values
    rotated = z_arr * phasor
    values = np.asarray(np.real(rotated), dtype=np.float64)

    # 2. Phase Derivative
    # d/d_phi (Re(F * e^iφ)) = Re(F * i * e^iφ) = -Im(F * e^iφ)
    # d/d_phase_deg = d/d_phi * (pi/180)
    d_phi = -np.imag(rotated)
    d_phase_deg = np.asarray(d_phi * (np.pi / 180.0), dtype=np.float64)

    # 3. Z-based parameter derivatives
    if _is_complex_deriv_map(z_derivs):
        # Case A: Dictionary of partial derivatives (e.g. from SP1/SP2 via J-coupling)
        # d/dX (Re(F * e^iφ)) = Re( dF/dX * e^iφ )
        derivs: dict[str, FloatArray] = {}
        for k, v in z_derivs.items():
            derivs[k] = np.asarray(np.real(v * phasor), dtype=np.float64)
        derivs["phase"] = d_phase_deg
        return values, derivs

    # Case B: Single dF/dz array (chain rule needed)
    # d/d_param = Re( dF/dz * dz/d_param * e^iφ )

    z_derivs_arr = np.asarray(z_derivs, dtype=np.complex128)

    # Rotate the dF/dz
    df_dz_rot = z_derivs_arr * phasor

    # d_dw: z = ... + i*aq*dw  => dz/d_dw = i*aq
    # d/d_dw = Re( df_dz_rot * i * aq ) = -Im(df_dz_rot) * aq
    d_dw = np.asarray((-np.imag(df_dz_rot) * aq), dtype=np.float64)

    # d_r2: z = aq*r2 + ... => dz/d_r2 = aq
    # d/d_r2 = Re( df_dz_rot * aq ) = Re(df_dz_rot) * aq
    d_r2 = np.asarray((np.real(df_dz_rot) * aq), dtype=np.float64)

    return values, {
        "phase": d_phase_deg,
        "dw": d_dw,
        "rate": d_r2,  # "rate" key maps to R2 here
    }


# =============================================================================
# Context + Spec Helpers
# =============================================================================


def estimate_cs_bounds_ppm(
    context: LineshapeContext | None,
    *,
    fwhm_estimate_hz: float = 50.0,
    default_ppm: float = 0.1,
) -> float:
    """Return a reasonable +/- ppm window around a peak center.

    If a spectral grid is available, the Hz estimate is converted to ppm using
    the grid's spectral parameters. Otherwise a fixed ppm window is returned.
    """
    if context is None or context.grid is None:
        return default_ppm
    return float(context.grid.spec_params.hz2ppm(fwhm_estimate_hz))


def require_grid(context: LineshapeContext | None, *, shape: str) -> SpectralGrid:
    """Return the grid from a LineshapeContext or raise a consistent error."""
    if context is None or context.grid is None:
        msg = f"{shape} requires LineshapeContext.grid for unit conversions."
        raise ValueError(msg)
    return context.grid


def doublet_offsets(
    x: npt.ArrayLike,
    cs: npt.ArrayLike,
    j_hz: npt.ArrayLike,
    grid: SpectralGrid,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Compute doublet +/- J/2 offsets and corresponding sign corrections."""
    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    cs_arr = np.atleast_1d(np.asarray(cs, dtype=np.float64))
    j_arr = np.atleast_1d(np.asarray(j_hz, dtype=np.float64))

    delta_ppm = grid.spec_params.hz2ppm(0.5 * j_arr)
    pos_plus = cs_arr + delta_ppm
    pos_minus = cs_arr - delta_ppm
    dw_plus, sign_plus = grid.compute_offsets(x_arr, pos_plus)
    dw_minus, sign_minus = grid.compute_offsets(x_arr, pos_minus)
    return dw_plus, sign_plus, dw_minus, sign_minus


def get_apodization_state(
    context: LineshapeContext,
    *,
    state_key: str,
    shape: str,
    make_state: Callable[[float, float, float], dict[str, complex | float]],
) -> dict[str, complex | float]:
    """Extract apodization state, preferring cached state or grid parameters."""
    state = context.extras.get(state_key)
    if isinstance(state, dict):
        return state

    if context.grid is not None:
        spec_params = context.grid.spec_params
        return make_state(spec_params.aq_time, spec_params.apodq1, spec_params.apodq2)

    aq = context.extras.get("aq")
    apodq1 = context.extras.get("apodq1")
    apodq2 = context.extras.get("apodq2")
    if aq is None or apodq1 is None or apodq2 is None:
        msg = f"{shape} requires apodization parameters in LineshapeContext."
        raise ValueError(msg)
    return make_state(float(aq), float(apodq1), float(apodq2))
