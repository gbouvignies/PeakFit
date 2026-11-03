"""Utilities for lineshape modules."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeGuard

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

    For pseudo-ND experiments:
    - F1 = pseudo-dimensional axis (intensities, CEST offsets, etc.)
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


# =============================================================================
# Transform Helpers
# =============================================================================


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
