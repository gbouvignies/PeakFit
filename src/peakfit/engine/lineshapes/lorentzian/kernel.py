"""Lorentzian kernel implementations."""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from peakfit.shared.typing import FloatArray


def kernel(dw: npt.ArrayLike, lw: npt.ArrayLike) -> FloatArray:
    """Compute Lorentzian kernel.

    Args:
        dw: Frequency offsets in Hz
        lw: Linewidth FWHM in Hz
    """
    dw_arr = np.asarray(dw, dtype=np.float64)
    lw_arr = np.asarray(lw, dtype=np.float64)
    # gamma = FWHM/2 (half-width at half-maximum in Hz)
    gamma = 0.5 * lw_arr
    gamma2 = gamma * gamma
    result: FloatArray = gamma2 / (gamma2 + dw_arr * dw_arr)
    return result


def kernel_with_derivs(
    dw: npt.ArrayLike, lw: npt.ArrayLike
) -> tuple[FloatArray, dict[str, FloatArray]]:
    """Compute Lorentzian kernel with derivatives.

    Args:
        dw: Frequency offsets in Hz
        lw: Linewidth FWHM in Hz
    """
    dw_arr = np.asarray(dw, dtype=np.float64)
    lw_arr = np.asarray(lw, dtype=np.float64)
    # gamma = FWHM/2 (half-width at half-maximum in Hz)
    gamma = 0.5 * lw_arr
    gamma2 = gamma * gamma
    dw2 = dw_arr * dw_arr
    denom = gamma2 + dw2
    lorentz: FloatArray = gamma2 / denom

    denom_inv2 = 1.0 / (denom * denom)
    d_dw = (-2.0 * gamma2 * dw_arr * denom_inv2).astype(np.float64)
    d_lw = (gamma * dw2 * denom_inv2).astype(np.float64)

    return lorentz, {"dw": d_dw, "lw": d_lw}


__all__ = ["kernel", "kernel_with_derivs"]
