"""Gaussian kernel implementations."""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from peakfit.engine.lineshapes.utils import _LN2

if TYPE_CHECKING:
    from peakfit.shared.typing import FloatArray


def kernel(dw: npt.ArrayLike, lw: npt.ArrayLike) -> FloatArray:
    """Compute Gaussian kernel.

    Args:
        dw: Frequency offsets in Hz
        lw: Linewidth FWHM in Hz
    """
    dw_arr = np.asarray(dw, dtype=np.float64)
    lw_arr = np.asarray(lw, dtype=np.float64)
    # gamma = FWHM/2 (half-width at half-maximum in Hz)
    gamma = 0.5 * lw_arr
    c = _LN2 / (gamma * gamma)
    result: FloatArray = np.exp(-c * dw_arr * dw_arr, dtype=np.float64)
    return result


def kernel_with_derivs(
    dw: npt.ArrayLike, lw: npt.ArrayLike
) -> tuple[FloatArray, dict[str, FloatArray]]:
    """Compute Gaussian kernel and derivatives.

    Args:
        dw: Frequency offsets in Hz
        lw: Linewidth FWHM in Hz
    """
    dw_arr = np.asarray(dw, dtype=np.float64)
    lw_arr = np.asarray(lw, dtype=np.float64)
    # gamma = FWHM/2 (half-width at half-maximum in Hz)
    gamma = 0.5 * lw_arr
    gamma2 = gamma * gamma
    c = _LN2 / gamma2
    dw2 = dw_arr * dw_arr
    gauss = np.exp(-c * dw2, dtype=np.float64)

    d_dw = (-2.0 * c * dw_arr * gauss).astype(np.float64)
    d_lw = (c * dw2 * gauss / gamma).astype(np.float64)

    return gauss, {"dw": d_dw, "lw": d_lw}


__all__ = ["kernel", "kernel_with_derivs"]
