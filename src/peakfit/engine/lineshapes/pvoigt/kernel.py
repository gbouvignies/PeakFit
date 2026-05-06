"""Pseudo-Voigt kernel implementations."""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from peakfit.engine.lineshapes.utils import _LN2

if TYPE_CHECKING:
    from peakfit.shared.typing import FloatArray


def kernel(dw: npt.ArrayLike, lw: npt.ArrayLike, eta: npt.ArrayLike) -> FloatArray:
    """Compute Pseudo-Voigt kernel.

    Args:
        dw: Frequency offsets in Hz
        lw: Linewidth FWHM in Hz
        eta: Lorentzian fraction (0 = pure Gaussian, 1 = pure Lorentzian)
    """
    dw_arr = np.asarray(dw, dtype=np.float64)
    lw_arr = np.asarray(lw, dtype=np.float64)
    eta_arr = np.asarray(eta, dtype=np.float64)
    eta_bc = eta_arr[None, :] if eta_arr.ndim == 1 else eta_arr

    # gamma = FWHM/2 (half-width at half-maximum in Hz)
    gamma = 0.5 * lw_arr
    gamma2 = gamma * gamma
    dw2 = dw_arr * dw_arr
    lorentz = gamma2 / (gamma2 + dw2)

    c = _LN2 / gamma2
    gauss = np.exp(-c * dw2)

    result: FloatArray = (eta_bc * lorentz + (1.0 - eta_bc) * gauss).astype(np.float64)
    return result


def kernel_with_derivs(
    dw: npt.ArrayLike, lw: npt.ArrayLike, eta: npt.ArrayLike
) -> tuple[FloatArray, dict[str, FloatArray]]:
    """Compute Pseudo-Voigt kernel with derivatives.

    Args:
        dw: Frequency offsets in Hz
        lw: Linewidth FWHM in Hz
        eta: Lorentzian fraction (0 = pure Gaussian, 1 = pure Lorentzian)
    """
    dw_arr = np.asarray(dw, dtype=np.float64)
    lw_arr = np.asarray(lw, dtype=np.float64)
    eta_arr = np.asarray(eta, dtype=np.float64)
    eta_bc = eta_arr[None, :] if eta_arr.ndim == 1 else eta_arr

    # gamma = FWHM/2 (half-width at half-maximum in Hz)
    gamma = 0.5 * lw_arr
    gamma2 = gamma * gamma
    dw2 = dw_arr * dw_arr
    denom = gamma2 + dw2
    lorentz = gamma2 / denom
    denom_inv2 = 1.0 / (denom * denom)
    d_lor_dw = -2.0 * gamma2 * dw_arr * denom_inv2
    d_lor_lw = gamma * dw2 * denom_inv2

    c = _LN2 / gamma2
    gauss = np.exp(-c * dw2)
    d_gau_dw = -2.0 * c * dw_arr * gauss
    d_gau_lw = c * dw2 * gauss / gamma

    one_minus_eta = 1.0 - eta_bc
    pvoigt = eta_bc * lorentz + one_minus_eta * gauss
    d_dw = eta_bc * d_lor_dw + one_minus_eta * d_gau_dw
    d_lw = eta_bc * d_lor_lw + one_minus_eta * d_gau_lw
    d_eta = lorentz - gauss

    return pvoigt.astype(np.float64), {
        "dw": d_dw.astype(np.float64),
        "lw": d_lw.astype(np.float64),
        "eta": d_eta.astype(np.float64),
    }


__all__ = ["kernel", "kernel_with_derivs"]
