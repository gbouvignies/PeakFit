"""No-apodization kernel implementations."""

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from peakfit.shared.typing import ComplexArray


def kernel(dw: npt.ArrayLike, lw: npt.ArrayLike, state: dict[str, Any]) -> ComplexArray:
    """Compute NoApod complex kernel.

    Args:
        dw: Frequency offsets in Hz
        lw: Linewidth FWHM in Hz
        state: State dict containing {"aq": acquisition_time}
    """
    aq = state["aq"]
    dw_arr = np.asarray(dw, dtype=np.float64)
    lw_arr = np.asarray(lw, dtype=np.float64)
    # z = aq × (rate + i×dw_rad) where rate = π×lw and dw_rad = 2π×dw_hz
    z: ComplexArray = aq * np.pi * (lw_arr + 2j * dw_arr)
    emz = np.exp(-z)
    return aq * (1.0 - emz) / z


def kernel_with_derivs(
    dw: npt.ArrayLike, lw: npt.ArrayLike, state: dict[str, Any]
) -> tuple[ComplexArray, dict[str, ComplexArray]]:
    """Compute NoApod kernel and derivatives for apply_phase.

    Args:
        dw: Frequency offsets in Hz
        lw: Linewidth FWHM in Hz
        state: State dict containing {"aq": acquisition_time}
    """
    aq = state["aq"]
    dw_arr = np.asarray(dw, dtype=np.float64)
    lw_arr = np.asarray(lw, dtype=np.float64)
    # z = aq × (rate + i×dw_rad) where rate = π×lw and dw_rad = 2π×dw_hz
    z: ComplexArray = aq * np.pi * (lw_arr + 2j * dw_arr)
    emz = np.exp(-z)
    kernel_val = aq * (1.0 - emz) / z
    d_z = aq * (emz * (z + 1.0) - 1.0) / (z * z)

    return kernel_val, {
        "lw": d_z * (np.pi * aq),
        "dw": d_z * (2j * np.pi * aq),
    }


__all__ = ["kernel", "kernel_with_derivs"]
