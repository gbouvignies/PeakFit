"""SP2 kernel implementations."""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from peakfit.shared.typing import ComplexArray


def make_state(aq: float, apodq1: float, apodq2: float) -> dict[str, complex | float]:
    """Precompute SP2 constants used by the kernel."""
    f1 = apodq1 * np.pi
    f2 = (apodq2 - apodq1) * np.pi

    e2if1 = np.exp(2j * f1)
    em2if1 = np.exp(-2j * f1)
    e2if2 = np.exp(2j * f2)
    em2if2 = np.exp(-2j * f2)

    return {
        "aq": aq,
        "aq_quarter": 0.25 * aq,
        "aq_half": 0.5 * aq,
        "i2f2": 2j * f2,
        "e2if1": e2if1,
        "em2if1": em2if1,
        "e2if2": e2if2,
        "em2if2": em2if2,
        "e2if12": e2if1 * e2if2,
        "em2if12": em2if1 * em2if2,
    }


def kernel_from_z(z: npt.ArrayLike, state: dict[str, complex | float]) -> ComplexArray:
    """Compute SP2 complex kernel."""
    z_arr = np.asarray(z, dtype=np.complex128)
    emz = np.exp(-z_arr)
    ez = 1.0 / emz

    denom1 = z_arr - state["i2f2"]
    denom2 = z_arr + state["i2f2"]

    num1 = (state["e2if2"] - ez) * state["e2if1"] * emz
    num2 = (state["em2if2"] - ez) * state["em2if1"] * emz
    num3 = 1.0 - emz

    term1 = state["aq_quarter"] * num1 / denom1
    term2 = state["aq_quarter"] * num2 / denom2
    term3 = state["aq_half"] * num3 / z

    result: ComplexArray = term1 + term2 + term3
    return result


def kernel_from_z_with_derivs(
    z: npt.ArrayLike, state: dict[str, complex | float]
) -> tuple[ComplexArray, ComplexArray]:
    """Compute SP2 kernel and dF/dz."""
    z_arr = np.asarray(z, dtype=np.complex128)
    emz = np.exp(-z_arr)
    ez = 1.0 / emz

    denom1 = z_arr - state["i2f2"]
    denom2 = z_arr + state["i2f2"]

    num1 = (state["e2if2"] - ez) * state["e2if1"] * emz
    num2 = (state["em2if2"] - ez) * state["em2if1"] * emz
    num3 = 1.0 - emz

    term1 = state["aq_quarter"] * num1 / denom1
    term2 = state["aq_quarter"] * num2 / denom2
    term3 = state["aq_half"] * num3 / z
    kernel_val = term1 + term2 + term3

    dnum1_dz = -state["e2if12"] * emz
    dnum2_dz = -state["em2if12"] * emz

    dterm1_dz = state["aq_quarter"] * (dnum1_dz * denom1 - num1) / (denom1 * denom1)
    dterm2_dz = state["aq_quarter"] * (dnum2_dz * denom2 - num2) / (denom2 * denom2)
    dterm3_dz = state["aq_half"] * (emz * (z_arr + 1.0) - 1.0) / (z_arr * z_arr)

    df_dz = dterm1_dz + dterm2_dz + dterm3_dz
    return kernel_val, df_dz


def kernel(dw: npt.ArrayLike, lw: npt.ArrayLike, state: dict[str, complex | float]) -> ComplexArray:
    """Compute SP2 kernel for a given dw and lw.

    Args:
        dw: Frequency offsets in Hz
        lw: Linewidth FWHM in Hz
        state: Precomputed kernel state from make_state
    """
    dw_arr = np.asarray(dw, dtype=np.float64)
    lw_arr = np.asarray(lw, dtype=np.float64)
    # z = aq × (rate + i×dw_rad) where rate = π×lw and dw_rad = 2π×dw_hz
    z: ComplexArray = state["aq"] * np.pi * (lw_arr + 2j * dw_arr)
    return kernel_from_z(z, state)


def kernel_with_derivs(
    dw: npt.ArrayLike, lw: npt.ArrayLike, state: dict[str, complex | float]
) -> tuple[ComplexArray, dict[str, ComplexArray]]:
    """Compute SP2 kernel and derivatives for apply_phase.

    Args:
        dw: Frequency offsets in Hz
        lw: Linewidth FWHM in Hz
        state: Precomputed kernel state from make_state
    """
    dw_arr = np.asarray(dw, dtype=np.float64)
    lw_arr = np.asarray(lw, dtype=np.float64)
    # z = aq × (rate + i×dw_rad) where rate = π×lw and dw_rad = 2π×dw_hz
    z: ComplexArray = state["aq"] * np.pi * (lw_arr + 2j * dw_arr)
    val, d_z = kernel_from_z_with_derivs(z, state)

    return val, {
        "lw": d_z * (np.pi * state["aq"]),
        "dw": d_z * (2j * np.pi * state["aq"]),
    }


__all__ = [
    "kernel",
    "kernel_from_z",
    "kernel_from_z_with_derivs",
    "kernel_with_derivs",
    "make_state",
]
