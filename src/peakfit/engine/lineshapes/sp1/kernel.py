"""SP1 kernel implementations."""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from peakfit.shared.typing import ComplexArray


def make_state(aq: float, apodq1: float, apodq2: float) -> dict[str, complex | float]:
    """Precompute SP1 constants used by the kernel."""
    f1 = apodq1 * np.pi
    f2 = (apodq2 - apodq1) * np.pi

    eif1 = np.exp(1j * f1)
    emif1 = np.exp(-1j * f1)
    eif2 = np.exp(1j * f2)
    emif2 = np.exp(-1j * f2)

    return {
        "aq": aq,
        "half_i_aq": 0.5j * aq,
        "if2": 1j * f2,
        "eif1": eif1,
        "emif1": emif1,
        "eif2": eif2,
        "emif2": emif2,
        "eif12": eif1 * eif2,
        "emif12": emif1 * emif2,
    }


def kernel_from_z(z: npt.ArrayLike, state: dict[str, complex | float]) -> ComplexArray:
    """Compute SP1 complex kernel."""
    z_arr = np.asarray(z, dtype=np.complex128)
    emz = np.exp(-z_arr)
    ez = 1.0 / emz

    denom1 = z_arr - state["if2"]
    denom2 = z_arr + state["if2"]

    num1 = (state["eif2"] - ez) * state["eif1"] * emz
    num2 = (ez - state["emif2"]) * state["emif1"] * emz

    term1 = state["half_i_aq"] * num1 / denom1
    term2 = state["half_i_aq"] * num2 / denom2

    result: ComplexArray = term1 + term2
    return result


def kernel_from_z_with_derivs(
    z: npt.ArrayLike, state: dict[str, complex | float]
) -> tuple[ComplexArray, ComplexArray]:
    """Compute SP1 kernel and dF/dz."""
    z_arr = np.asarray(z, dtype=np.complex128)
    emz = np.exp(-z_arr)
    ez = 1.0 / emz

    denom1 = z_arr - state["if2"]
    denom2 = z_arr + state["if2"]

    num1 = (state["eif2"] - ez) * state["eif1"] * emz
    num2 = (ez - state["emif2"]) * state["emif1"] * emz

    term1 = state["half_i_aq"] * num1 / denom1
    term2 = state["half_i_aq"] * num2 / denom2
    kernel_val = term1 + term2

    dnum1_dz = -state["eif12"] * emz
    dnum2_dz = state["emif12"] * emz

    dterm1_dz = state["half_i_aq"] * (dnum1_dz * denom1 - num1) / (denom1 * denom1)
    dterm2_dz = state["half_i_aq"] * (dnum2_dz * denom2 - num2) / (denom2 * denom2)

    df_dz = dterm1_dz + dterm2_dz
    return kernel_val, df_dz


def kernel(dw: npt.ArrayLike, lw: npt.ArrayLike, state: dict[str, complex | float]) -> ComplexArray:
    """Compute SP1 kernel for a given dw and lw.

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
    """Compute SP1 kernel and derivatives for apply_phase.

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
