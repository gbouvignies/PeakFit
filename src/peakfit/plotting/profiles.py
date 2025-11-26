"""Profile visualization functions for PeakFit.

This module provides plotting functions for NMR profile data:
- Intensity profiles
- CEST profiles
- CPMG relaxation dispersion profiles

All functions return matplotlib Figure objects for flexible usage.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from peakfit.core.shared.typing import FloatArray


def make_intensity_figure(name: str, data: np.ndarray) -> Figure:
    """Create intensity profile plot.

    Args:
        name: Peak/cluster name for title
        data: Structured array with 'xlabel', 'intensity', 'error' fields

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(data["xlabel"], data["intensity"], yerr=data["error"], fmt=".", markersize=8)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_ylabel("Intensity", fontsize=11)
    ax.set_xlabel("Index", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def make_cest_figure(
    name: str,
    offset: FloatArray,
    intensity: FloatArray,
    error: FloatArray,
) -> Figure:
    """Create CEST profile plot.

    Args:
        name: Peak/cluster name for title
        offset: B1 offset frequencies (Hz)
        intensity: Normalized intensities (I/I0)
        error: Intensity errors

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(offset, intensity, yerr=error, fmt=".", markersize=8, capsize=3)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel(r"$B_1$ offset (Hz)", fontsize=11)
    ax.set_ylabel(r"$I/I_0$", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig


def make_cpmg_figure(
    name: str,
    nu_cpmg: FloatArray,
    r2_exp: FloatArray,
    r2_err_down: FloatArray,
    r2_err_up: FloatArray,
) -> Figure:
    """Create CPMG relaxation dispersion plot.

    Args:
        name: Peak/cluster name for title
        nu_cpmg: CPMG frequencies (Hz)
        r2_exp: Experimental R2eff values (s^-1)
        r2_err_down: Lower error bounds
        r2_err_up: Upper error bounds

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(nu_cpmg, r2_exp, yerr=(r2_err_down, r2_err_up), fmt="o", markersize=8, capsize=3)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel(r"$\nu_{CPMG}$ (Hz)", fontsize=11)
    ax.set_ylabel(r"$R_{2,\mathrm{eff}}$ (s$^{-1}$)", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ==================== CPMG HELPER FUNCTIONS ====================


def ncyc_to_nu_cpmg(ncyc: FloatArray, time_t2: float) -> FloatArray:
    """Convert ncyc values to nu_CPMG values.

    Args:
        ncyc: Number of CPMG cycles
        time_t2: T2 relaxation time in seconds

    Returns:
        CPMG frequencies in Hz
    """
    return np.where(ncyc > 0, ncyc / time_t2, 0.5 / time_t2)


def intensity_to_r2eff(
    intensity: FloatArray,
    intensity_ref: FloatArray | float,
    time_t2: float,
) -> FloatArray:
    """Convert intensity values to R2 effective values.

    Args:
        intensity: Measured intensities
        intensity_ref: Reference intensity (at ncyc=0)
        time_t2: T2 relaxation time in seconds

    Returns:
        R2eff values in s^-1
    """
    return -np.log(intensity / intensity_ref) / time_t2


def make_intensity_ensemble(data: np.ndarray, size: int = 1000) -> FloatArray:
    """Generate ensemble of intensity values for error estimation.

    Args:
        data: Structured array with 'intensity' and 'error' fields
        size: Number of ensemble members

    Returns:
        Array of shape (size, n_points) with sampled intensities
    """
    rng = np.random.default_rng()
    return data["intensity"] + data["error"] * rng.standard_normal((size, len(data["intensity"])))
