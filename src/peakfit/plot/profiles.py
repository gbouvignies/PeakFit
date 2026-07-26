"""Profile visualization functions for PeakFit."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.figure import Figure


def make_intensity_figure(name: str, data: np.ndarray) -> Figure:
    """Create intensity profile plot.

    Args:
        name: Peak/cluster name for title
        data: Structured array with 'xlabel', 'intensity', 'error' fields

    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(data["xlabel"], data["intensity"], yerr=data["error"], fmt=".", markersize=8)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_ylabel("Intensity", fontsize=11)
    ax.set_xlabel("Index", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def make_cest_figure(name: str, data: np.ndarray) -> Figure:
    """Create a CEST profile plot from normalized intensity data."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        data["offset"],
        data["intensity"],
        yerr=data["error"],
        fmt=".",
        markersize=8,
        capsize=3,
    )
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel(r"$B_1$ offset (Hz)", fontsize=11)
    ax.set_ylabel(r"$I/I_0$", fontsize=11)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def make_cpmg_figure(name: str, data: np.ndarray) -> Figure:
    """Create a CPMG relaxation-dispersion plot from R2eff data."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        data["nu_cpmg"],
        data["r2eff"],
        yerr=data["error"],
        fmt="o",
        markersize=8,
        capsize=3,
    )
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel(r"$\nu_{CPMG}$ (Hz)", fontsize=11)
    ax.set_ylabel(r"$R_{2,\mathrm{eff}}$ (s$^{-1}$)", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
