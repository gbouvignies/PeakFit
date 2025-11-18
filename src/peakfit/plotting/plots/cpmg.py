import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from peakfit.plotting.common import plot_wrapper


def ncyc_to_nu_cpmg(ncyc: np.ndarray, time_t2: float) -> np.ndarray:
    """Converts ncyc values to nu_CPMG values."""
    return np.where(ncyc > 0, ncyc / time_t2, 0.5 / time_t2)


def intensity_to_r2eff(
    intensity: np.ndarray, intensity_ref: np.ndarray | float, time_t2: float
) -> np.ndarray:
    """Converts intensity values to R2 effective values."""
    return -np.log(intensity / intensity_ref) / time_t2


def make_ens(data: np.ndarray, size: int = 1000) -> np.ndarray:
    """Generates an ensemble of intensity values."""
    rng = np.random.default_rng()
    return data["intensity"] + data["error"] * rng.standard_normal(
        (size, len(data["intensity"]))
    )


def make_fig(
    name: str,
    nu_cpmg: np.ndarray,
    r2_exp: np.ndarray,
    r2_erd: np.ndarray,
    r2_eru: np.ndarray,
) -> Figure:
    """Creates a figure for the CPMG plot."""
    fig, ax = plt.subplots()
    ax.errorbar(nu_cpmg, r2_exp, yerr=(r2_erd, r2_eru), fmt="o")
    ax.set_title(name)
    ax.set_xlabel(r"$\nu_{CPMG}$ (Hz)")
    ax.set_ylabel(r"$R_{2,eff}$ (s$^{-1}$)")
    plt.close()
    return fig


@plot_wrapper
def plot_cpmg(file: Path, args: argparse.Namespace) -> Figure:
    """Plots CPMG data from a file."""
    data = np.loadtxt(
        file,
        dtype={"names": ("ncyc", "intensity", "error"), "formats": ("i4", "f8", "f8")},
    )
    data_ref = data[data["ncyc"] == 0]
    data_cpmg = data[data["ncyc"] != 0]
    intensity_ref = float(np.mean(data_ref["intensity"]))
    error_ref = np.mean(data_ref["error"]) / np.sqrt(len(data_ref))
    nu_cpmg = ncyc_to_nu_cpmg(data_cpmg["ncyc"], args.time_t2)
    r2_exp = intensity_to_r2eff(data_cpmg["intensity"], intensity_ref, args.time_t2)
    data_ref = np.array(
        [(intensity_ref, error_ref)], dtype=[("intensity", float), ("error", float)]
    )
    r2_ens = intensity_to_r2eff(make_ens(data_cpmg), make_ens(data_ref), args.time_t2)
    r2_erd, r2_eru = abs(np.percentile(r2_ens, [15.9, 84.1], axis=0) - r2_exp)
    return make_fig(file.name, nu_cpmg, r2_exp, r2_erd, r2_eru)
