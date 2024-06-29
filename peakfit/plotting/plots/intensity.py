import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from peakfit.plotting.common import plot_wrapper


def make_fig(name: str, data: np.ndarray) -> Figure:
    """Creates a figure for the intensity plot."""
    fig, ax = plt.subplots()
    ax.errorbar(data["xlabel"], data["intensity"], yerr=data["error"], fmt=".")
    ax.set_title(name)
    ax.set_ylabel(r"Intensities")
    plt.close()
    return fig


@plot_wrapper
def plot_intensities(file: Path, _args: argparse.Namespace) -> Figure:
    """Plots intensity data from a file."""
    data = np.genfromtxt(file, dtype=None, names=("xlabel", "intensity", "error"))
    return make_fig(file.name, data)
