import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from peakfit.plotting.common import plot_wrapper

THRESHOLD = 1e4


def make_fig(
    name: str, offset: np.ndarray, intensity: np.ndarray, error: np.ndarray
) -> Figure:
    """Creates a figure for the CEST plot."""
    fig, ax = plt.subplots()
    ax.errorbar(offset, intensity, yerr=error, fmt=".")
    ax.set_title(name)
    ax.set_xlabel(r"$B_1$ offset (Hz)")
    ax.set_ylabel(r"$I/I_0$")
    plt.close()
    return fig


@plot_wrapper
def plot_cest(file: Path, args: argparse.Namespace) -> Figure:
    """Plots CEST data from a file."""
    offset, intensity, error = np.loadtxt(file, unpack=True)
    if args.ref == [-1]:
        ref = abs(offset) >= THRESHOLD
    else:
        ref = np.full_like(offset, fill_value=False, dtype=bool)
        ref[args.ref] = True

    intensity_ref = np.mean(intensity[ref])
    offset = offset[~ref]
    intensity = intensity[~ref] / intensity_ref
    error = error[~ref] / abs(intensity_ref)

    return make_fig(file.name, offset, intensity, error)
