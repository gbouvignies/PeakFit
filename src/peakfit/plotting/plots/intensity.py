import argparse
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from peakfit.plotting.common import plot_wrapper
from peakfit.plotting.profiles import make_intensity_figure


@plot_wrapper
def plot_intensities(file: Path, _args: argparse.Namespace) -> Figure:
    """Plots intensity data from a file."""
    data = np.genfromtxt(file, dtype=None, names=("xlabel", "intensity", "error"))
    return make_intensity_figure(file.name, data)
