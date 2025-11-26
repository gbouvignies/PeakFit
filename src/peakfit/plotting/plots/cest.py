import argparse
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from peakfit.plotting.common import plot_wrapper
from peakfit.plotting.profiles import make_cest_figure

THRESHOLD = 1e4


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

    return make_cest_figure(file.name, offset, intensity, error)
