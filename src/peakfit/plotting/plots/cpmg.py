import argparse
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from peakfit.plotting.common import plot_wrapper
from peakfit.plotting.profiles import (
    intensity_to_r2eff,
    make_cpmg_figure,
    make_intensity_ensemble,
    ncyc_to_nu_cpmg,
)


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
    r2_ens = intensity_to_r2eff(
        make_intensity_ensemble(data_cpmg), make_intensity_ensemble(data_ref), args.time_t2
    )
    r2_erd, r2_eru = abs(np.percentile(r2_ens, [15.9, 84.1], axis=0) - r2_exp)
    return make_cpmg_figure(file.name, nu_cpmg, r2_exp, r2_erd, r2_eru)
