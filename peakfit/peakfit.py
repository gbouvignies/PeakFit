"""Main module for peak fitting."""

from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path

import lmfit as lf
import nmrglue as ng
import numpy as np
import numpy.typing as npt

from peakfit.cli import build_parser
from peakfit.clustering import Cluster, create_clusters
from peakfit.computing import (
    residuals,
    simulate_data,
    update_cluster_corrections,
)
from peakfit.messages import (
    export_html,
    print_fit_report,
    print_fitting,
    print_logo,
    print_peaks,
    print_refining,
    print_writing_spectra,
)
from peakfit.noise import prepare_noise_level
from peakfit.peak import create_params
from peakfit.peaklist import read_list
from peakfit.spectra import Spectra, get_shape_names, read_spectra
from peakfit.writing import write_profiles, write_shifts

FloatArray = npt.NDArray[np.float64]


def update_params(params: lf.Parameters, params_all: lf.Parameters) -> lf.Parameters:
    """Update the parameters with the global parameters."""
    for key in params:
        if key in params_all:
            params[key] = params_all[key]
    return params


def fit_clusters(clargs: Namespace, clusters: Sequence[Cluster]) -> lf.Parameters:
    """Fit all clusters and return shifts."""
    print_fitting()
    params_all = lf.Parameters()

    for index in range(clargs.refine_nb + 1):
        if index > 0:
            print_refining(index, clargs.refine_nb)
        for cluster in clusters:
            print_peaks(cluster.peaks)
            params = create_params(cluster.peaks, fixed=clargs.fixed)
            params = update_params(params, params_all)
            out = lf.minimize(residuals, params, args=(cluster, clargs.noise))
            print_fit_report(out)
            params_all.update(out.params)

        update_cluster_corrections(params_all, clusters)

    return params_all


def write_spectra(
    path: Path, spectra: Spectra, clusters: Sequence[Cluster], params: lf.Parameters
) -> None:
    print_writing_spectra()
    cluster_all = Cluster.from_clusters(clusters)

    data_simulated = simulate_data(params, cluster_all, spectra.data)

    ng.pipe.write(
        str(path / "simulated.ft2"),
        spectra.dic,
        (data_simulated).astype(np.float32),
        overwrite=True,
    )
    ng.pipe.write(
        str(path / "difference.ft2"),
        spectra.dic,
        (spectra.data - data_simulated).astype(np.float32),
        overwrite=True,
    )


def main() -> None:
    """Run peakfit."""
    print_logo()

    parser = build_parser()
    clargs = parser.parse_args()

    spectra = read_spectra(clargs.path_spectra, clargs.path_z_values, clargs.exclude)

    clargs.noise = prepare_noise_level(clargs, spectra)

    shape_names = get_shape_names(clargs, spectra)
    peaks = read_list(clargs.path_list, spectra, shape_names)

    clargs.contour_level = clargs.contour_level or 10.0 * clargs.noise
    clusters = create_clusters(spectra, peaks, clargs.contour_level)

    clargs.path_output.mkdir(parents=True, exist_ok=True)
    params = fit_clusters(clargs, clusters)

    write_profiles(clargs.path_output, spectra.z_values, clusters, params)

    export_html(clargs.path_output / "logs.html")

    write_shifts(peaks, params, clargs.path_output / "shifts.list")

    write_spectra(clargs.path_output, spectra, clusters, params)


if __name__ == "__main__":
    main()
