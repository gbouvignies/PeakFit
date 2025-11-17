"""Main module for peak fitting."""

import multiprocessing as mp
from collections.abc import Sequence
from pathlib import Path

import nmrglue as ng
import numpy as np
from scipy.optimize import least_squares

from peakfit.cli_legacy import Arguments, parse_args
from peakfit.clustering import Cluster, create_clusters
from peakfit.computing import (
    residuals,
    simulate_data,
    update_cluster_corrections,
)
from peakfit.core.fast_fit import fit_clusters_fast
from peakfit.core.fitting import Parameters
from peakfit.core.parallel import fit_clusters_parallel_refined
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


def update_params(params: Parameters, params_all: Parameters) -> Parameters:
    """Update the parameters with the global parameters."""
    for key in params:
        if key in params_all:
            params[key] = params_all[key]
    return params


def _residual_wrapper(x: np.ndarray, params: Parameters, cluster: Cluster, noise: float) -> np.ndarray:
    """Wrapper to convert array to Parameters for residual calculation."""
    vary_names = params.get_vary_names()
    for i, name in enumerate(vary_names):
        params[name].value = x[i]
    return residuals(params, cluster, noise)


def fit_clusters(clargs: Arguments, clusters: Sequence[Cluster]) -> Parameters:
    """Fit all clusters and return shifts."""
    print_fitting()
    params_all = Parameters()

    for index in range(clargs.refine_nb + 1):
        if index > 0:
            print_refining(index, clargs.refine_nb)
            update_cluster_corrections(params_all, clusters)
        for cluster in clusters:
            print_peaks(cluster.peaks)
            params = create_params(cluster.peaks, fixed=clargs.fixed)
            params = update_params(params, params_all)

            # Get varying parameters
            vary_names = params.get_vary_names()
            x0 = params.get_vary_values()
            bounds_lower = np.array([params[name].min for name in vary_names])
            bounds_upper = np.array([params[name].max for name in vary_names])

            # Run optimization with scipy.optimize.least_squares
            result = least_squares(
                _residual_wrapper,
                x0,
                args=(params, cluster, clargs.noise),
                bounds=(bounds_lower, bounds_upper),
                ftol=1e-7,
                xtol=1e-7,
                max_nfev=1000,
                verbose=2,
            )

            # Update parameters with optimized values
            for i, name in enumerate(vary_names):
                params[name].value = result.x[i]

            print_fit_report(result)
            params_all.update(params)

    return params_all


def write_spectra(
    path: Path, spectra: Spectra, clusters: Sequence[Cluster], params: Parameters
) -> None:
    """Write simulated spectra to a file.

    Args:
        path (Path): The path to the file where the spectra will be written.
        spectra (Spectra): The original spectra.
        clusters (Sequence[Cluster]): The clusters used for simulation.
        params (lf.Parameters): The parameters used for simulation.

    Returns:
        None
    """
    print_writing_spectra()

    data_simulated = simulate_data(params, clusters, spectra.data)

    if spectra.pseudo_dim_added:
        data_simulated = np.squeeze(data_simulated, axis=0)

    ng.pipe.write(
        str(path / f"simulated.ft{data_simulated.ndim}"),
        spectra.dic,
        (data_simulated).astype(np.float32),
        overwrite=True,
    )


def main() -> None:
    """Run peakfit."""
    print_logo()

    clargs = parse_args()

    spectra = read_spectra(clargs.path_spectra, clargs.path_z_values, clargs.exclude)

    clargs.noise = prepare_noise_level(clargs, spectra)

    shape_names = get_shape_names(clargs, spectra)
    peaks = read_list(spectra, shape_names, clargs)

    clargs.contour_level = clargs.contour_level or 5.0 * clargs.noise
    clusters = create_clusters(spectra, peaks, clargs.contour_level)

    # Choose fitting method based on --parallel and --fast flags
    if clargs.parallel and len(clusters) > 1:
        n_workers = clargs.n_workers or mp.cpu_count()
        print(f"\nParallel fitting enabled: {len(clusters)} clusters on {n_workers} workers")
        print("Using fast scipy optimization (bypasses lmfit overhead)")
        params = fit_clusters_parallel_refined(
            clusters=clusters,
            noise=clargs.noise,
            refine_iterations=clargs.refine_nb,
            fixed=clargs.fixed,
            n_workers=n_workers,
            verbose=True,
        )
    elif clargs.fast:
        print("\nFast scipy optimization enabled (bypasses lmfit overhead)")
        params = fit_clusters_fast(
            clusters=list(clusters),
            noise=clargs.noise,
            refine_iterations=clargs.refine_nb,
            fixed=clargs.fixed,
            verbose=True,
        )
    else:
        params = fit_clusters(clargs, clusters)

    clargs.path_output.mkdir(parents=True, exist_ok=True)

    write_profiles(clargs.path_output, spectra.z_values, clusters, params, clargs)

    export_html(clargs.path_output / "logs.html")

    write_shifts(peaks, params, clargs.path_output / "shifts.list")

    write_spectra(clargs.path_output, spectra, clusters, params)


if __name__ == "__main__":
    main()
