"""Main module."""

from argparse import Namespace
from collections.abc import Sequence

import lmfit as lf
import numpy as np
import numpy.typing as npt
from matplotlib.backends.backend_pdf import PdfPages

from peakfit.cli import build_parser
from peakfit.clustering import Cluster, create_clusters
from peakfit.computing import calculate_shape_heights, residuals
from peakfit.messages import (
    export_html,
    print_estimated_noise,
    print_fit_report,
    print_fitting,
    print_logo,
    print_peaks,
)
from peakfit.noise import estimate_noise, monte_carlo
from peakfit.peak import create_params
from peakfit.peaklist import read_list
from peakfit.plots import plot_pdf
from peakfit.spectra import Spectra, get_shape_names, read_spectra
from peakfit.writing import write_profiles, write_shifts

FloatArray = npt.NDArray[np.float64]


def prepare_noise_level(clargs: Namespace, spectra: Spectra) -> float:
    """Prepare the noise level for fitting."""
    if clargs.noise is not None and clargs.noise < 0.0:
        clargs.noise = None

    if clargs.noise is None:
        clargs.noise = estimate_noise(spectra.data)
        print_estimated_noise(clargs.noise)

    return clargs.noise


def calculate_errors(
    clargs: Namespace,
    spectra: Spectra,
    cluster: Cluster,
    out: lf.minimizer.MinimizerResult,
    heights: FloatArray,
) -> tuple[dict, FloatArray]:
    """Calculate parameter and height errors."""
    params_err = {param.name: param.stderr for param in out.params.values()}
    height_err = np.full_like(heights, clargs.noise)

    if clargs.mc and int(clargs.mc[4]) > 1:
        params_err, height_err = monte_carlo(
            clargs.mc, spectra, cluster, out, clargs.noise
        )
        for param in out.params.values():
            param.stderr = params_err[param.name]

    return params_err, height_err


def run_cluster_fit(
    clargs: Namespace, spectra: Spectra, cluster: Cluster, shifts: dict, pdf: PdfPages
) -> None:
    """Run the fitting process for a single cluster."""
    print_peaks(cluster.peaks)
    params = create_params(cluster.peaks, fixed=clargs.fixed)

    out = lf.minimize(
        residuals,
        params,
        args=(cluster, clargs.noise),
        method="least_squares",
        verbose=2,
    )

    _, heights = calculate_shape_heights(out.params, cluster)
    print_fit_report(out)

    params_err, height_err = calculate_errors(clargs, spectra, cluster, out, heights)

    for peak in cluster.peaks:
        peak.update_positions(out.params)

    plot_pdf(pdf, spectra, cluster, clargs.contour_level, out)
    shifts.update({peak.name: peak.positions for peak in cluster.peaks})

    write_profiles(
        clargs.path_output,
        spectra.z_values,
        cluster,
        out.params,
        heights,
        height_err,
    )


def run_fit(clargs: Namespace, spectra: Spectra, clusters: Sequence[Cluster]) -> dict:
    """Run the fitting process for all clusters."""
    print_fitting()
    shifts: dict[str, FloatArray] = {}
    with PdfPages(clargs.path_output / "clusters.pdf") as pdf:
        for cluster in clusters:
            run_cluster_fit(clargs, spectra, cluster, shifts, pdf)
    return shifts


def main() -> None:
    """Run peakfit."""
    print_logo()

    parser = build_parser()
    clargs = parser.parse_args()

    spectra = read_spectra(clargs.path_spectra, clargs.path_z_values, clargs.exclude)
    shape_names = get_shape_names(clargs, spectra)

    clargs.noise = prepare_noise_level(clargs, spectra)
    peaks = read_list(clargs.path_list, spectra, shape_names)

    clargs.contour_level = clargs.contour_level or 5.0 * clargs.noise
    clusters = create_clusters(spectra, peaks, clargs.contour_level)

    clargs.path_output.mkdir(parents=True, exist_ok=True)
    shifts = run_fit(clargs, spectra, clusters)

    export_html(clargs.path_output / "logs.html")
    write_shifts(peaks, shifts, clargs.path_output / "shifts.list")


if __name__ == "__main__":
    main()
