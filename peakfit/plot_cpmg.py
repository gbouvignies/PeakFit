import argparse
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from peakfit.messages import print_filename, print_plotting, print_reading_files


def get_args():
    """Parse command line arguments for plotting CPMG R2eff profiles.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Plot CPMG R2eff profiles.")
    parser.add_argument("-f", "--files", nargs="+", type=pathlib.Path)
    parser.add_argument("-t", "--time_t2", type=float)
    return parser.parse_args()


def ncyc_to_nu_cpmg(ncyc, time_t2):
    """Convert a list of ncyc values to nu_cpmg values.

    Parameters:
    ncyc (list): A list of ncyc values.
    time_t2 (float): The time_t2 value.

    Returns:
    list: A list of nu_cpmg values.
    """
    nu_cpmg = []

    for a_ncyc in ncyc:
        if a_ncyc > 0.0:
            nu_cpmg.append(a_ncyc / time_t2)
        else:
            nu_cpmg.append(0.5 / time_t2)

    return nu_cpmg


def intensity_to_r2eff(intensity, intensity_ref, time_t2):
    """Convert intensity to R2eff.

    Parameters:
    intensity (float): The intensity value.
    intensity_ref (float): The reference intensity value.
    time_t2 (float): The T2 relaxation time.

    Returns:
    float: The R2eff value.
    """
    return -np.log(intensity / intensity_ref) / time_t2


def make_ens(data, size=1000):
    """Generate an ensemble of data points by adding random noise to the input data.

    Parameters:
    - data (dict): A dictionary containing the input data with keys "intensity" and "error".
    - size (int): The number of data points to generate in the ensemble. Default is 1000.

    Returns:
    - ensemble (ndarray): An array of shape (size, len(data["intensity"])) representing the ensemble of data points.
    """
    return data["intensity"] + data["error"] * np.random.randn(
        size,
        len(data["intensity"]),
    )


def make_fig(name, nu_cpmg, r2_exp, r2_erd, r2_eru):
    """Create a figure with errorbar plot of R2_eff values.

    Parameters:
    name (str): The title of the figure.
    nu_cpmg (array-like): Array of CPMG frequencies.
    r2_exp (array-like): Array of experimental R2_eff values.
    r2_erd (array-like): Array of lower error bounds for R2_eff values.
    r2_eru (array-like): Array of upper error bounds for R2_eff values.

    Returns:
    matplotlib.figure.Figure: The created figure object.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(nu_cpmg, r2_exp, yerr=(r2_erd, r2_eru), fmt="o")
    ax.set_title(name)
    ax.set_xlabel(r"$\nu_{CPMG}$ (Hz)")
    ax.set_ylabel(r"$R_{2,eff}$ (s$^{-1}$)")
    plt.close()

    return fig


def plot(files, time_t2) -> None:
    """Plot CPMG profiles.

    Args:
        files (list): List of file paths.
        time_t2 (float): Time constant T2.

    Returns:
        None
    """
    figs = {}

    print_reading_files()

    files_ordered = sorted(files, key=lambda x: int(re.sub(r"\D", "", str(x))))

    for a_file in files_ordered:
        print_filename(a_file)
        data = np.loadtxt(
            a_file,
            dtype={
                "names": ("ncyc", "intensity", "error"),
                "formats": ("i4", "f8", "f8"),
            },
        )
        data_ref = data[data["ncyc"] == 0]
        data_cpmg = data[data["ncyc"] != 0]
        intensity_ref = np.mean(data_ref["intensity"])
        error_ref = np.mean(data_ref["error"]) / np.sqrt(len(data_ref))
        nu_cpmg = ncyc_to_nu_cpmg(data_cpmg["ncyc"], time_t2)
        r2_exp = intensity_to_r2eff(data_cpmg["intensity"], intensity_ref, time_t2)
        r2_ens = intensity_to_r2eff(
            make_ens(data_cpmg),
            make_ens(
                {
                    "intensity": np.array([intensity_ref]),
                    "error": np.array([error_ref]),
                },
            ),
            time_t2,
        )
        r2_erd, r2_eru = abs(np.percentile(r2_ens, [15.9, 84.1], axis=0) - r2_exp)
        figs[a_file.name] = make_fig(a_file.name, nu_cpmg, r2_exp, r2_erd, r2_eru)

    print_plotting()

    with PdfPages("profiles.pdf") as pdf:
        for fig in figs.values():
            pdf.savefig(fig)


def main() -> None:
    """Entry point of the program.

    This function parses command line arguments using `get_args` function and calls `plot` function
    with the parsed arguments.

    Returns:
        None
    """
    args = get_args()
    plot(args.files, args.time_t2)
