import numpy as np
from lmfit.minimizer import MinimizerResult
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from peakfit.clustering import Cluster
from peakfit.computing import calculate_shape_heights
from peakfit.spectra import Spectra

# Constants
NDIM = 3
CONTOUR_NUM = 25
CONTOUR_FACTOR = 1.30
CONTOUR_COLOR_1 = "#FCB5AC"
CONTOUR_COLOR_2 = "#B5E5CF"
SCATTER_COLOR_1 = "#B99095"
SCATTER_COLOR_2 = "#3D5B59"
CHI2_BOX_PROPS = {"boxstyle": "round", "facecolor": "white", "alpha": 0.5}


def plot_pdf(
    pdf: PdfPages,
    spectra: Spectra,
    cluster: Cluster,
    contour_level: float,
    out: MinimizerResult,
) -> None:
    """Plot the spectra and cluster data with contour levels.

    Parameters:
    spectra (Spectra): The spectral data to be plotted.
    cluster (Cluster): The cluster information.
    contour_level (float): The starting contour level.
    out (MinimizerResult): The result from the minimizer containing fit parameters.
    """
    if spectra.data.ndim != NDIM:
        return

    # Calculate contour levels
    cl = contour_level * CONTOUR_FACTOR ** np.arange(CONTOUR_NUM)
    cl = np.concatenate((-cl[::-1], cl))

    y, x = cluster.positions

    # Find the spectrum with the more signal
    plane_index = np.linalg.norm(spectra.data[:, y, x], axis=1).argmax()

    data = spectra.data[plane_index]
    cluster_data = np.zeros_like(data)
    cluster_calc = np.zeros_like(data)

    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)

    min_x -= max((max_x - min_x) // 5, 1)
    max_x += max((max_x - min_x) // 5, 1)
    min_y -= max((max_y - min_y) // 5, 1)
    max_y += max((max_y - min_y) // 5, 1)

    cluster_data[y, x] = data[y, x]
    shapes, amp_values = calculate_shape_heights(out.params, cluster)
    cluster_calc[y, x] = shapes.dot(amp_values)[:, 0]

    # Create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    peaks = cluster.peaks
    positions = np.array([peak.positions for peak in peaks]).T
    positions_start = np.array([peak.positions_start for peak in peaks]).T

    y_pts, x_pts = (
        np.remainder(params.ppm2pts(position), params.size)
        for position, params in zip(positions, spectra.params[1:], strict=True)
    )
    y_pts_init, x_pts_init = (
        np.remainder(params.ppm2pts(position), params.size)
        for position, params in zip(positions_start, spectra.params[1:], strict=True)
    )

    # Plot the contours
    ax.contour(cluster_data, cl, colors=CONTOUR_COLOR_1)
    ax.contour(cluster_calc, cl, colors=CONTOUR_COLOR_2)

    # Plot the peak positions
    ax.scatter(x_pts_init, y_pts_init, color=SCATTER_COLOR_1, s=20, label="Initial")
    ax.scatter(x_pts, y_pts, color=SCATTER_COLOR_2, s=20, label="Fit")

    # Print chi2 and reduced chi2
    chi2red_str = r"$\chi^2_{red}$:"
    ax.text(
        0.05,
        0.95,
        f"{chi2red_str} {out.redchi:.2f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=CHI2_BOX_PROPS,
    )
    ax.text(
        0.85,
        0.05,
        f"Plane: {plane_index}\n",
        transform=ax.transAxes,
        verticalalignment="top",
    )

    # Decorate the axes
    ax.set_ylabel("F1 (pt)")
    ax.set_xlabel("F2 (pt)")
    ax.set_title(", ".join(peak.name for peak in peaks))
    ax.set_xlim(float(min_x), float(max_x))
    ax.set_ylim(float(min_y), float(max_y))
    ax.legend()

    pdf.savefig()

    plt.close()
