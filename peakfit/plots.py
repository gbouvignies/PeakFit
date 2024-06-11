import numpy as np
from lmfit.minimizer import MinimizerResult
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from peakfit.clustering import Cluster, Spectra
from peakfit.computing import calculate_shape_heights

# Constants
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
    # Calculate contour levels
    cl = contour_level * CONTOUR_FACTOR ** np.arange(CONTOUR_NUM)
    cl = np.concatenate((-cl[::-1], cl))

    # Find the spectrum with the more signal
    plane_index = np.linalg.norm(spectra.data[:, cluster.y, cluster.x], axis=1).argmax()

    data = spectra.data[plane_index]
    cluster_data = np.zeros_like(data)
    cluster_calc = np.zeros_like(data)

    nz, ny, nx = spectra.data.shape

    min_x, max_x = np.min(cluster.x), np.max(cluster.x)
    min_y, max_y = np.min(cluster.y), np.max(cluster.y)

    min_x -= max((max_x - min_x) // 5, 1)
    max_x += max((max_x - min_x) // 5, 1)
    min_y -= max((max_y - min_y) // 5, 1)
    max_y += max((max_y - min_y) // 5, 1)

    cluster_data[cluster.y, cluster.x] = data[cluster.y, cluster.x]
    shapes, amp_values = calculate_shape_heights(out.params, cluster)
    cluster_calc[cluster.y, cluster.x] = shapes.dot(amp_values)[:, 0]

    # Create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.set_size_inches(2*len(cluster.peaks),3*len(cluster.peaks))
    params = out.params
    peaks = cluster.peaks
    x_pts = [params[f"p{i}_x0_pt"].value for i in range(len(peaks))]
    y_pts = [params[f"p{i}_y0_pt"].value for i in range(len(peaks))]
    x_pts_init = peaks["x0_pt"] % nx
    y_pts_init = peaks["y0_pt"] % ny

    # Plot the contours
    ax.contour(cluster_data, cl, colors=CONTOUR_COLOR_1)
    ax.contour(cluster_calc, cl, colors=CONTOUR_COLOR_2)

    # Plot the peak positions
    ax.scatter(x_pts_init, y_pts_init, color=SCATTER_COLOR_1, s=20, label="Initial")
    ax.scatter(x_pts, y_pts, color=SCATTER_COLOR_2, s=20, label="Fit")
    for bla in range(len(peaks['name'])):
        plt.annotate(peaks['name'][bla],xy=(x_pts[bla],y_pts[bla]),size=10)  #adds the peak name next to the fitted peak position
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
    ax.set_title(", ".join(peaks["name"]))
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.legend()

    pdf.savefig()

    plt.close()
