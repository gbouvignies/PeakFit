import lmfit as lf
import numpy as np
import numpy.typing as npt

from peakfit.clustering import Cluster

FloatArray = npt.NDArray[np.float64]


def calculate_shape_heights(
    params: lf.Parameters, cluster: Cluster
) -> tuple[FloatArray, FloatArray]:
    shapes = np.array(
        [peak.evaluate(cluster.positions, params) for peak in cluster.peaks]
    ).T
    amp_values = np.linalg.lstsq(shapes, cluster.data, rcond=None)[0]
    return shapes, amp_values


def residuals(params: lf.Parameters, cluster: Cluster, noise: float) -> FloatArray:
    shapes, amplitudes = calculate_shape_heights(params, cluster)
    return (cluster.data - shapes @ amplitudes).ravel() / noise


def simulate_data(params: lf.Parameters, cluster: Cluster) -> FloatArray:
    shapes, amplitudes = calculate_shape_heights(params, cluster)
    return shapes @ amplitudes
