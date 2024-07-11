from collections.abc import Sequence

import lmfit as lf
import numpy as np
import numpy.typing as npt

from peakfit.clustering import Cluster

FloatArray = npt.NDArray[np.float64]


def calculate_shapes(params: lf.Parameters, cluster: Cluster) -> FloatArray:
    return np.array(
        [peak.evaluate(cluster.positions, params) for peak in cluster.peaks]
    )


def calculate_amplitudes(shapes: FloatArray, data: FloatArray) -> FloatArray:
    return np.linalg.lstsq(shapes.T, data, rcond=None)[0]


def calculate_amplitudes_err(
    shapes: FloatArray, data: FloatArray
) -> tuple[FloatArray, FloatArray]:
    amplitudes, chi2, _rank, _s = np.linalg.lstsq(shapes.T, data, rcond=None)
    cov = np.linalg.pinv(shapes @ shapes.T)
    n = data.shape[0]
    k = amplitudes.shape[0]
    cov_scaled = cov * chi2.reshape(-1, 1, 1) / (n - k)
    amplitudes_err = np.sqrt(np.diagonal(cov_scaled, axis1=1, axis2=2))

    amplitudes_err = np.full_like(amplitudes_err, np.mean(amplitudes_err))

    return amplitudes, amplitudes_err.T


def calculate_shape_heights(
    params: lf.Parameters, cluster: Cluster
) -> tuple[FloatArray, FloatArray]:
    shapes = calculate_shapes(params, cluster)
    amplitudes = calculate_amplitudes(shapes, cluster.corrected_data)
    return shapes, amplitudes


def residuals(params: lf.Parameters, cluster: Cluster, noise: float) -> FloatArray:
    shapes, amplitudes = calculate_shape_heights(params, cluster)
    return (cluster.corrected_data - shapes.T @ amplitudes).ravel() / noise


def simulate_data(
    params: lf.Parameters, cluster: Cluster, data: FloatArray
) -> FloatArray:
    _shapes, amplitudes = calculate_shape_heights(params, cluster)
    cluster_all = Cluster.from_clusters([cluster])
    cluster_all.positions = [
        indices.ravel() for indices in list(np.indices(data.shape[1:]))
    ]
    shapes = calculate_shapes(params, cluster_all)

    return (amplitudes.T @ shapes).reshape(-1, *data.shape[1:])


def update_cluster_corrections(
    params: lf.Parameters, clusters: Sequence[Cluster]
) -> None:
    cluster_all = Cluster.from_clusters(clusters)
    _shapes_all, amplitudes_all = calculate_shape_heights(params, cluster_all)
    for cluster in clusters:
        indexes = [
            index
            for index, peak in enumerate(cluster_all.peaks)
            if peak not in cluster.peaks
        ]
        shapes = np.array(
            [
                cluster_all.peaks[index].evaluate(cluster.positions, params)
                for index in indexes
            ]
        ).T
        amplitudes = amplitudes_all[indexes, :]
        cluster.corrections = shapes @ amplitudes
