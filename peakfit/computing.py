from collections.abc import Sequence

import lmfit as lf
import numpy as np

from peakfit.clustering import Cluster
from peakfit.typing import FloatArray


def calculate_shapes(params: lf.Parameters, cluster: Cluster) -> FloatArray:
    return np.array(
        [peak.evaluate(cluster.positions, params) for peak in cluster.peaks]
    )


def calculate_amplitudes(shapes: FloatArray, data: FloatArray) -> FloatArray:
    return np.linalg.lstsq(shapes.T, data, rcond=None)[0]


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
    params: lf.Parameters, clusters: Sequence[Cluster], data: FloatArray
) -> FloatArray:
    amplitudes_list: list[FloatArray] = []
    for cluster in clusters:
        shapes, amplitudes = calculate_shape_heights(params, cluster)
        amplitudes_list.append(amplitudes)
    amplitudes = np.concatenate(amplitudes_list)
    cluster_all = Cluster.from_clusters(clusters)
    cluster_all.positions = [
        indices.ravel() for indices in list(np.indices(data.shape[1:]))
    ]

    return sum(
        (
            amplitudes[index][:, np.newaxis]
            * peak.evaluate(cluster_all.positions, params)
            for index, peak in enumerate(cluster_all.peaks)
        ),
        start=np.array(0.0),
    ).reshape(data.shape)


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
