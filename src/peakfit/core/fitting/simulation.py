"""Simulate NMR spectra from fitted parameters."""

from collections.abc import Sequence

import numpy as np

from peakfit.core.domain.cluster import Cluster
from peakfit.core.fitting.computation import calculate_shape_heights
from peakfit.core.fitting.parameters import Parameters
from peakfit.core.shared.typing import FloatArray


def simulate_data(params: Parameters, clusters: Sequence[Cluster], data: FloatArray) -> FloatArray:
    """Simulate spectrum from fitted parameters.

    Args:
        params: Fitted parameter values
        clusters: List of fitted clusters
        data: Original data (for shape information)

    Returns:
        Simulated spectrum with same shape as input
    """
    amplitudes_list: list[FloatArray] = []
    for cluster in clusters:
        _shapes, amplitudes = calculate_shape_heights(params, cluster)
        amplitudes_list.append(amplitudes)
    amplitudes = np.concatenate(amplitudes_list)
    cluster_all = Cluster.from_clusters(clusters)
    cluster_all.positions = [indices.ravel() for indices in list(np.indices(data.shape[1:]))]

    return sum(
        (
            amplitudes[index][:, np.newaxis] * peak.evaluate(cluster_all.positions, params)
            for index, peak in enumerate(cluster_all.peaks)
        ),
        start=np.array(0.0),
    ).reshape(data.shape)
