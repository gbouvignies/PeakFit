"""Core computational functions for peak fitting."""

from collections.abc import Sequence

import numpy as np

from peakfit.core.domain.cluster import Cluster
from peakfit.core.fitting.parameters import Parameters
from peakfit.core.shared.typing import FloatArray


def calculate_shapes(params: Parameters, cluster: Cluster) -> FloatArray:
    """Calculate shapes for all peaks in a cluster.

    Args:
        params: Current parameter values
        cluster: Cluster containing peaks

    Returns:
        Array of shape (n_peaks, n_points) with evaluated lineshapes
    """
    return np.array([peak.evaluate(cluster.positions, params) for peak in cluster.peaks])


def calculate_amplitudes(shapes: FloatArray, data: FloatArray) -> FloatArray:
    """Calculate peak amplitudes via linear least squares.

    Args:
        shapes: Peak lineshapes, shape (n_peaks, n_points)
        data: Data to fit, shape (n_points,) or (n_planes, n_points)

    Returns:
        Optimal amplitudes for each peak
    """
    return np.linalg.lstsq(shapes.T, data, rcond=None)[0]


def calculate_shape_heights(params: Parameters, cluster: Cluster) -> tuple[FloatArray, FloatArray]:
    """Calculate shapes and optimal amplitudes for a cluster.

    Args:
        params: Current parameter values
        cluster: Cluster to analyze

    Returns:
        Tuple of (shapes, amplitudes)
    """
    shapes = calculate_shapes(params, cluster)
    amplitudes = calculate_amplitudes(shapes, cluster.corrected_data)
    return shapes, amplitudes


def residuals(params: Parameters, cluster: Cluster, noise: float) -> FloatArray:
    """Compute residuals for fitting.

    Args:
        params: Current parameter values
        cluster: Cluster being fitted
        noise: Noise level for normalization

    Returns:
        Residual array normalized by noise
    """
    shapes, amplitudes = calculate_shape_heights(params, cluster)
    return (cluster.corrected_data - shapes.T @ amplitudes).ravel() / noise


def update_cluster_corrections(params: Parameters, clusters: Sequence[Cluster]) -> None:
    """Update cross-talk corrections for clusters.

    This function computes the contribution of peaks from other clusters
    to each cluster's data, allowing for iterative refinement of fits.

    Args:
        params: Current parameter values (all clusters)
        clusters: List of clusters to update
    """
    cluster_list = list(clusters)
    cluster_all = Cluster.from_clusters(cluster_list)
    _shapes_all, amplitudes_all = calculate_shape_heights(params, cluster_all)
    for cluster in cluster_list:
        indexes = [
            index for index, peak in enumerate(cluster_all.peaks) if peak not in cluster.peaks
        ]
        shapes = np.array(
            [cluster_all.peaks[index].evaluate(cluster.positions, params) for index in indexes]
        ).T
        amplitudes = amplitudes_all[indexes, :]
        cluster.corrections = shapes @ amplitudes
