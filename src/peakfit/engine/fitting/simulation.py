"""Simulate NMR spectra from fitted parameters."""

from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.algorithms.common import calculate_shape_heights
from peakfit.engine.domain.cluster import Cluster

if TYPE_CHECKING:
    from collections.abc import Sequence

    from peakfit.engine.domain.params_scalar import Parameters
    from peakfit.engine.results import FitResult
    from peakfit.shared.typing import FloatArray


def simulate_data(params: Parameters, clusters: Sequence[Cluster], data: FloatArray) -> FloatArray:
    """Simulate spectrum from fitted parameters.

    Args:
        params: Fitted parameter values
        clusters: List of fitted clusters
        data: Original data (for shape information)

    Returns:
    -------
        Simulated spectrum with same shape as input
    """
    amplitudes_list: list[FloatArray] = []

    # Calculate amplitudes for each cluster (on its segmented support)
    for cluster in clusters:
        _shapes, amplitudes = calculate_shape_heights(params, cluster)
        amplitudes_list.append(amplitudes)

    amplitudes = np.concatenate(amplitudes_list)
    cluster_all = Cluster.from_clusters(list(clusters))

    n_series = data.shape[0] if data.ndim > 1 else 1
    grid_shape = data.shape[1:]

    # Full-grid coordinate vectors: one coordinate per point (flattened)
    grid_indices = np.indices(grid_shape)
    positions = [idx.ravel() for idx in grid_indices]

    shapes_full = cluster_all.evaluate(params, positions)  # (n_peaks, n_points)

    # Ensure amplitudes is 2D (n_peaks, n_series)
    amp = np.asarray(amplitudes)
    if amp.ndim == 1:
        amp = amp[:, np.newaxis]
    if amp.shape[1] != n_series:
        amp = np.broadcast_to(amp, (amp.shape[0], n_series))

    # Model: (n_points, n_peaks) @ (n_peaks, n_series) -> (n_points, n_series)
    model = shapes_full.T @ amp

    # Return shape matches input (series, *grid)
    if n_series == 1:
        return model[:, 0].reshape((1, *grid_shape)).astype(float)

    return model.T.reshape((n_series, *grid_shape)).astype(float)


def simulate_from_result(
    result: FitResult, clusters: Sequence[Cluster], data: FloatArray
) -> FloatArray:
    """Simulate spectrum from a FitResult.

    Args:
        result: FitResult object containing parameters
        clusters: List of clusters
        data: Original data

    Returns:
    -------
        Simulated spectrum
    """
    return simulate_data(result.params, clusters, data)
