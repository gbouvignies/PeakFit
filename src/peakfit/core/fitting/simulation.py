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

    Returns
    -------
        Simulated spectrum with same shape as input
    """
    amplitudes_list: list[FloatArray] = []
    for cluster in clusters:
        _shapes, amplitudes = calculate_shape_heights(params, cluster)
        amplitudes_list.append(amplitudes)
    amplitudes = np.concatenate(amplitudes_list)
    cluster_all = Cluster.from_clusters(clusters)
    cluster_all.positions = [indices.ravel() for indices in list(np.indices(data.shape[1:]))]

    # Build the simulated spectrum using in-place accumulation to avoid
    # large temporary arrays. The previous sum() construction could allocate
    # an array per peak before summing which causes large memory growth.
    simulated = np.zeros_like(data, dtype=float)

    n_planes = data.shape[0]
    grid_shape = data.shape[1:]
    n_points = int(np.prod(grid_shape))

    for index, peak in enumerate(cluster_all.peaks):
        amp = np.asarray(amplitudes[index])
        vals = np.asarray(peak.evaluate(cluster_all.positions, params))

        # Flatten vals if it represents the grid
        if vals.ndim == len(grid_shape):
            # e.g. vals is (rows, cols) -> flatten to (n_points,)
            vals_flat = vals.ravel()
        elif vals.ndim == 1:
            vals_flat = vals
        elif vals.ndim == 2 and vals.shape[0] == n_planes and vals.shape[1] == n_points:
            # Already in (n_planes, n_points) form
            vals_flat = None
        else:
            # Fallback: flatten everything
            vals_flat = vals.ravel()

        if vals_flat is not None:
            # vals_flat: (n_points,)
            if amp.ndim == 0:
                # scalar amplitude -> expand vals to full grid and broadcast along planes
                contrib = (amp * vals_flat).reshape((1, *grid_shape))
                simulated += np.broadcast_to(contrib, data.shape)
            else:
                # amp is vector (n_planes,) or (n_planes,1)
                amp_vec = amp.reshape(-1)
                if amp_vec.shape[0] != n_planes:
                    # if amplitude shape doesn't match planes, try align
                    amp_vec = np.broadcast_to(amp_vec, (n_planes,))
                prod = amp_vec.reshape(-1, 1) * vals_flat[None, :]
                simulated += prod.reshape((n_planes, *grid_shape))
        else:
            # vals already shaped as (n_planes, n_points)
            if vals.shape[0] == n_planes and vals.shape[1] == n_points:
                if amp.ndim == 0:
                    simulated += amp * vals.reshape((n_planes, *grid_shape))
                else:
                    amp_vec = amp.reshape(-1)
                    if amp_vec.shape[0] != n_planes:
                        amp_vec = np.broadcast_to(amp_vec, (n_planes,))
                    prod = amp_vec.reshape(-1, 1) * vals
                    simulated += prod.reshape((n_planes, *grid_shape))
            else:
                # As a robust fallback, reshape vals to grid and multiply
                v = vals.ravel()[:n_points]
                if amp.ndim == 0:
                    simulated += (amp * v).reshape((1, *grid_shape))
                else:
                    amp_vec = amp.reshape(-1)
                    amp_vec = np.broadcast_to(amp_vec, (n_planes,))
                    prod = amp_vec.reshape(-1, 1) * v[None, :]
                    simulated += prod.reshape((n_planes, *grid_shape))

    return simulated.reshape(data.shape)
