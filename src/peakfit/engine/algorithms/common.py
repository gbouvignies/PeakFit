from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.algorithms.linear_algebra import (
    calculate_amplitudes,
    calculate_amplitudes_with_uncertainty,
)
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.param_id import PSEUDO_AXIS, ParameterId

if TYPE_CHECKING:
    from collections.abc import Sequence

    from peakfit.engine.domain.params_scalar import Parameters
    from peakfit.shared.typing import FloatArray


def calculate_shape_heights(
    params: Parameters,
    cluster: Cluster,
) -> tuple[FloatArray, FloatArray]:
    """Calculate shapes and optimal amplitudes for a cluster.

    Returns:
    -------
        shapes: (n_peaks, n_points)
        amplitudes: (n_peaks, n_series)
    """
    shapes = cluster.evaluate(params)
    amplitudes = calculate_amplitudes(shapes, cluster.corrected_data)

    return shapes, amplitudes


def residuals(params: Parameters, cluster: Cluster, noise: float) -> FloatArray:
    """Compute residuals for fitting.

    Returns:
    -------
        1D array of residuals normalized by noise.
    """
    shapes, amplitudes = calculate_shape_heights(params, cluster)

    model = shapes.T @ amplitudes  # (n_points, n_series)

    data = cluster.corrected_data

    diff = data - model
    return diff.ravel() / noise


def update_cluster_corrections(
    params: Parameters,
    clusters: Sequence[Cluster],
) -> None:
    """Update cross-talk corrections for clusters."""
    cluster_list = list(clusters)
    if not cluster_list:
        return

    cluster_all = Cluster.from_clusters(cluster_list)
    # Underscore prefix indicates unused variable
    _shapes_all, amplitudes_all = calculate_shape_heights(params, cluster_all)
    # amplitudes_all shape: (n_total_peaks, n_series)

    for cluster in cluster_list:
        # Find which peaks are NOT in this cluster
        external_indices = [
            i for i, peak in enumerate(cluster_all.peaks) if peak not in cluster.peaks
        ]

        if not external_indices:
            cluster.corrections = np.zeros_like(cluster.data)
            continue

        external_peaks = [cluster_all.peaks[i] for i in external_indices]

        # Evaluate external peaks on THIS cluster's positions using the same
        # vectorized path as the main fit. This keeps 1-peak and N-peak behavior
        # identical and avoids accidental singleton dimensions.
        shapes_ext = Cluster.evaluate_peaks(
            external_peaks,
            params,
            cluster.grid_indices,
        )

        # Get their amplitudes
        amps_ext = amplitudes_all[external_indices, :]  # (n_ext, n_series)

        # Calculate contribution: (n_points, n_ext) @ (n_ext, n_series)
        # -> (n_points, n_series)
        contribution = shapes_ext.T @ amps_ext
        cluster.corrections = contribution.astype(float)


def inject_amplitude_parameters(
    params: Parameters,
    cluster: Cluster,
    noise: float,
) -> None:
    """Inject amplitude parameters from linear least-squares into params."""
    shapes = cluster.evaluate(params)
    amplitudes, errors, _covariance = calculate_amplitudes_with_uncertainty(
        shapes, cluster.corrected_data, noise
    )

    n_series = amplitudes.shape[1]

    for i, peak in enumerate(cluster.peaks):
        peak_error = float(errors[i])

        for j in range(n_series):
            if n_series == 1:
                amp_id = ParameterId(
                    peak_name=peak.name,
                    axis=PSEUDO_AXIS,
                    label="I",
                    index=0,
                )
            else:
                amp_id = ParameterId(
                    peak_name=peak.name,
                    axis=PSEUDO_AXIS,
                    label="I",
                    index=j,
                )

            val = float(amplitudes[i, j])

            if amp_id.name in params:
                p = params[amp_id.name]
                p.value = val
                p.vary = False
                p.computed = True
                p.stderr = peak_error
            else:
                params.add(
                    amp_id,
                    value=val,
                    vary=False,
                    computed=True,
                    stderr=peak_error,
                )
