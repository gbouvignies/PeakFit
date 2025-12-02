"""Core computational functions for peak fitting."""

from collections.abc import Sequence

import numpy as np

from peakfit.core.domain.cluster import Cluster
from peakfit.core.fitting.parameters import PSEUDO_AXIS, ParameterId, Parameters
from peakfit.core.shared.typing import FloatArray


def calculate_shapes(params: Parameters, cluster: Cluster) -> FloatArray:
    """Calculate shapes for all peaks in a cluster.

    Args:
        params: Current parameter values
        cluster: Cluster containing peaks

    Returns
    -------
        Array of shape (n_peaks, n_points) with evaluated lineshapes
    """
    return np.array([peak.evaluate(cluster.positions, params) for peak in cluster.peaks])


def calculate_amplitudes(shapes: FloatArray, data: FloatArray) -> FloatArray:
    """Calculate peak amplitudes via linear least squares.

    Uses normal equations (S @ S.T) @ a = S @ data, which is faster than
    np.linalg.lstsq when n_peaks << n_points (typical case: ~5 peaks, ~10k points).
    Falls back to lstsq for numerically singular cases.

    Args:
        shapes: Peak lineshapes, shape (n_peaks, n_points)
        data: Data to fit, shape (n_points,) or (n_points, n_planes)

    Returns
    -------
        Optimal amplitudes for each peak, shape (n_peaks,) or (n_peaks, n_planes)
    """
    # Normal equations: (S @ S.T) @ a = S @ data
    # This is ~3-8x faster than lstsq when n_peaks << n_points
    sts = shapes @ shapes.T  # (n_peaks, n_peaks)
    rhs = shapes @ data  # (n_peaks,) or (n_peaks, n_planes)
    try:
        return np.linalg.solve(sts, rhs)
    except np.linalg.LinAlgError:
        # Fallback to lstsq for singular matrices
        return np.linalg.lstsq(shapes.T, data, rcond=None)[0]


def calculate_amplitude_covariance(shapes: FloatArray, noise: float) -> FloatArray:
    """Calculate covariance matrix for amplitudes from linear least squares.

    For the linear model data = shapes.T @ amplitudes + noise, the covariance
    of the amplitudes is given by: Cov(a) = (S^T S)^{-1} * sigma^2

    Args:
        shapes: Peak lineshapes, shape (n_peaks, n_points)
        noise: Standard deviation of the noise

    Returns
    -------
        Covariance matrix for amplitudes, shape (n_peaks, n_peaks)
    """
    # shapes.T has shape (n_points, n_peaks)
    # S^T S has shape (n_peaks, n_peaks)
    sts = shapes @ shapes.T
    try:
        sts_inv = np.linalg.inv(sts)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse for singular matrices
        sts_inv = np.linalg.pinv(sts)
    return sts_inv * (noise**2)


def calculate_amplitudes_with_uncertainty(
    shapes: FloatArray, data: FloatArray, noise: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Calculate amplitudes and their uncertainties via linear least squares.

    Args:
        shapes: Peak lineshapes, shape (n_peaks, n_points)
        data: Data to fit, shape (n_points,) or (n_planes, n_points)
        noise: Standard deviation of the noise

    Returns
    -------
        Tuple of (amplitudes, amplitude_errors, amplitude_covariance)
        - amplitudes: shape (n_peaks,) or (n_peaks, n_planes)
        - amplitude_errors: shape (n_peaks,) standard errors
        - amplitude_covariance: shape (n_peaks, n_peaks) covariance matrix
    """
    amplitudes = calculate_amplitudes(shapes, data)
    covariance = calculate_amplitude_covariance(shapes, noise)
    errors = np.sqrt(np.diag(covariance))
    return amplitudes, errors, covariance


def calculate_shape_heights(params: Parameters, cluster: Cluster) -> tuple[FloatArray, FloatArray]:
    """Calculate shapes and optimal amplitudes for a cluster.

    Args:
        params: Current parameter values
        cluster: Cluster to analyze

    Returns
    -------
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

    Returns
    -------
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


def inject_amplitude_parameters(
    params: Parameters,
    cluster: Cluster,
    noise: float,
) -> None:
    """Inject amplitude parameters computed via linear least-squares.

    Computes amplitudes and their uncertainties analytically and adds them
    as computed parameters (computed=True) to the Parameters collection.
    This allows amplitudes to be included in statistics and reporting while
    remaining excluded from nonlinear optimization.

    The amplitude naming convention uses ParameterId for consistency:
    "{peak_name}.I[{plane_idx}]"

    Args:
        params: Parameters collection to update in-place
        cluster: Cluster containing peaks and data
        noise: Noise standard deviation for uncertainty estimation

    Note:
        This function modifies params in-place. Amplitudes are added with
        param_type=AMPLITUDE and computed=True.
    """
    shapes = calculate_shapes(params, cluster)
    amplitudes, errors, _covariance = calculate_amplitudes_with_uncertainty(
        shapes, cluster.corrected_data, noise
    )

    n_planes = cluster.corrected_data.shape[0] if cluster.corrected_data.ndim > 1 else 1

    for i, peak in enumerate(cluster.peaks):
        peak_amplitudes = amplitudes[i]
        peak_error = errors[i]  # Same error for all planes (from covariance diagonal)

        if n_planes == 1:
            # Single plane case
            amp_value = (
                float(peak_amplitudes)
                if np.ndim(peak_amplitudes) == 0
                else float(peak_amplitudes[0])
            )
            amp_id = ParameterId.amplitude(peak.name, PSEUDO_AXIS)
            params.add(
                amp_id,
                value=amp_value,
                min=-np.inf,  # Allow negative amplitudes (e.g., CEST, anti-phase)
                max=np.inf,
                vary=False,
                computed=True,
            )
            params[amp_id.name].stderr = float(peak_error)
        else:
            # Multi-plane case
            for j in range(n_planes):
                amp_id = ParameterId.amplitude(peak.name, PSEUDO_AXIS, j)
                params.add(
                    amp_id,
                    value=float(peak_amplitudes[j]),
                    min=-np.inf,  # Allow negative amplitudes (e.g., CEST, anti-phase)
                    max=np.inf,
                    vary=False,
                    computed=True,
                )
                params[amp_id.name].stderr = float(peak_error)
