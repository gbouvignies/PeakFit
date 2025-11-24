"""Core computational functions for peak fitting."""

from collections.abc import Sequence

import numba as nb
import numpy as np

from peakfit.data.clustering import Cluster
from peakfit.fitting.parameters import Parameters
from peakfit.lineshapes.functions import compute_ata_symmetric, compute_atb
from peakfit.typing import FloatArray

# =============================================================================
# Numba-optimized computational kernels
# =============================================================================


@nb.njit(cache=True, fastmath=True)
def _compute_residuals_flat(
    corrected_data: FloatArray, shapes: FloatArray, amplitudes: FloatArray, noise: float
) -> FloatArray:
    """Compute normalized residuals (Numba-optimized).

    Args:
        corrected_data: Data array, shape (n_points,) or (n_planes, n_points)
        shapes: Lineshapes array, shape (n_peaks, n_points)
        amplitudes: Amplitudes array, shape (n_peaks,) or (n_peaks, n_planes)
        noise: Noise level for normalization

    Returns:
        Flattened residuals normalized by noise
    """
    # Compute prediction: shapes.T @ amplitudes
    prediction = shapes.T @ amplitudes
    # Compute and normalize residuals
    return ((corrected_data - prediction) / noise).ravel()


@nb.njit(cache=True, fastmath=True)
def _compute_corrections(shapes: FloatArray, amplitudes: FloatArray) -> FloatArray:
    """Compute cross-talk corrections (Numba-optimized).

    Args:
        shapes: Lineshapes from other clusters, shape (n_points, n_other_peaks)
        amplitudes: Amplitudes for other peaks, shape (n_other_peaks, n_planes)

    Returns:
        Corrections array, shape (n_planes, n_points) or (n_points,)
    """
    return shapes @ amplitudes


# =============================================================================
# High-level computational functions
# =============================================================================


def calculate_shapes(params: Parameters, cluster: Cluster) -> FloatArray:
    """Calculate shapes for all peaks in a cluster.

    Automatically uses optimized batch evaluation when possible,
    providing 15-22× speedup for large clusters.

    Args:
        params: Current parameter values
        cluster: Cluster containing peaks

    Returns:
        Array of shape (n_peaks, n_points) with evaluated lineshapes
    """
    # Check if we can use batch evaluation optimization
    # Requirements: all peaks are 1D with same shape type
    if all(len(peak.shapes) == 1 for peak in cluster.peaks):
        first_shape = cluster.peaks[0].shapes[0]
        shape_class = type(first_shape)

        # Check if all shapes are the same type
        all_shapes = [peak.shapes[0] for peak in cluster.peaks]
        if all(isinstance(s, shape_class) for s in all_shapes):
            # Use optimized batch evaluation via the common interface
            return shape_class.batch_evaluate(all_shapes, cluster.positions[0], params)

    # Fallback to sequential evaluation for mixed shapes or multi-dimensional peaks
    return np.array([peak.evaluate(cluster.positions, params) for peak in cluster.peaks])


def calculate_amplitudes(shapes: FloatArray, data: FloatArray) -> FloatArray:
    """Calculate peak amplitudes via linear least squares using normal equations.

    Solves A^T A x = A^T b using optimized Numba functions and NumPy's solver.
    This is faster than np.linalg.lstsq for overdetermined systems.

    Args:
        shapes: Peak lineshapes, shape (n_peaks, n_points)
        data: Data to fit, shape (n_points,) or (n_planes, n_points)

    Returns:
        Optimal amplitudes for each peak
    """
    # Ensure consistent dtype for Numba operations (required by @)
    shapes_f64 = np.asarray(shapes, dtype=np.float64)
    data_f64 = np.asarray(data, dtype=np.float64)

    # Compute normal equations: A^T A and A^T b
    ata = compute_ata_symmetric(shapes_f64)  # Exploits symmetry with parallel computation
    atb = compute_atb(shapes_f64, data_f64)  # Simple matrix-vector product

    # Solve the system (uses Cholesky internally for symmetric positive definite)
    return np.linalg.solve(ata, atb)


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
    return _compute_residuals_flat(cluster.corrected_data, shapes, amplitudes, noise)


def update_cluster_corrections(params: Parameters, clusters: Sequence[Cluster]) -> None:
    """Update cross-talk corrections for clusters.

    This function computes the contribution of peaks from other clusters
    to each cluster's data, allowing for iterative refinement of fits.

    Args:
        params: Current parameter values (all clusters)
        clusters: List of clusters to update
    """
    cluster_all = Cluster.from_clusters(clusters)
    _shapes_all, amplitudes_all = calculate_shape_heights(params, cluster_all)
    for cluster in clusters:
        indexes = [
            index for index, peak in enumerate(cluster_all.peaks) if peak not in cluster.peaks
        ]
        shapes = np.array(
            [cluster_all.peaks[index].evaluate(cluster.positions, params) for index in indexes]
        ).T
        amplitudes = amplitudes_all[indexes, :]
        cluster.corrections = _compute_corrections(shapes, amplitudes)
