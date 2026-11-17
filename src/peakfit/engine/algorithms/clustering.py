import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure, label
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components as connected_comps

from peakfit.engine.algorithms.noise import prepare_noise_level
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.constraints import apply_constraints
from peakfit.engine.domain.params_scalar import Parameters

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from peakfit.engine.domain.config import PeakFitConfig
    from peakfit.engine.domain.peaks import Peak
    from peakfit.engine.domain.spectrum import Spectra
    from peakfit.shared.typing import FloatArray, IntArray


def group_connected_pairs(pairs: Iterable[tuple[int, int]]) -> list[list[int]]:
    """Group connected pairs using a graph-based approach.

    Args:
        pairs: Iterable of pairs of connected indices.

    Returns:
    -------
        List of grouped and sorted connected components.
    """
    pairs_list = list(pairs)
    if not pairs_list:
        return []

    # Map sparse indices to dense range [0, N)
    unique_nodes = sorted({node for pair in pairs_list for node in pair})
    node_map = {node: i for i, node in enumerate(unique_nodes)}
    inverse_map = {i: node for node, i in node_map.items()}
    n_nodes = len(unique_nodes)

    # Build adjacency matrix
    row = [node_map[p[0]] for p in pairs_list]
    col = [node_map[p[1]] for p in pairs_list]
    data = np.ones(len(pairs_list), dtype=int)

    # connectivity matrix (symmetric)
    adj = coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
    adj = adj + adj.T

    # Find connected components
    _, labels = connected_comps(csr_matrix(adj), directed=False)

    # Group original node indices by component label
    groups: dict[int, list[int]] = {}
    for i, group_label in enumerate(labels):
        original_node = inverse_map[i]
        groups.setdefault(group_label, []).append(original_node)

    return [sorted(g) for g in groups.values()]


def merge_connected_segments(segments: IntArray) -> IntArray:
    """Merge connected segments in a labeled array (handling wrapping)."""
    # Iterate over each dimension to check for wrapping (first/last index connectivity)
    for axis in range(segments.ndim):
        segs_moved = np.moveaxis(segments, axis, 0)
        seg0 = segs_moved[0]
        segn = segs_moved[-1]

        # Check connectivity at boundaries
        merge_mask = (seg0 > 0) & (segn > 0)
        if not np.any(merge_mask):
            continue

        a = seg0[merge_mask]
        b = segn[merge_mask]

        connected_pairs = list(zip(a, b, strict=False))
        if not connected_pairs:
            continue

        connected_groups = group_connected_pairs(connected_pairs)

        for group in connected_groups:
            primary_label = group[0]
            for target_label in group[1:]:
                segments[segments == target_label] = primary_label

    return segments


def segment_data(
    data: FloatArray,
    contour_level: float,
    footprints: Sequence[Sequence[IntArray]],
) -> IntArray:
    """Segment the spectral data based on the contour level.

    Each footprint (per axis, the points a peak's components can reach) is marked,
    dilated by one point and joined to the points above the contour. A singlet's
    footprint is its centre point. A doublet's spans its components' whole range, so
    that its two components, each its own contour island, fall in one segment.
    """
    # 1. Threshold Mask
    data_above_threshold = np.any(np.abs(data) >= contour_level, axis=0)

    # 2. Peak Footprint Mask
    data_around_peaks = np.zeros_like(data_above_threshold, dtype=bool)
    for footprint in footprints:
        data_around_peaks[np.ix_(*footprint)] = True

    # 3. Dilate Peak Mask
    connectivity = data.ndim - 1
    structuring_element = generate_binary_structure(connectivity, connectivity)
    data_around_peaks = binary_dilation(data_around_peaks, structuring_element)

    # 4. Combine & Label
    data_selected = data_above_threshold | data_around_peaks
    labeled_segments, _ = label(data_selected, structure=structuring_element)

    # 5. Merge Wrapping Segments
    return merge_connected_segments(np.asarray(labeled_segments, dtype=int))


def assign_peaks_to_segments(peaks: list[Peak], segments: IntArray) -> dict[int, list[Peak]]:
    """Assign peaks to their respective segments."""
    peak_segments_dict: dict[int, list[Peak]] = {}
    for peak in peaks:
        pos = peak.positions_i
        pos_tup = tuple(int(x) for x in pos) if isinstance(pos, np.ndarray) else pos

        segment_id = int(segments[pos_tup])
        if segment_id > 0:
            peak_segments_dict.setdefault(segment_id, []).append(peak)

    return peak_segments_dict


def prepare_clusters(
    spectra: Spectra, peaks: list[Peak], config: PeakFitConfig
) -> tuple[float, float, list[Cluster]]:
    """Noise, clustering threshold and clusters: the fit preparation of the CLI and GUI.

    The threshold is the configured contour level, else ``contour_factor`` × noise,
    and must be positive and finite. Doublet footprints take J's upper bound after the
    configured parameter constraints, the bound the fit itself uses.
    """
    noise = prepare_noise_level(config.noise_level, spectra)
    threshold = config.clustering.contour_level
    if threshold is None:
        threshold = config.clustering.contour_factor * noise
    if not math.isfinite(threshold) or threshold <= 0:
        msg = (
            f"The clustering threshold must be positive and finite, got {threshold}; "
            "set a positive contour level."
        )
        raise ValueError(msg)
    bounds = apply_constraints(Parameters.from_peaks(peaks), config.parameters)
    return noise, float(threshold), create_clusters(spectra, peaks, threshold, bounds)


def create_clusters(
    spectra: Spectra,
    peaks: list[Peak],
    contour_level: float,
    bounds: Parameters | None = None,
) -> list[Cluster]:
    """Create clusters from spectral data based on peaks and contour levels.

    ``bounds`` are the peaks' parameters as the fit will bound them; a doublet's
    footprint spans cs ± J_max/2 with J_max its J's upper bound there. Without them,
    each axis model's own J bound applies.
    """
    if bounds is None:
        bounds = Parameters.from_peaks(peaks)
    footprints = [peak.footprint_i(bounds) for peak in peaks]
    segments = segment_data(spectra.data, contour_level, footprints)
    peak_segments_dict = assign_peaks_to_segments(peaks, segments)

    clusters: list[Cluster] = []

    for segment_id, peaks_in_segment in peak_segments_dict.items():
        if not peaks_in_segment:
            continue

        for peak in peaks_in_segment:
            peak.set_cluster_id(segment_id)

        # Extract cluster data
        grid_indices = np.where(segments == segment_id)
        segment_indices = list(grid_indices)

        # Extract data for this segment (all planes)
        indices = (slice(None), *grid_indices)
        segmented_data = spectra.data[indices].T.astype(float)

        clusters.append(
            Cluster(
                cluster_id=segment_id,
                peaks=peaks_in_segment,
                grid_indices=segment_indices,
                data=segmented_data,
            )
        )

    return sorted(clusters, key=lambda c: len(c.peaks))
