from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation, generate_binary_structure, label

from peakfit.core.shared.typing import FloatArray, IntArray

if TYPE_CHECKING:
    from peakfit.core.domain.peaks import Peak
    from peakfit.core.domain.spectrum import Spectra

from peakfit.core.domain.cluster import Cluster


class PeakLike(Protocol):
    """Minimal interface required for segmentation."""

    @property
    def positions_i(self) -> IntArray | tuple[int, ...]: ...


def group_connected_pairs(pairs: Iterable[tuple[int, int]]) -> list[list[int]]:
    """Group connected pairs using a graph-based approach.

    Args:
        pairs (Iterable[tuple[int, int]]): Iterable of pairs of connected indices.

    Returns:
        list[list[int]]: List of grouped and sorted connected components.
    """
    graph = nx.Graph()
    graph.add_edges_from(pairs)
    return [sorted(component) for component in nx.connected_components(graph)]


def merge_connected_segments(segments: IntArray) -> IntArray:
    """Merge connected segments in a labeled array.

    Args:
        segments (IntArray): Array with labeled segments.

    Returns:
        IntArray: Array with merged segments.
    """
    for _ in range(segments.ndim):
        merge_mask = np.logical_and(segments[0] > 0, segments[-1] > 0)
        segs = np.asarray(segments)
        seg0 = np.take(segs, 0, axis=0)
        segn = np.take(segs, -1, axis=0)
        a = np.atleast_1d(seg0[merge_mask])
        b = np.atleast_1d(segn[merge_mask])
        connected_pairs = zip(a, b, strict=True)  # type: ignore[reportGeneralTypeIssues]
        connected_groups = group_connected_pairs(connected_pairs)

        for group in connected_groups:
            primary_segment_label = group[0]
            for segment_number in group[1:]:
                segments[segments == segment_number] = primary_segment_label

        segments = np.moveaxis(segments, 0, -1)

    return segments


def segment_data(
    data: FloatArray,
    contour_level: float,
    peaks: Sequence[PeakLike],
) -> IntArray:
    """Segment the spectral data based on the contour level.

    Args:
        data (np.ndarray): The spectral data.
        contour_level (float): Contour level for segmenting the data.
        peaks (list[Peak]): List of detected peaks.

    Returns:
        IntArray: Labeled segments array.
    """
    data_above_threshold = np.any(np.abs(data) >= contour_level, axis=0)
    data_around_peaks = np.zeros_like(data_above_threshold, dtype=bool)

    for peak in peaks:
        position = tuple(int(idx) for idx in np.asarray(peak.positions_i))
        data_around_peaks[position] = True

    connectivity: Any = data.ndim - 1
    structuring_element: NDArray[np.int_] = np.asarray(
        generate_binary_structure(data.ndim - 1, connectivity), dtype=np.int_
    )
    data_around_peaks = binary_dilation(data_around_peaks, structuring_element)
    data_selected = np.logical_or(data_above_threshold, data_around_peaks)
    labeled_segments, _ = label(  # type: ignore[reportGeneralTypeIssues]
        cast(NDArray[np.bool_], np.asarray(data_selected, dtype=np.bool_)),
        structure=structuring_element,
    )
    segments = np.asarray(labeled_segments, dtype=np.int_)

    return merge_connected_segments(segments)


def assign_peaks_to_segments(peaks: list[Peak], segments: IntArray) -> dict[int, list[Peak]]:
    """Assign peaks to their respective segments.

    Args:
        peaks (list[Peak]): List of detected peaks.
        segments (IntArray): Array with labeled segments.

    Returns:
        dict[int, list[Peak]]: Dictionary mapping segment IDs to peaks.
    """
    peak_segments_dict: dict[int, list[Peak]] = {}
    for peak in peaks:
        position_indices = tuple(int(idx) for idx in np.asarray(peak.positions_i))
        segment_id = int(segments[position_indices])
        peak_segments_dict.setdefault(segment_id, []).append(peak)
    return peak_segments_dict


def create_clusters(spectra: Spectra, peaks: list[Peak], contour_level: float) -> list[Cluster]:
    """Create clusters from spectral data based on peaks and contour levels.

    Args:
        spectra (Spectra): Spectra object containing the data.
        peaks (list[Peak]): List of detected peaks.
        contour_level (float): Contour level for segmenting the data.

    Returns:
        list[Cluster]: List of created clusters.
    """
    segments = segment_data(spectra.data, contour_level, peaks)
    peak_segments_dict = assign_peaks_to_segments(peaks, segments)

    clusters: list[Cluster] = []
    for segment_id, peaks_in_segment in peak_segments_dict.items():
        for peak in peaks_in_segment:
            peak.set_cluster_id(segment_id)
        segment_positions = [*np.where(segments == segment_id)]
        segmented_data = spectra.data[:, *segment_positions].T
        clusters.append(Cluster(segment_id, peaks_in_segment, segment_positions, segmented_data))

    return sorted(clusters, key=lambda cluster: len(cluster.peaks))
