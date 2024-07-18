from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Self

import networkx as nx
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure, label

from peakfit.messages import print_segmenting
from peakfit.peak import Peak
from peakfit.spectra import Spectra
from peakfit.typing import FloatArray, IntArray


@dataclass
class Cluster:
    cluster_id: int
    peaks: list[Peak]
    positions: list[IntArray]
    data: FloatArray
    corrections: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.corrections = np.zeros_like(self.data)

    @classmethod
    def from_clusters(cls, clusters: Sequence[Self]) -> Self:
        return sum(clusters[1:], clusters[0])

    @property
    def corrected_data(self) -> FloatArray:
        return self.data - self.corrections

    def __add__(self, other: object) -> Self:
        if not isinstance(other, Cluster):
            return NotImplemented
        return type(self)(
            self.cluster_id,
            self.peaks + other.peaks,
            [
                np.concatenate((positions_self, positions_other))
                for positions_self, positions_other in zip(
                    self.positions, other.positions, strict=False
                )
            ],
            np.concatenate((self.data, other.data), axis=0),
        )


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
        connected_pairs = zip(
            segments[0][merge_mask], segments[-1][merge_mask], strict=True
        )
        connected_groups = group_connected_pairs(connected_pairs)

        for group in connected_groups:
            primary_segment_label = group[0]
            for segment_number in group[1:]:
                segments[segments == segment_number] = primary_segment_label

        segments = np.moveaxis(segments, 0, -1)

    return segments


def segment_data(data: FloatArray, contour_level: float, peaks: list[Peak]) -> IntArray:
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
        data_around_peaks[tuple(peak.positions_i)] = True

    structuring_element = generate_binary_structure(data.ndim - 1, data.ndim - 1)
    data_around_peaks = binary_dilation(data_around_peaks, structuring_element)
    data_selected = np.logical_or(data_above_threshold, data_around_peaks)
    segments, _ = label(data_selected, structure=structuring_element)

    return merge_connected_segments(segments)


def assign_peaks_to_segments(peaks: list[Peak], segments: IntArray) -> dict:
    """Assign peaks to their respective segments.

    Args:
        peaks (list[Peak]): List of detected peaks.
        segments (IntArray): Array with labeled segments.

    Returns:
        dict: Dictionary mapping segment IDs to peaks.
    """
    peak_segments_dict = {}
    for peak in peaks:
        segment_id = segments[*peak.positions_i]
        peak_segments_dict.setdefault(segment_id, []).append(peak)
    return peak_segments_dict


def create_clusters(
    spectra: Spectra, peaks: list[Peak], contour_level: float
) -> list[Cluster]:
    """Create clusters from spectral data based on peaks and contour levels.

    Args:
        spectra (Spectra): Spectra object containing the data.
        peaks (list[Peak]): List of detected peaks.
        contour_level (float): Contour level for segmenting the data.

    Returns:
        list[Cluster]: List of created clusters.
    """
    print_segmenting()
    segments = segment_data(spectra.data, contour_level, peaks)
    peak_segments_dict = assign_peaks_to_segments(peaks, segments)

    clusters: list[Cluster] = []
    for segment_id, peaks_in_segment in peak_segments_dict.items():
        for peak in peaks_in_segment:
            peak.set_cluster_id(segment_id)
        segment_positions = [*np.where(segments == segment_id)]
        segmented_data = spectra.data[:, *segment_positions].T
        clusters.append(
            Cluster(segment_id, peaks_in_segment, segment_positions, segmented_data)
        )

    return sorted(clusters, key=lambda cluster: len(cluster.peaks))
