"""Domain model representing a cluster of peaks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from peakfit.core.shared.typing import FloatArray, IntArray

if TYPE_CHECKING:
    from peakfit.core.domain.peaks import Peak


@dataclass
class Cluster:
    """Grouped peaks sharing a contiguous spectral segment."""

    cluster_id: int
    peaks: list[Peak]
    positions: list[IntArray]
    data: FloatArray
    corrections: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize correction array for this cluster (zeros of same shape)."""
        self.corrections = np.zeros_like(self.data)

    @classmethod
    def from_clusters(cls, clusters: list[Cluster] | Sequence[Cluster]) -> Cluster:
        """Create a single cluster by merging a list of clusters.

        Returns a combined cluster produced by summing all clusters.
        """
        if not clusters:
            msg = "clusters list cannot be empty"
            raise ValueError(msg)
        return sum(clusters[1:], clusters[0])

    @property
    def corrected_data(self) -> FloatArray:
        """Return data with corrections subtracted (ready for processing)."""
        return self.data - self.corrections

    def __add__(self, other: object) -> Cluster:
        """Concatenate two clusters, preserving peaks and data arrays."""
        if not isinstance(other, Cluster):
            return NotImplemented
        return type(self)(
            self.cluster_id,
            self.peaks + other.peaks,
            [
                np.concatenate((positions_self, positions_other))
                for positions_self, positions_other in zip(
                    self.positions,
                    other.positions,
                    strict=False,
                )
            ],
            np.concatenate((self.data, other.data), axis=0),
        )
