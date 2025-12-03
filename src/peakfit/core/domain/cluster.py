"""Domain model representing a cluster of peaks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

import numpy as np

from peakfit.core.domain.peaks import Peak

if TYPE_CHECKING:
    from collections.abc import Sequence

    from peakfit.core.shared.typing import FloatArray


class Cluster(BaseModel):
    """Grouped peaks sharing a contiguous spectral segment."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    cluster_id: int
    peaks: list[Peak]
    positions: list[Any] = Field(description="List of IntArray positions")
    data: Any = Field(description="Spectral data (FloatArray)")
    corrections: Any = Field(default=None, init=False, description="Correction array (FloatArray)")

    @model_validator(mode="after")
    def initialize_corrections(self) -> Cluster:
        """Initialize correction array for this cluster (zeros of same shape)."""
        if self.corrections is None:
            # Ensure data is numpy array
            if not isinstance(self.data, np.ndarray):
                self.data = np.array(self.data, dtype=float)
            self.corrections = np.zeros_like(self.data)
        return self

    @classmethod
    def from_clusters(cls, clusters: list[Cluster] | Sequence[Cluster]) -> Cluster:
        """Create a single cluster by merging a list of clusters.

        Returns a combined cluster produced by summing all clusters.
        """
        if not clusters:
            msg = "clusters list cannot be empty"
            raise ValueError(msg)
        # Start with the first cluster and add the rest
        result = clusters[0]
        for other in clusters[1:]:
            result = result + other
        return result

    @property
    def corrected_data(self) -> FloatArray:
        """Return data with corrections subtracted (ready for processing)."""
        return self.data - self.corrections

    def __add__(self, other: object) -> Cluster:
        """Concatenate two clusters, preserving peaks and data arrays."""
        if not isinstance(other, Cluster):
            return NotImplemented

        # Concatenate positions
        new_positions = [
            np.concatenate((positions_self, positions_other))
            for positions_self, positions_other in zip(
                self.positions,
                other.positions,
                strict=False,
            )
        ]

        # Concatenate data
        new_data = np.concatenate((self.data, other.data), axis=0)

        return type(self)(
            cluster_id=self.cluster_id,
            peaks=self.peaks + other.peaks,
            positions=new_positions,
            data=new_data,
        )
