"""Domain representation of serialized fitting state artifacts."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.params_vector import FitParameters


class FittingState(BaseModel):
    """In-memory representation of a saved fitting run."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    clusters: list[Cluster] = Field(description="List of Cluster objects")
    params: FitParameters = Field(description="Vectorized fitting parameters")
    scalar_params: Parameters = Field(
        default_factory=Parameters, description="Rich parameter metadata (legacy)"
    )
    noise: float | None = Field(default=None, description="Estimated noise level")
    version: str = Field(default="1.1", description="State format version")

    @model_validator(mode="before")
    @classmethod
    def strip_legacy_peaks(cls, data: Any) -> Any:
        """Drop legacy top-level peaks from serialized state inputs."""
        if isinstance(data, dict):
            data.pop("peaks", None)
        return data

    @property
    def peaks(self):
        """Flattened list of peaks derived from clusters."""
        return [peak for cluster in self.clusters for peak in cluster.peaks]

    @model_validator(mode="after")
    def validate_consistency(self) -> FittingState:
        """Ensure consistency between peaks and parameters."""
        n_peaks = len(self.peaks)
        n_params = len(self.params.positions)

        # FitParameters flattens multi-dimensional peaks (one entry per dimension/shape)
        # So we expect n_params to equal total number of shapes, not just peaks
        # However, verifying exact match requires iterating peaks which is fine
        expected_params = sum(len(p.shapes) for p in self.peaks)

        if n_params != expected_params:
            msg = (
                f"Number of parameters ({n_params}) does not match "
                f"total peak shapes ({expected_params}) from {n_peaks} peaks"
            )
            raise ValueError(msg)

        return self


# Resolve forward references
FittingState.model_rebuild()
