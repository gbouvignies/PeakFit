"""In-memory state a completed fit run leaves for continuation."""

from pydantic import BaseModel, ConfigDict, Field

from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.params_scalar import Parameters


class FittingState(BaseModel):
    """Terminal clusters, nonlinear parameters and noise of a completed fit run."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    clusters: list[Cluster] = Field(description="List of Cluster objects")
    scalar_params: Parameters = Field(
        default_factory=Parameters,
        description="Scalar parameter metadata used for output and reuse",
    )
    noise: float | None = Field(default=None, description="Estimated noise level")

    @property
    def peaks(self):
        """Flattened list of peaks derived from clusters."""
        return [peak for cluster in self.clusters for peak in cluster.peaks]


# Resolve forward references
FittingState.model_rebuild()
