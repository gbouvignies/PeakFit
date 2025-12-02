"""Domain representation of serialized fitting state artifacts."""

from __future__ import annotations


from pydantic import BaseModel, ConfigDict, Field

from peakfit.core.domain.cluster import Cluster
from peakfit.core.domain.peaks import Peak
from peakfit.core.fitting.parameters import Parameters  # noqa: TC001


class FittingState(BaseModel):
    """In-memory representation of a saved fitting run."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    clusters: list[Cluster] = Field(description="List of Cluster objects")
    params: Parameters
    noise: float
    peaks: list[Peak] = Field(description="List of Peak objects")
    version: str = Field(default="1.0")

    def to_payload(self) -> dict[str, object]:
        """Convert the state into a pickle-friendly payload.

        Deprecated: Use model_dump() instead.
        """
        return self.model_dump()

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> FittingState:
        """Construct a state object from a serialized payload.

        Deprecated: Use model_validate() instead.
        """
        return cls.model_validate(payload)
