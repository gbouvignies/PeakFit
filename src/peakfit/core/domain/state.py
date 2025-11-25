"""Domain representation of serialized fitting state artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from peakfit.core.domain.cluster import Cluster
from peakfit.core.domain.peaks import Peak
from peakfit.core.fitting.parameters import Parameters


@dataclass(slots=True)
class FittingState:
    """In-memory representation of a saved fitting run."""

    clusters: list[Cluster]
    params: Parameters
    noise: float
    peaks: list[Peak]
    version: str = "1.0"

    def to_payload(self) -> dict[str, Any]:
        """Convert the state into a pickle-friendly payload."""
        return {
            "clusters": self.clusters,
            "params": self.params,
            "noise": float(self.noise),
            "peaks": self.peaks,
            "version": self.version,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> FittingState:
        """Construct a state object from a serialized payload."""
        required_keys = {"clusters", "params", "noise", "peaks"}
        missing = required_keys - payload.keys()
        if missing:
            missing_list = ", ".join(sorted(missing))
            msg = f"Missing state fields: {missing_list}"
            raise ValueError(msg)

        version = payload.get("version", "1.0")
        return cls(
            clusters=payload["clusters"],
            params=payload["params"],
            noise=float(payload["noise"]),
            peaks=payload["peaks"],
            version=version,
        )
