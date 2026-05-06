"""Shared domain data structures."""

from dataclasses import dataclass


@dataclass
class PeakData:
    """Peak metadata with N-dimensional position support."""

    name: str
    positions: list[float]
    cluster_id: int | None = None

    def get_positions(self) -> list[float]:
        """Return peak positions ordered from F1 (indirect) to Fn (direct)."""
        return self.positions

    @property
    def n_dims(self) -> int:
        """Number of dimensions represented by the peak."""
        return len(self.positions)


__all__ = ["PeakData"]
