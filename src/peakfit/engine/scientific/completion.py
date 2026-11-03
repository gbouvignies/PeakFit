"""Terminal pipeline records that finalization turns into a completed fit."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from peakfit.engine.algorithms.evaluation import FitEvaluation
    from peakfit.engine.domain.cluster import Cluster
    from peakfit.engine.domain.state import FittingState
    from peakfit.engine.results import FitResult
    from peakfit.shared.typing import FloatArray


@dataclass(frozen=True, slots=True)
class CorrectionSnapshot:
    """Read-only correction arrays used by a single optimizer pass."""

    revision: int
    corrections: Mapping[int, FloatArray]

    @classmethod
    def from_clusters(
        cls,
        clusters: Sequence[Cluster],
        revision: int,
    ) -> CorrectionSnapshot:
        """Copy corrections so later scheduling cannot alter this pass."""
        copied_corrections: dict[int, FloatArray] = {}
        for cluster in clusters:
            correction = cluster.corrections.copy()
            correction.flags.writeable = False
            copied_corrections[cluster.cluster_id] = correction
        return cls(revision=revision, corrections=MappingProxyType(copied_corrections))


@dataclass(frozen=True)
class PipelineCompletion:
    """Internal terminal attempts and state consumed once by finalization."""

    state: FittingState
    results: list[FitResult]
    evaluations: list[FitEvaluation] = field(default_factory=list)
    correction_snapshot: CorrectionSnapshot | None = None
    n_optimizer_passes: int = 0
    n_correction_updates: int = 0


__all__ = ["CorrectionSnapshot", "PipelineCompletion"]
