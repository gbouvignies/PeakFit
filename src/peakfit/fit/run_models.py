"""Runtime containers for fit orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from peakfit.engine.scientific.summary import RunSummary

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.engine.algorithms.evaluation import FitOutcomeClassification
    from peakfit.engine.domain.cluster import Cluster
    from peakfit.engine.domain.peaks import Peak
    from peakfit.engine.domain.spectrum import Spectra
    from peakfit.engine.scientific.completed import CompletedFit
    from peakfit.engine.scientific.final_outcome import FinalFitOutcome


@dataclass(frozen=True)
class LoadedData:
    """Container for loaded fitting data."""

    spectra: Spectra
    peaks: list[Peak]
    noise: float
    noise_source: str
    shape_names: list[str]
    contour_level: float
    clusters: list[Cluster]


@dataclass(frozen=True)
class FitRun:
    """A completed CLI fitting run: the completed fit, its spectra and output directory."""

    completed_fit: CompletedFit
    output_dir: Path
    spectra: Spectra

    @property
    def outcome(self) -> FinalFitOutcome:
        """The single authority for completed values and optimizer diagnostics."""
        return self.completed_fit.outcome

    @property
    def summary(self) -> RunSummary:
        """Return the current summary projection of the immutable outcome."""
        return RunSummary.from_outcome(self.outcome)


@dataclass(frozen=True)
class ProgressStart:
    """Event emitted at pipeline start with total steps."""

    total_steps: int
    n_clusters: int
    n_workers: int


@dataclass(frozen=True)
class PassStart:
    """Event emitted when an optimizer pass of a multi-iteration fit step starts."""

    iteration: int  # 1-based
    iterations: int


@dataclass(frozen=True)
class CorrectionUpdate:
    """Event emitted when neighbor corrections are updated between optimizer passes."""


@dataclass(frozen=True)
class ClusterReview:
    """Data for a cluster that needs review."""

    cluster_id: str
    peak_names: tuple[str, ...]
    classification: FitOutcomeClassification
    reason: str
    redchi: float | None
    at_bounds: list[str]
    unusable_reason: str | None
    termination_message: str | None


__all__ = [
    "ClusterReview",
    "CorrectionUpdate",
    "FitRun",
    "LoadedData",
    "PassStart",
    "ProgressStart",
]
