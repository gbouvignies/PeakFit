"""Runtime containers for fit orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.algorithms.evaluation import FitOutcomeClassification

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.engine.domain.cluster import Cluster
    from peakfit.engine.domain.peaks import Peak
    from peakfit.engine.domain.spectrum import Spectra
    from peakfit.engine.domain.state import FittingState
    from peakfit.fit.final_outcome import FinalFitOutcome
    from peakfit.fit.simulation import FinalModelSnapshot


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
class RunSummary:
    """Summary statistics for a fitting run."""

    n_clusters: int
    n_peaks: int
    success_rate: float
    n_converged: int
    n_usable_non_converged: int
    n_unusable: int
    redchi_population_size: int
    mean_redchi: float | None
    std_redchi: float | None
    median_redchi: float | None

    @property
    def n_usable(self) -> int:
        """Return the number of outcomes that supply scientific quantities."""
        return self.n_converged + self.n_usable_non_converged

    @classmethod
    def from_outcome(cls, outcome: FinalFitOutcome) -> RunSummary:
        """Project completed-run counts and distributions from final outcomes only."""
        outcomes = outcome.clusters
        n_converged = sum(
            cluster.classification is FitOutcomeClassification.CONVERGED for cluster in outcomes
        )
        n_usable_non_converged = sum(
            cluster.classification is FitOutcomeClassification.USABLE_NON_CONVERGED
            for cluster in outcomes
        )
        n_unusable = sum(
            cluster.classification is FitOutcomeClassification.UNUSABLE for cluster in outcomes
        )
        redchis = [
            cluster.analytical_evaluation.statistics.reduced_chi_squared
            for cluster in outcomes
            if cluster.usable and cluster.analytical_evaluation is not None
        ]
        if len(redchis) != n_converged + n_usable_non_converged:
            raise ValueError("Usable final outcomes must contain an analytical evaluation.")
        n_clusters = len(outcomes)
        return cls(
            n_clusters=n_clusters,
            n_peaks=sum(len(cluster.peak_names) for cluster in outcomes),
            success_rate=n_converged / n_clusters if n_clusters else 0.0,
            n_converged=n_converged,
            n_usable_non_converged=n_usable_non_converged,
            n_unusable=n_unusable,
            redchi_population_size=len(redchis),
            mean_redchi=float(np.mean(redchis)) if redchis else None,
            std_redchi=float(np.std(redchis)) if redchis else None,
            median_redchi=float(np.median(redchis)) if redchis else None,
        )


@dataclass(frozen=True)
class FitRun:
    """Result of a complete fitting run."""

    outcome: FinalFitOutcome
    continuation_state: FittingState
    output_dir: Path
    spectra: Spectra | None = None
    simulation_snapshot: FinalModelSnapshot | None = None

    @property
    def success(self) -> bool:
        """Return overall convergence derived from the immutable outcome."""
        return self.outcome.overall_converged

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
    "FitRun",
    "LoadedData",
    "ProgressStart",
    "RunSummary",
]
