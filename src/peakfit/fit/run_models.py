"""Runtime containers for fit orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.engine.domain.cluster import Cluster
    from peakfit.engine.domain.peaks import Peak
    from peakfit.engine.domain.spectrum import Spectra
    from peakfit.engine.domain.state import FittingState
    from peakfit.engine.results import FitResult


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
    mean_redchi: float
    std_redchi: float
    median_redchi: float


@dataclass(frozen=True)
class FitRun:
    """Result of a complete fitting run."""

    state: FittingState
    results: list[FitResult]
    output_dir: Path
    success: bool
    summary: RunSummary
    spectra: Spectra | None = None


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
    peak_names: list[str]
    reason: str
    redchi: float
    at_bounds: list[str]


__all__ = [
    "ClusterReview",
    "FitRun",
    "LoadedData",
    "ProgressStart",
    "RunSummary",
]
