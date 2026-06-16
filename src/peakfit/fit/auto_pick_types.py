"""Auto-pick result and UI callback contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np

    from peakfit.engine.domain.peaks import Peak


@dataclass(frozen=True)
class AutoPickDiagnostics:
    """Diagnostic counters for automatic peak picking."""

    iterations: int
    accepted_rois: int
    rejected_rois: int
    accepted_peaks: int
    stopped_by_user: bool


@dataclass(frozen=True)
class AutoPickResult:
    """Result container for automatic peak picking."""

    peaks: list[Peak]
    diagnostics: AutoPickDiagnostics


@dataclass(frozen=True)
class FTestDecision:
    """Detailed F-test acceptance decision for one trial."""

    accepted: bool
    reason: str
    old_rss: float
    new_rss: float
    df1: int
    df2: int
    f_stat: float | None
    p_value: float | None


@dataclass(frozen=True)
class AutoPickTrialReport:
    """Per-trial report inside one ROI cycle."""

    trial_index: int
    candidate_point: tuple[int, ...]
    candidate_ppm: tuple[float, ...]
    candidate_score: float
    fit_success: bool
    accepted: bool
    reason: str
    f_test: FTestDecision | None
    fit_step_rounds: int
    cs_at_constraint: bool
    zero_amplitude_peak: bool


@dataclass(frozen=True)
class AutoPickCycleReport:
    """Cycle-level report used for terminal logging and step-wise control."""

    iteration: int
    seed_point: tuple[int, ...]
    seed_ppm: tuple[float, ...]
    seed_height: float
    roi_size: int
    add_threshold: float
    accepted: bool
    peaks_added: int
    total_peaks: int
    working_max_after: float
    trials: list[AutoPickTrialReport]
    contour_level: float
    experimental_projection: np.ndarray
    simulated_projection: np.ndarray
    current_peaks: list[Peak]
    roi_peaks: list[Peak]
    roi_x_limits: tuple[float, float] | None
    roi_y_limits: tuple[float, float] | None
    next_candidate_ppm: tuple[float, float] | None
    next_candidate_name: str | None
    feedback_message: str | None = None
    stage: str = "cycle_complete"


@dataclass(frozen=True)
class AutoPickCycleAction:
    """User action for interactive auto-pick stepping."""

    command: Literal[
        "continue",
        "remove_last_peak",
        "release_linewidths",
        "next_cluster",
        "previous_cluster",
        "stop",
    ] = "continue"
    candidate_ppm: tuple[float, float] | None = None
    candidate_ppm_list: list[tuple[float, float]] | None = None
    allow_suggested_fallback: bool = True


AutoPickCycleCallback = Callable[[AutoPickCycleReport], AutoPickCycleAction]


__all__ = [
    "AutoPickCycleAction",
    "AutoPickCycleCallback",
    "AutoPickCycleReport",
    "AutoPickDiagnostics",
    "AutoPickResult",
    "AutoPickTrialReport",
    "FTestDecision",
]
