"""Internal state containers for auto-pick fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from peakfit.auto_pick.types import AutoPickTrialReport
    from peakfit.engine.domain.params_scalar import Parameters
    from peakfit.engine.domain.peaks import Peak
    from peakfit.shared.typing import FloatArray


@dataclass(frozen=True)
class TrialState:
    """Intermediate fit state used for ROI peak-growth decisions."""

    peaks: list[Peak]
    data: FloatArray
    model: FloatArray
    residual: FloatArray
    footprint: np.ndarray
    n_params: int
    dof_scale: float
    params: Parameters


@dataclass(frozen=True)
class RoiFitResult:
    """Final accepted state and trial trace for a ROI."""

    accepted_state: TrialState | None
    trials: list[AutoPickTrialReport]
    add_threshold: float
    stopped_by_user: bool = False
    previous_cluster_requested: bool = False


@dataclass(frozen=True)
class TrialFitOutcome:
    """Detailed fit outcome for one trial peak candidate."""

    state: TrialState
    fit_step_rounds: int
    cs_at_constraint: bool
    zero_amplitude_peak: bool


@dataclass(frozen=True)
class AutoPickSnapshot:
    """Checkpoint of global auto-pick state before processing one ROI."""

    working_data: np.ndarray
    calculated_data: np.ndarray
    processed_mask: np.ndarray
    accepted_peaks: list[Peak]
    accepted_rois: int
    rejected_rois: int
    iterations: int
    next_peak_number: int


__all__ = [
    "AutoPickSnapshot",
    "RoiFitResult",
    "TrialFitOutcome",
    "TrialState",
]
