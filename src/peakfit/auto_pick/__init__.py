"""Experimental automatic peak-picking workflow."""

from peakfit.auto_pick.algorithm import auto_pick_peaks
from peakfit.auto_pick.types import (
    AutoPickCycleAction,
    AutoPickCycleCallback,
    AutoPickCycleReport,
    AutoPickDiagnostics,
    AutoPickResult,
    AutoPickTrialReport,
    FTestDecision,
)

__all__ = [
    "AutoPickCycleAction",
    "AutoPickCycleCallback",
    "AutoPickCycleReport",
    "AutoPickDiagnostics",
    "AutoPickResult",
    "AutoPickTrialReport",
    "FTestDecision",
    "auto_pick_peaks",
]
