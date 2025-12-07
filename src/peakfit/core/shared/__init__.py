"""Shared foundational utilities for PeakFit."""

from peakfit.core.shared import constants, events, reporter, typing
from peakfit.core.shared.exceptions import (
    ConfigError,
    DataIOError,
    NumericsError,
    PeakFitError,
    OptimizationError,
)
from peakfit.core.shared.events import Event, EventDispatcher, EventType, FitProgressEvent
from peakfit.core.shared.reporter import CompositeReporter, LoggingReporter, NullReporter, Reporter

__all__ = [
    "CompositeReporter",
    "ConfigError",
    "DataIOError",
    "Event",
    "EventDispatcher",
    "EventType",
    "FitProgressEvent",
    "LoggingReporter",
    "NumericsError",
    "PeakFitError",
    "OptimizationError",
    "NullReporter",
    "Reporter",
    "constants",
    "events",
    "reporter",
    "typing",
]
