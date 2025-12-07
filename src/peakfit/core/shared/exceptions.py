"""Exception taxonomy for PeakFit.

This module defines a small, coherent hierarchy of exceptions to improve
error handling across the codebase. Use these instead of generic Exception
to communicate intent and allow callers to handle errors precisely.
"""

from __future__ import annotations


class PeakFitError(Exception):
    """Base class for all PeakFit-specific exceptions."""


class ConfigError(PeakFitError):
    """Configuration-related errors (invalid/missing options, schema issues)."""


class DataIOError(PeakFitError):
    """Data loading/saving errors (files, formats, permissions)."""


class OptimizationError(PeakFitError):
    """Errors occurring during optimization or strategy execution."""


class NumericsError(PeakFitError):
    """Numeric instability or invalid arithmetic conditions (NaNs, overflows)."""


__all__ = [
    "PeakFitError",
    "ConfigError",
    "DataIOError",
    "OptimizationError",
    "NumericsError",
]
