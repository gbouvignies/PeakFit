"""PeakFit-specific exceptions used by public workflows."""


class PeakFitError(Exception):
    """Base class for all PeakFit-specific exceptions."""


class DataIOError(PeakFitError):
    """Data loading/saving errors (files, formats, permissions)."""


__all__ = [
    "DataIOError",
    "PeakFitError",
]
