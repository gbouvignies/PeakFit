"""PeakFit-specific exceptions used by public workflows."""

import numpy as np


class PeakFitError(Exception):
    """Base class for all PeakFit-specific exceptions."""


class ScientificIntegrityError(PeakFitError, ValueError):
    """Recorded or reconstructed scientific evidence disagrees with itself."""


def require_close(actual, expected, *, rtol: float, atol: float, message: str) -> None:
    """Raise unless values agree under numpy's ``assert_allclose`` rule.

    Shapes must match unless one side is a scalar; NaNs compare equal.
    """
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (actual.shape and expected.shape and actual.shape != expected.shape) or not np.all(
        np.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    ):
        raise ScientificIntegrityError(message)


__all__ = [
    "PeakFitError",
    "ScientificIntegrityError",
    "require_close",
]
