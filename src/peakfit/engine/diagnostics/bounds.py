"""The one review rule for a fitted parameter that sits at a bound.

A varied parameter is "At bound" when its distance from the nearest bound is at
most ``AT_BOUND_FRACTION`` of its bound interval. The tolerance is relative to the
interval, so it is the same fraction of the allowed range at every parameter scale.

- A fixed parameter is never flagged: its value was not fitted.
- The rule needs two finite bounds. A parameter with an infinite or one-sided
  interval has no interval to scale the tolerance and is never flagged.
- A non-finite value is never flagged; a value outside its bounds is.
- A zero-width interval (``min == max``) is flagged: the value can only be at it.

This is a warning for review. Whether symmetric uncertainties are withheld at a
constrained optimum is a separate scientific decision made by VarPro with its own
absolute tolerance (``engine.algorithms.varpro._calculate_errors``).
"""

import math
from typing import Protocol

AT_BOUND_FRACTION = 1e-6


class BoundedParameter(Protocol):
    """The fields the rule reads, shared by working and recorded parameters."""

    @property
    def value(self) -> float: ...
    @property
    def min(self) -> float: ...
    @property
    def max(self) -> float: ...
    @property
    def vary(self) -> bool: ...


def at_bound(parameter: BoundedParameter) -> bool:
    """Whether a fitted parameter is at a bound under the review rule above."""
    value, minimum, maximum = parameter.value, parameter.min, parameter.max
    if not parameter.vary or not all(map(math.isfinite, (value, minimum, maximum))):
        return False
    span = maximum - minimum
    if not math.isfinite(span) or span < 0:
        return False
    return min(value - minimum, maximum - value) <= AT_BOUND_FRACTION * span


__all__ = ["AT_BOUND_FRACTION", "at_bound"]
