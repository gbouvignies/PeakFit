"""Small formatting helpers shared by completed-result writers."""

from __future__ import annotations

import math


def format_float(
    value: float | None,
    precision: int = 6,
    scientific_threshold: int = 4,
) -> str:
    """Format an optional scalar for a durable human-readable artifact."""
    if value is None:
        return ""
    if value == 0:
        return f"{0:.{precision}f}"
    if math.isinf(value) or math.isnan(value):
        return str(value)
    if abs(math.log10(abs(value))) > scientific_threshold:
        return f"{value:.{precision}e}"
    return f"{value:.{precision}f}"


__all__ = ["format_float"]
