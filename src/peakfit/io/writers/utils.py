"""Utility functions for output writers."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from peakfit.engine.results import ParameterEstimate

type JsonValue = dict[str, "JsonValue"] | list["JsonValue"] | str | int | float | bool | None


def format_float(
    value: float | None,
    precision: int = 6,
    scientific_threshold: int = 4,
) -> str:
    """Format a float with appropriate notation.

    Uses scientific notation for very large or small values,
    fixed-point otherwise.

    Args:
        value: Value to format (can be None)
        precision: Number of decimal places
        scientific_threshold: Use scientific notation if |log10(value)| > threshold

    Returns:
    -------
        Formatted string or empty string if None
    """
    if value is None:
        return ""

    if value == 0:
        return f"{0:.{precision}f}"

    if math.isinf(value) or math.isnan(value):
        return str(value)

    log_val = math.log10(abs(value))
    if abs(log_val) > scientific_threshold:
        return f"{value:.{precision}e}"
    return f"{value:.{precision}f}"


def get_peak_name(param: ParameterEstimate, peak_names: list[str]) -> str:
    """Extract the original peak name from a parameter.

    Args:
        param: ParameterEstimate object
        peak_names: List of peak names in the cluster

    Returns:
    -------
        Original peak name like '2N-H'
    """
    if param.param_id is not None:
        if param.param_id.peak_name:
            return param.param_id.peak_name
        if param.param_id.cluster_id is not None:
            return f"cluster_{param.param_id.cluster_id}"

    param_name = param.name
    cluster_match = re.match(r"cluster_(\d+)\.", param_name)
    if cluster_match:
        return f"cluster_{cluster_match.group(1)}"

    for peak_name in peak_names:
        if param_name.startswith(f"{peak_name}."):
            return peak_name

    return peak_names[0] if peak_names else ""


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types and Path objects."""

    def default(self, o: Any) -> JsonValue:
        """Convert numpy types and Path objects to Python types."""
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


__all__ = [
    "JsonValue",
    "NumpyEncoder",
    "format_float",
    "get_peak_name",
]
