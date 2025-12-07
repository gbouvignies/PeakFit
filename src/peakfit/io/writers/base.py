"""Base writer interface and configuration.

This module defines the OutputWriter protocol and common configuration
for all output writers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from peakfit.core.results.estimates import ParameterEstimate
    from peakfit.core.results.fit_results import FitResults


class Verbosity(str, Enum):
    """Output verbosity levels."""

    MINIMAL = "minimal"  # Essential outputs only
    STANDARD = "standard"  # Default outputs
    FULL = "full"  # All outputs including debug info


@dataclass
class WriterConfig:
    """Configuration for output writers.

    Attributes
    ----------
        verbosity: Level of detail in outputs
        precision: Decimal precision for floating point values
        scientific_notation_threshold: Use scientific notation for values
            smaller than 10^(-threshold) or larger than 10^threshold
        include_comments: Include explanatory comments in outputs
        include_metadata: Include metadata headers
        compress: Compress output files where applicable
        overwrite: Overwrite existing files
    """

    verbosity: Verbosity = Verbosity.STANDARD
    precision: int = 6
    scientific_notation_threshold: int = 4
    include_comments: bool = True
    include_metadata: bool = True
    compress: bool = False
    overwrite: bool = True

    # Format-specific options
    csv_delimiter: str = ","
    csv_quoting: bool = False
    json_indent: int = 2
    json_sort_keys: bool = False


@runtime_checkable
class OutputWriter(Protocol):
    """Protocol for output writers.

    All output writers must implement these methods to ensure
    consistent behavior across formats.
    """

    def write_results(self, results: FitResults, path: Path) -> None:
        """Write complete fit results to a file.

        Args:
            results: FitResults object containing all output data
            path: Output file path
        """
        ...

    def write_parameters(self, results: FitResults, path: Path) -> None:
        """Write parameter estimates to a file.

        Args:
            results: FitResults object
            path: Output file path
        """
        ...

    def write_statistics(self, results: FitResults, path: Path) -> None:
        """Write fit statistics to a file.

        Args:
            results: FitResults object
            path: Output file path
        """
        ...

    def write_diagnostics(self, results: FitResults, path: Path) -> None:
        """Write MCMC diagnostics to a file.

        Args:
            results: FitResults object
            path: Output file path
        """
        ...


def format_float(
    value: float,
    precision: int = 6,
    scientific_threshold: int = 4,
) -> str:
    """Format a float with appropriate notation.

    Uses scientific notation for very large or small values,
    fixed-point otherwise.

    Args:
        value: Value to format
        precision: Number of decimal places
        scientific_threshold: Use scientific notation if |log10(value)| > threshold

    Returns
    -------
        Formatted string
    """
    import math

    if value == 0:
        return f"{0:.{precision}f}"

    if math.isinf(value) or math.isnan(value):
        return str(value)

    log_val = math.log10(abs(value))
    if abs(log_val) > scientific_threshold:
        return f"{value:.{precision}e}"
    return f"{value:.{precision}f}"


def format_uncertainty(
    value: float,
    error: float,
    precision: int = 6,
    scientific_threshold: int = 4,
) -> str:
    """Format a value with its uncertainty.

    Args:
        value: Central value
        error: Uncertainty (standard error)
        precision: Decimal precision
        scientific_threshold: Threshold for scientific notation

    Returns
    -------
        Formatted string like "1.234 ± 0.056"
    """
    val_str = format_float(value, precision, scientific_threshold)
    err_str = format_float(error, precision, scientific_threshold)
    return f"{val_str} ± {err_str}"


def format_asymmetric_uncertainty(
    value: float,
    error_lower: float,
    error_upper: float,
    precision: int = 6,
    scientific_threshold: int = 4,
) -> str:
    """Format a value with asymmetric uncertainties.

    Args:
        value: Central value
        error_lower: Lower error bar (positive value)
        error_upper: Upper error bar (positive value)
        precision: Decimal precision
        scientific_threshold: Threshold for scientific notation

    Returns
    -------
        Formatted string like "1.234 +0.056/-0.045"
    """
    val_str = format_float(value, precision, scientific_threshold)
    up_str = format_float(error_upper, precision, scientific_threshold)
    lo_str = format_float(error_lower, precision, scientific_threshold)
    return f"{val_str} +{up_str}/-{lo_str}"


def get_peak_name(param: ParameterEstimate, peak_names: list[str]) -> str:
    """Extract the original peak name from a parameter.

    Args:
        param: ParameterEstimate object
        peak_names: List of peak names in the cluster

    Returns
    -------
        Original peak name like '2N-H'
    """
    # Use param_id if available (new format)
    if param.param_id is not None:
        return param.param_id.peak_name

    # Legacy fallback: parse from internal name
    param_name = param.name
    for peak_name in peak_names:
        # New format: "peak_name.axis.type"
        if param_name.startswith(peak_name + "."):
            return peak_name
        # Legacy format: sanitized prefix
        safe_prefix = re.sub(r"\W+|^(?=\d)", "_", peak_name)
        if param_name.startswith(safe_prefix + "_"):
            return peak_name

    # Fallback to first peak
    return peak_names[0] if peak_names else ""


def flatten_diagnostics(
    results: FitResults,
) -> Generator[tuple[int, list[str], str, float | None, float | None, float | None, str]]:
    """Yield flattened diagnostic data.

    Yields
    ------
        Tuple of (cluster_id, peak_names, param_name, rhat, ess_bulk, ess_tail, status)
    """
    if not results.mcmc_diagnostics:
        return

    for i, diag in enumerate(results.mcmc_diagnostics):
        cluster_id = results.clusters[i].cluster_id
        peak_names = results.clusters[i].peak_names

        for pd in diag.parameter_diagnostics:
            yield (
                cluster_id,
                peak_names,
                pd.name,
                pd.rhat,
                pd.ess_bulk,
                pd.ess_tail,
                pd.status.value,
            )
