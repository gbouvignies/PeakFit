"""Base writer interface and configuration.

This module defines the OutputWriter protocol and common configuration
for all output writers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from peakfit.core.results.fit_results import FitResults


class Verbosity(str, Enum):
    """Output verbosity levels."""

    MINIMAL = "minimal"  # Essential outputs only
    STANDARD = "standard"  # Default outputs
    FULL = "full"  # All outputs including debug info


@dataclass
class WriterConfig:
    """Configuration for output writers.

    Attributes:
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

    Returns:
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

    Returns:
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

    Returns:
        Formatted string like "1.234 +0.056/-0.045"
    """
    val_str = format_float(value, precision, scientific_threshold)
    up_str = format_float(error_upper, precision, scientific_threshold)
    lo_str = format_float(error_lower, precision, scientific_threshold)
    return f"{val_str} +{up_str}/-{lo_str}"
