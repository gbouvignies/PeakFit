"""Configuration for output writers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Verbosity(StrEnum):
    """Output verbosity levels."""

    MINIMAL = "minimal"  # Essential outputs only
    STANDARD = "standard"  # Default outputs
    FULL = "full"  # All outputs including debug info


@dataclass
class WriterConfig:
    """Configuration for output writers.

    Attributes:
    ----------
        verbosity: Level of detail in outputs
        precision: Decimal precision for floating point values
        scientific_notation_threshold: Use scientific notation for values
            smaller than 10^(-threshold) or larger than 10^threshold
        include_comments: Include explanatory comments in outputs
        include_metadata: Include metadata headers
        compress: Compress output files where applicable
        overwrite: Overwrite existing files
        csv_delimiter: Delimiter for CSV files
        csv_quoting: Whether to minimally quote CSV fields
        json_indent: Indentation level for JSON output
        json_sort_keys: Whether to sort keys in JSON output
    """

    verbosity: Verbosity = Verbosity.STANDARD
    formats: tuple[str, ...] = ("json", "csv", "txt")
    precision: int = 6
    scientific_notation_threshold: int = 4
    include_comments: bool = False
    include_metadata: bool = True
    include_legacy: bool = False
    include_amplitudes_in_summary: bool = False
    save_simulated: bool = False
    compress: bool = False
    overwrite: bool = True

    # Format-specific options
    csv_delimiter: str = ","
    csv_quoting: bool = False
    json_indent: int = 2
    json_sort_keys: bool = False

    def enables(self, fmt: str) -> bool:
        """Return whether a format is enabled."""
        return fmt.lower() in {item.lower() for item in self.formats}


__all__ = [
    "Verbosity",
    "WriterConfig",
]
