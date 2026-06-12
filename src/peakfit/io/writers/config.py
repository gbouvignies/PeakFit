"""Configuration for output writers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WriterConfig:
    """Configuration for output writers.

    Attributes:
    ----------
        precision: Decimal precision for floating point values
        scientific_notation_threshold: Use scientific notation for values
            smaller than 10^(-threshold) or larger than 10^threshold
    """

    formats: tuple[str, ...] = ("json", "csv")
    precision: int = 6
    scientific_notation_threshold: int = 4

    def enables(self, fmt: str) -> bool:
        """Return whether a format is enabled."""
        return fmt.lower() in {item.lower() for item in self.formats}


__all__ = [
    "WriterConfig",
]
