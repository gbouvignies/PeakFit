from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from peakfit.core.domain.spectrum import Spectra


class PeakInput(BaseModel):
    """Simple CLI model for a single peak entry with 2D coordinates."""

    name: str
    x: float
    y: float


class SpectraInput(BaseModel):
    """CLI model for inputting a spectrum file with optional z-values."""

    path: Path
    z_values_path: Path | None = None
    exclude_list: list[int] | None = None

    def load(self) -> Spectra:
        """Load and return a `Spectra` object from the provided paths."""
        from peakfit.core.domain.spectrum import read_spectra

        return read_spectra(self.path, self.z_values_path, self.exclude_list)
