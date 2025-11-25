from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from peakfit.core.domain.spectrum import Spectra


class PeakInput(BaseModel):
    name: str
    x: float
    y: float


class SpectraInput(BaseModel):
    path: Path
    z_values_path: Path | None = None
    exclude_list: list[int] | None = None

    def load(self) -> Spectra:
        from peakfit.core.domain.spectrum import read_spectra

        return read_spectra(self.path, self.z_values_path, self.exclude_list)
