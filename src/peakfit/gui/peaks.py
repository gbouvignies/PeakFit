"""Editable peak coordinates, independent of Qt and fitting objects."""

import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from collections.abc import Mapping

_SPECTRAL_DIMENSIONS = 2


@dataclass(frozen=True, slots=True)
class WorkingPeak:
    """One assignment with stable identity and unfolded (indirect, direct) ppm."""

    id: str
    name: str
    positions: tuple[float, float]

    def __post_init__(self) -> None:
        """Reject invalid coordinates before committing an edit."""
        if not self.name.strip():
            raise ValueError("Give the peak a non-empty name.")
        if len(self.positions) != _SPECTRAL_DIMENSIONS or not all(
            math.isfinite(p) for p in self.positions
        ):
            raise ValueError("Peak positions must be two finite ppm coordinates.")

    @classmethod
    def new(cls, name: str, positions: tuple[float, float]) -> WorkingPeak:
        """Allocate an identity unrelated to the assignment or collection order."""
        return cls(uuid4().hex, name, positions)


class PeakEdits:
    """A small linear history of immutable working peak collections."""

    def __init__(self, peaks: tuple[WorkingPeak, ...] = ()) -> None:
        self._history = [peaks]
        self._index = 0
        self._saved = peaks

    @property
    def peaks(self) -> tuple[WorkingPeak, ...]:
        """Return the committed working coordinates."""
        return self._history[self._index]

    @property
    def dirty(self) -> bool:
        """Whether coordinates or assignments differ from the last saved state."""
        return self.peaks != self._saved

    @property
    def can_undo(self) -> bool:
        """Whether an earlier edit exists."""
        return self._index > 0

    @property
    def can_redo(self) -> bool:
        """Whether an undone edit exists."""
        return self._index + 1 < len(self._history)

    def mark_saved(self) -> None:
        """Record a successful export without discarding undo history."""
        self._saved = self.peaks

    def _commit(self, peaks: tuple[WorkingPeak, ...]) -> None:
        if peaks != self.peaks:
            del self._history[self._index + 1 :]
            self._history.append(peaks)
            self._index += 1

    def add(self, positions: tuple[float, float]) -> WorkingPeak:
        """Add a peak with a readable unused default name."""
        names = {p.name for p in self.peaks}
        number = 1
        while f"Peak {number}" in names:
            number += 1
        peak = WorkingPeak.new(f"Peak {number}", positions)
        self._commit((*self.peaks, peak))
        return peak

    def update(self, peak_id: str, name: str, positions: tuple[float, float]) -> None:
        """Validate and replace a complete peak in one undoable edit."""
        peak = next(p for p in self.peaks if p.id == peak_id)
        updated = replace(peak, name=name, positions=positions)
        self._commit(tuple(updated if p.id == peak_id else p for p in self.peaks))

    def delete(self, peak_id: str) -> None:
        """Delete by identity, preserving assignments of coincident peaks."""
        self._commit(tuple(p for p in self.peaks if p.id != peak_id))

    def update_positions(self, positions: Mapping[str, tuple[float, float]]) -> None:
        """Validate an entire group before committing one position-only history entry."""
        ids = [peak.id for peak in self.peaks]
        if len(set(ids)) != len(ids) or not positions.keys() <= set(ids):
            raise ValueError("Position updates require unique, existing working peak IDs.")
        updated = tuple(
            replace(peak, positions=positions[peak.id]) if peak.id in positions else peak
            for peak in self.peaks
        )
        self._commit(updated)

    def undo(self) -> None:
        """Restore the previous committed collection."""
        if self.can_undo:
            self._index -= 1

    def redo(self) -> None:
        """Restore the next committed collection."""
        if self.can_redo:
            self._index += 1


def export_peaks(path: Path, peaks: tuple[WorkingPeak, ...]) -> None:
    """Atomically export canonical PeakFit CSV with round-trip float precision."""
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.writer(stream)
            writer.writerow(["name", "F1_ppm", "F2_ppm"])
            for peak in peaks:
                writer.writerow([peak.name, *(repr(p) for p in peak.positions)])
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
