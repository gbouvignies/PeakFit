"""Editable peak coordinates and the working-input history, independent of Qt and fitting."""

import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from peakfit.gui import formatting

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


class WorkingEdits[S]:
    """One linear history of the working inputs: peak collections and fit settings.

    Peak edits and settings edits share one Undo. ``S`` is the settings value, compared
    by equality; this module never interprets it.
    """

    def __init__(self, peaks: tuple[WorkingPeak, ...], settings: S, *, revision: int = 0) -> None:
        self._history: list[tuple[tuple[WorkingPeak, ...], S]] = [(peaks, settings)]
        self._index = 0
        self.revision = revision
        self._labels = ["edit"]

    @property
    def peaks(self) -> tuple[WorkingPeak, ...]:
        """Return the committed working coordinates."""
        return self._history[self._index][0]

    @property
    def settings(self) -> S:
        """Return the committed working fit settings."""
        return self._history[self._index][1]

    @property
    def can_undo(self) -> bool:
        """Whether an earlier edit exists."""
        return self._index > 0

    @property
    def can_redo(self) -> bool:
        """Whether an undone edit exists."""
        return self._index + 1 < len(self._history)

    def accept_completion(self, peaks: tuple[WorkingPeak, ...]) -> None:
        """Atomically adopt verified run positions without a manual revision or Undo.

        The settings stay as submitted. The caller publishes DatasetCompletion in the
        same event delivery; the analysis stays Modified until it is saved.
        """
        if tuple(p.id for p in peaks) != tuple(p.id for p in self.peaks):
            raise ValueError("Completion must preserve ordered stable peak identities.")
        self._history = [(peaks, self.settings)]
        self._labels = ["edit"]
        self._index = 0

    @property
    def undo_label(self) -> str:
        """Describe the scientific edit that Undo would restore."""
        return self._labels[self._index] if self.can_undo else "edit"

    @property
    def redo_label(self) -> str:
        """Describe the scientific edit that Redo would replay."""
        return self._labels[self._index + 1] if self.can_redo else "edit"

    def replace_all(self, peaks: tuple[WorkingPeak, ...], label: str = "import peaks") -> None:
        """Replace or append a validated import as a single undoable scientific edit."""
        if len({p.id for p in peaks}) != len(peaks):
            raise ValueError("Working peak IDs must be unique.")
        self._commit(peaks, self.settings, label)

    def configure(self, settings: S, label: str = "settings change") -> None:
        """Commit edited fit settings as one undoable scientific edit."""
        self._commit(self.peaks, settings, label)

    def delete_many(self, identities: set[str]) -> None:
        """Delete a list selection atomically; names need not be unique."""
        self._commit(
            tuple(p for p in self.peaks if p.id not in identities),
            self.settings,
            f"delete {len(identities)} peaks" if len(identities) != 1 else "delete peak",
        )

    def _commit(self, peaks: tuple[WorkingPeak, ...], settings: S, label: str) -> None:
        if (peaks, settings) != self._history[self._index]:
            del self._history[self._index + 1 :]
            self._history.append((peaks, settings))
            del self._labels[self._index + 1 :]
            self._labels.append(label)
            self._index += 1
            self.revision += 1

    def add(self, positions: tuple[float, float]) -> WorkingPeak:
        """Add a peak with a readable unused default name."""
        names = {p.name for p in self.peaks}
        number = 1
        while f"Peak {number}" in names:
            number += 1
        peak = WorkingPeak.new(f"Peak {number}", positions)
        self._commit((*self.peaks, peak), self.settings, f"add {peak.name}")
        return peak

    def update(self, peak_id: str, name: str, positions: tuple[float, float]) -> None:
        """Validate and replace a complete peak in one undoable edit."""
        peak = next(p for p in self.peaks if p.id == peak_id)
        updated = replace(peak, name=name, positions=positions)
        label = "move" if peak.name == name else "edit"
        self._commit(
            tuple(updated if p.id == peak_id else p for p in self.peaks),
            self.settings,
            f"{label} {name}",
        )

    def delete(self, peak_id: str) -> None:
        """Delete by identity, preserving assignments of coincident peaks."""
        self.delete_many({peak_id})

    def undo(self) -> None:
        """Restore the previous committed peaks and settings."""
        if self.can_undo:
            self._index -= 1
            self.revision += 1

    def redo(self) -> None:
        """Restore the next committed peaks and settings."""
        if self.can_redo:
            self._index += 1
            self.revision += 1


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
                writer.writerow([peak.name, *(formatting.exact(p) for p in peak.positions)])
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
