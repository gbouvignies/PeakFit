"""Editable peak coordinates and the working-input history, independent of Qt and fitting."""

import csv
import math
from dataclasses import astuple, dataclass, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from peakfit.engine.domain.constraints import PARAMETER_KEY
from peakfit.gui import formatting

_SPECTRAL_DIMENSIONS = 2


@dataclass(frozen=True, slots=True)
class ParameterLimit:
    """A peak's own constraint on one of its parameters; None keeps the model settings'.

    ``vary`` False holds the parameter at its start, True fits one a rule holds.
    ``window`` applies to a position only: how far (ppm) it may move from the working
    position. Values are in the parameter's unit and need not be consistent with each
    other here: the editor and the preparation explain and refuse inconsistent ones.
    """

    start: float | None = None
    lower: float | None = None
    upper: float | None = None
    vary: bool | None = None
    window: float | None = None

    def __post_init__(self) -> None:
        """Reject values no fit can use: non-finite numbers or a window that is not positive."""
        numbers = (self.start, self.lower, self.upper, self.window)
        if any(v is not None and not math.isfinite(v) for v in numbers):
            raise ValueError("Constraint values must be finite numbers.")
        if self.window is not None and self.window <= 0:
            raise ValueError("A position window must be a positive number of ppm.")

    @property
    def empty(self) -> bool:
        """Whether the limit leaves everything to the model settings."""
        return all(value is None for value in astuple(self))


type PeakLimits = tuple[tuple[str, ParameterLimit], ...]  # by parameter (``F3.lw``), sorted


@dataclass(frozen=True, slots=True)
class WorkingPeak:
    """One assignment with stable identity, unfolded (indirect, direct) ppm and own limits.

    ``limits`` are the peak's own parameter constraints. They belong to this working
    peak, not to its assignment: they follow it through renames, and peaks sharing an
    assignment have their own.
    """

    id: str
    name: str
    positions: tuple[float, float]
    limits: PeakLimits = ()

    def __post_init__(self) -> None:
        """Reject invalid coordinates or limits before committing an edit."""
        if not self.name.strip():
            raise ValueError("Give the peak a non-empty name.")
        if len(self.positions) != _SPECTRAL_DIMENSIONS or not all(
            math.isfinite(p) for p in self.positions
        ):
            raise ValueError("Peak positions must be two finite ppm coordinates.")
        keys = [key for key, _limit in self.limits]
        if keys != sorted(set(keys)) or not all(PARAMETER_KEY.fullmatch(k) for k in keys):
            raise ValueError("Peak limits must name each parameter (F3.lw) once, in order.")
        for key, limit in self.limits:
            if limit.empty:
                raise ValueError("An empty peak limit is no limit; leave it out.")
            if limit.window is not None and not key.endswith(".cs"):
                raise ValueError("A position window applies to a position only.")

    def limit(self, key: str) -> ParameterLimit:
        """The peak's own limit on parameter ``key``; empty when it has none."""
        return dict(self.limits).get(key, ParameterLimit())

    def with_limit(self, key: str, limit: ParameterLimit) -> WorkingPeak:
        """This peak with ``limit`` on ``key``; an empty limit removes it."""
        limits = {**dict(self.limits), key: limit}
        kept = sorted(((k, v) for k, v in limits.items() if not v.empty), key=lambda kv: kv[0])
        return replace(self, limits=tuple(kept))

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

    def constrain(self, peak_id: str, limits: PeakLimits) -> None:
        """Replace one peak's own parameter limits in one undoable edit."""
        peak = next(p for p in self.peaks if p.id == peak_id)
        updated = replace(peak, limits=limits)
        self._commit(
            tuple(updated if p.id == peak_id else p for p in self.peaks),
            self.settings,
            f"constraints of {peak.name}",
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
