"""Input validation for fit workflows.

This module validates spectrum and peak list files while staying free of UI concerns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from peakfit.io.readers.spectrum import read_spectra

if TYPE_CHECKING:
    from pathlib import Path

_MIN_PARTS_FOR_NAME_AND_POSITION = 2

type PeakRow = tuple[str, list[float]]


@dataclass
class ValidationResult:
    """Blocking input validation errors for a fit run."""

    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return len(self.errors) == 0


def validate_inputs(spectrum_path: Path, peaklist_path: Path | None) -> ValidationResult:
    """Validate input files.

    Args:
        spectrum_path: Path to spectrum file.
        peaklist_path: Path to peak list file (optional).

    Returns:
    -------
        ValidationResult with all validation information.
    """
    result = ValidationResult()

    _validate_spectrum(spectrum_path, result)

    if peaklist_path is not None:
        _validate_peaklist(peaklist_path, result)

    return result


def _validate_spectrum(spectrum_path: Path, result: ValidationResult) -> None:
    """Validate spectrum file and update result."""
    try:
        read_spectra(spectrum_path, None, None)
    except (OSError, FileNotFoundError, ValueError, ImportError, TypeError) as e:
        result.errors.append(f"Failed to read spectrum: {e}")


def _validate_peaklist(peaklist_path: Path, result: ValidationResult) -> None:
    """Validate peak list file and update result."""
    try:
        _load_peak_rows(peaklist_path)
    except (OSError, FileNotFoundError, ValueError, ImportError, TypeError) as e:
        result.errors.append(f"Failed to read peak list: {e}")


def _load_peak_rows(peaklist_path: Path) -> list[PeakRow]:
    """Load enough peak-list data to validate the file."""
    suffix = peaklist_path.suffix.lower()

    if suffix == ".list":
        return _read_sparky_list(peaklist_path)
    if suffix == ".csv":
        return _read_csv_list(peaklist_path)
    if suffix == ".json":
        return _read_json_list(peaklist_path)
    if suffix in {".xlsx", ".xls"}:
        return _read_excel_list(peaklist_path)
    raise ValueError(f"Unknown peak list format: {suffix}")


def _read_sparky_list(path: Path) -> list[PeakRow]:
    """Read Sparky format peak list with N-dimensional support."""
    peaks = []
    with path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith(("#", "Assignment")):
                continue
            parts = line.split()
            if len(parts) >= _MIN_PARTS_FOR_NAME_AND_POSITION:  # At least name + 1 position
                name = parts[0]
                # All remaining numeric parts are positions
                positions = []
                for part in parts[1:]:
                    try:
                        positions.append(float(part))
                    except ValueError:
                        break  # Stop at first non-numeric
                if positions:
                    peaks.append((name, positions))
    return peaks


def _read_csv_list(path: Path) -> list[PeakRow]:
    """Read CSV format peak list with N-dimensional support."""
    df = pd.read_csv(path)
    return _parse_peaks_from_dataframe(df)


def _read_excel_list(path: Path) -> list[PeakRow]:
    """Read Excel format peak list with N-dimensional support."""
    df = pd.read_excel(path)
    return _parse_peaks_from_dataframe(df)


def _parse_peaks_from_dataframe(df: Any) -> list[PeakRow]:
    """Parse peaks from a pandas DataFrame."""
    peaks = []
    pos_cols = _detect_position_columns(df)
    if not pos_cols:
        msg = "Peak table must include position columns named F1_ppm, w1, or Pos F1"
        raise ValueError(msg)

    for _, row in df.iterrows():
        name_value = row.get("Assign F1", row.get("#", row.get("name", "")))
        positions = [_to_float(row.get(col)) for col in pos_cols]
        peaks.append((str(name_value), positions))
    return peaks


def _detect_position_columns(df: Any) -> list[str]:
    """Detect position columns in a DataFrame."""
    columns = df.columns.tolist()
    pos_cols = []

    # Try canonical 'Fn_ppm' pattern.
    for i in range(1, 5):
        col = f"F{i}_ppm"
        if col in columns:
            pos_cols.append(col)

    if pos_cols:
        return pos_cols

    # Try CCPN 'Pos Fn' pattern.
    for i in range(1, 5):
        col = f"Pos F{i}"
        if col in columns:
            pos_cols.append(col)

    if pos_cols:
        return pos_cols

    # Try Sparky 'wn' pattern.
    for i in range(1, 5):
        col = f"w{i}"
        if col in columns:
            pos_cols.append(col)

    return pos_cols


def _read_json_list(path: Path) -> list[PeakRow]:
    """Read JSON format peak list with N-dimensional support."""
    with path.open() as f:
        data = json.load(f)

    if isinstance(data, list):
        peaks = []
        for p in data:
            name = str(p.get("name", p.get("Assign F1", "")))
            # Try 'positions' array first
            if "positions" in p and isinstance(p["positions"], list):
                positions = [float(x) for x in p["positions"]]
            else:
                # Fall back to individual position fields
                positions = []
                for i in range(1, 5):  # F1 to F4
                    pos = p.get(f"Pos F{i}") or p.get(f"w{i}")
                    if pos is not None:
                        positions.append(_to_float(pos))
            peaks.append((name, positions))
        return peaks
    return []


def _to_float(value: Any) -> float:
    """Convert a peak-list value to float."""
    if value is None:
        raise ValueError("Missing peak position value")

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid peak position value: {value!r}") from exc


__all__ = [
    "ValidationResult",
    "validate_inputs",
]
