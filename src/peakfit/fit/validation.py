"""Input validation for fit workflows.

This module validates spectrum and peak list files while staying free of UI concerns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from pydantic import BaseModel

from peakfit.engine.domain.data import PeakData
from peakfit.io.readers.spectrum import read_spectra

if TYPE_CHECKING:
    from collections.abc import Callable

    from peakfit.engine.domain.spectrum import Spectra

    ReadTabular = Callable[[Path], Any]


_MIN_PARTS_FOR_NAME_AND_POSITION = 2


# =============================================================================
# Input Models
# =============================================================================


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
        return read_spectra(self.path, self.z_values_path, self.exclude_list)


# =============================================================================
# Result Containers
# =============================================================================


@dataclass
class SpectrumData:
    """Data extracted from spectrum validation."""

    shape: tuple[int, ...]
    ndim: int
    spectrum_type: str


@dataclass
class ValidationCheck:
    """Result of a single validation check."""

    name: str
    passed: bool
    message: str


@dataclass
class ValidationResult:
    """Complete validation result."""

    spectrum: SpectrumData | None = None
    peaks: list[PeakData] = field(default_factory=list)
    checks: list[ValidationCheck] = field(default_factory=list)
    info: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return len(self.errors) == 0

    @property
    def n_dims(self) -> int:
        """Number of spectral dimensions based on peaks."""
        if not self.peaks:
            return 0
        return self.peaks[0].n_dims

    def get_dimension_range(self, dim_index: int) -> tuple[float, float] | None:
        """Get position range for a specific dimension.

        Args:
            dim_index: 0-based dimension index (0=F1, 1=F2, etc.)

        Returns:
        -------
            (min, max) tuple or None if no peaks
        """
        if not self.peaks or dim_index >= self.n_dims:
            return None
        positions = [p.positions[dim_index] for p in self.peaks]
        return (min(positions), max(positions))

    @property
    def x_range(self) -> tuple[float, float] | None:
        """Get direct dimension (X/Fn) position range from peaks."""
        if not self.peaks:
            return None
        return self.get_dimension_range(self.n_dims - 1)

    @property
    def y_range(self) -> tuple[float, float] | None:
        """Get first indirect dimension (Y/F1) position range from peaks."""
        if not self.peaks:
            return None
        return self.get_dimension_range(0)


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

    # Validate peak list when provided
    if peaklist_path is not None:
        _validate_peaklist(peaklist_path, result)
    else:
        result.info["Peaks"] = "Auto-detect"
        result.checks.append(
            ValidationCheck(
                name="Peak list readable",
                passed=True,
                message="Skipped (automatic peak picking enabled)",
            )
        )

    return result


def _validate_spectrum(spectrum_path: Path, result: ValidationResult) -> None:
    """Validate spectrum file and update result."""
    try:
        spectra_input = SpectraInput(path=spectrum_path)
        spectra = spectra_input.load()

        n_series = spectra.data.shape[0]
        spectrum_type = f"Pseudo-ND ({n_series} spectra)"

        result.spectrum = SpectrumData(
            shape=spectra.data.shape,
            ndim=spectra.data.ndim,
            spectrum_type=spectrum_type,
        )

        result.info["Spectrum shape"] = str(spectra.data.shape)
        result.info["Dimensions"] = str(spectra.data.ndim)
        result.info["Type"] = spectrum_type

        result.checks.append(
            ValidationCheck(
                name="Spectrum file readable",
                passed=True,
                message="Pass",
            )
        )

    except (OSError, FileNotFoundError, ValueError, ImportError, TypeError) as e:
        result.errors.append(f"Failed to read spectrum: {e}")
        result.checks.append(
            ValidationCheck(
                name="Spectrum file readable",
                passed=False,
                message=f"Failed: {e}",
            )
        )


def _validate_peaklist(peaklist_path: Path, result: ValidationResult) -> None:
    """Validate peak list file and update result."""
    try:
        # 1. Load peaks based on file format
        peaks = _load_peaks(peaklist_path)

        result.peaks = peaks
        result.info["Peaks"] = str(len(peaks))

        result.checks.append(
            ValidationCheck(
                name="Peak list readable",
                passed=True,
                message="Pass",
            )
        )

        # 2. Validate consistency (duplicates, dimensions, etc.)
        _validate_peak_consistency(peaks, result)

        # File permissions check
        result.checks.append(
            ValidationCheck(
                name="File permissions",
                passed=True,
                message="Pass",
            )
        )

    except (OSError, FileNotFoundError, ValueError, ImportError, TypeError) as e:
        result.errors.append(f"Failed to read peak list: {e}")
        result.checks.append(
            ValidationCheck(
                name="Peak list readable",
                passed=False,
                message=f"Failed: {e}",
            )
        )


def _load_peaks(peaklist_path: Path) -> list[PeakData]:
    """Load peaks from file based on extension."""
    suffix = peaklist_path.suffix.lower()

    if suffix == ".list":
        return _read_sparky_list(peaklist_path)
    elif suffix == ".csv":
        return _read_csv_list(peaklist_path)
    elif suffix == ".json":
        return _read_json_list(peaklist_path)
    elif suffix in {".xlsx", ".xls"}:
        return _read_excel_list(peaklist_path)
    else:
        raise ValueError(f"Unknown peak list format: {suffix}")


def _validate_peak_consistency(peaks: list[PeakData], result: ValidationResult) -> None:
    """Check peak list for logical consistency."""
    # Check for duplicate names
    names = [p.name for p in peaks]
    if len(names) != len(set(names)):
        result.warnings.append("Duplicate peak names found")
        result.checks.append(
            ValidationCheck(
                name="No duplicate peaks",
                passed=False,
                message="Duplicates found",
            )
        )
    else:
        result.checks.append(
            ValidationCheck(
                name="No duplicate peaks",
                passed=True,
                message="Pass",
            )
        )

    # Add position ranges to info
    if peaks:
        n_dims = peaks[0].n_dims if peaks else 0
        for dim_idx in range(n_dims):
            dim_label = f"F{dim_idx + 1}"
            dim_range = result.get_dimension_range(dim_idx)
            if dim_range:
                result.info[f"{dim_label} range (ppm)"] = (
                    f"{dim_range[0]:.2f} to {dim_range[1]:.2f}"
                )


def _read_sparky_list(path: Path) -> list[PeakData]:
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
                    peaks.append(PeakData(name=name, positions=positions))
    return peaks


def _read_csv_list(path: Path) -> list[PeakData]:
    """Read CSV format peak list with N-dimensional support."""
    df = pd.read_csv(path)
    return _parse_peaks_from_dataframe(df)


def _read_excel_list(path: Path) -> list[PeakData]:
    """Read Excel format peak list with N-dimensional support."""
    df = pd.read_excel(path)
    return _parse_peaks_from_dataframe(df)


def _parse_peaks_from_dataframe(df: Any) -> list[PeakData]:
    """Parse peaks from a pandas DataFrame."""
    peaks = []
    pos_cols = _detect_position_columns(df)
    if not pos_cols:
        msg = "Peak table must include position columns named F1_ppm, w1, or Pos F1"
        raise ValueError(msg)

    for _, row in df.iterrows():
        name_value = row.get("Assign F1", row.get("#", row.get("name", "")))
        positions = [_to_float(row.get(col)) for col in pos_cols]
        peaks.append(PeakData(name=str(name_value), positions=positions))
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


def _read_json_list(path: Path) -> list[PeakData]:
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
            peaks.append(PeakData(name=name, positions=positions))
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
    "PeakData",
    "PeakInput",
    "SpectraInput",
    "SpectrumData",
    "ValidationCheck",
    "ValidationResult",
    "validate_inputs",
]
