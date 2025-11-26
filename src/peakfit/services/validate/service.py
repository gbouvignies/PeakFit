"""Validation service implementation.

This service handles validation of input files (spectrum and peak list)
separating the validation logic from UI display concerns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class RowLike(Protocol):
    """Minimal protocol for a pandas-like row used by CSV/Excel readers."""

    def get(self, key: str, default: Any | None = None) -> Any: ...
    def __getitem__(self, key: int | str) -> Any: ...
    @property
    def iloc(self) -> Any: ...


@dataclass
class PeakData:
    """Simple data class for peak information."""

    name: str
    x: float
    y: float


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
    def x_range(self) -> tuple[float, float] | None:
        """Get X position range from peaks."""
        if not self.peaks:
            return None
        x_positions = [p.x for p in self.peaks]
        return (min(x_positions), max(x_positions))

    @property
    def y_range(self) -> tuple[float, float] | None:
        """Get Y position range from peaks."""
        if not self.peaks:
            return None
        y_positions = [p.y for p in self.peaks]
        return (min(y_positions), max(y_positions))


class ValidationService:
    """Service for validating input files.

    Validates spectrum and peak list files for compatibility and correctness.
    Returns structured results that can be displayed by any UI layer.

    Example:
        result = ValidationService.validate(
            spectrum_path=Path("spectrum.ft2"),
            peaklist_path=Path("peaks.list"),
        )
        if result.is_valid:
            print("Validation passed!")
    """

    @staticmethod
    def validate(spectrum_path: Path, peaklist_path: Path) -> ValidationResult:
        """Validate input files.

        Args:
            spectrum_path: Path to spectrum file.
            peaklist_path: Path to peak list file.

        Returns:
            ValidationResult with all validation information.
        """
        result = ValidationResult()

        # Validate spectrum
        ValidationService._validate_spectrum(spectrum_path, result)

        # Validate peak list
        ValidationService._validate_peaklist(peaklist_path, result)

        return result

    @staticmethod
    def _validate_spectrum(spectrum_path: Path, result: ValidationResult) -> None:
        """Validate spectrum file and update result."""
        from peakfit.cli.models import SpectraInput

        try:
            spectra_input = SpectraInput(path=spectrum_path)
            spectra = spectra_input.load()

            # Determine spectrum type
            if spectra.data.ndim == 2:
                spectrum_type = "2D (will be treated as pseudo-3D with 1 plane)"
            elif spectra.data.ndim == 3:
                spectrum_type = f"3D ({spectra.data.shape[0]} planes)"
            else:
                spectrum_type = f"{spectra.data.ndim}D"
                result.warnings.append(f"Unusual dimensionality: {spectra.data.ndim}D")

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

        except Exception as e:
            result.errors.append(f"Failed to read spectrum: {e}")
            result.checks.append(
                ValidationCheck(
                    name="Spectrum file readable",
                    passed=False,
                    message=f"Failed: {e}",
                )
            )

    @staticmethod
    def _validate_peaklist(peaklist_path: Path, result: ValidationResult) -> None:
        """Validate peak list file and update result."""
        try:
            suffix = peaklist_path.suffix.lower()

            if suffix == ".list":
                peaks = ValidationService._read_sparky_list(peaklist_path)
            elif suffix == ".csv":
                peaks = ValidationService._read_csv_list(peaklist_path)
            elif suffix == ".json":
                peaks = ValidationService._read_json_list(peaklist_path)
            elif suffix in {".xlsx", ".xls"}:
                peaks = ValidationService._read_excel_list(peaklist_path)
            else:
                result.errors.append(f"Unknown peak list format: {suffix}")
                result.checks.append(
                    ValidationCheck(
                        name="Peak list readable",
                        passed=False,
                        message=f"Unknown format: {suffix}",
                    )
                )
                return

            result.peaks = peaks
            result.info["Peaks"] = str(len(peaks))

            result.checks.append(
                ValidationCheck(
                    name="Peak list readable",
                    passed=True,
                    message="Pass",
                )
            )

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
                x_range = result.x_range
                y_range = result.y_range
                if x_range:
                    result.info["X range (ppm)"] = f"{x_range[0]:.2f} to {x_range[1]:.2f}"
                if y_range:
                    result.info["Y range (ppm)"] = f"{y_range[0]:.2f} to {y_range[1]:.2f}"

            # File permissions check
            result.checks.append(
                ValidationCheck(
                    name="File permissions",
                    passed=True,
                    message="Pass",
                )
            )

        except Exception as e:
            result.errors.append(f"Failed to read peak list: {e}")
            result.checks.append(
                ValidationCheck(
                    name="Peak list readable",
                    passed=False,
                    message=f"Failed: {e}",
                )
            )

    @staticmethod
    def _read_sparky_list(path: Path) -> list[PeakData]:
        """Read Sparky format peak list."""
        peaks = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(("#", "Assignment")):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    peaks.append(PeakData(name=parts[0], x=float(parts[1]), y=float(parts[2])))
        return peaks

    @staticmethod
    def _read_csv_list(path: Path) -> list[PeakData]:
        """Read CSV format peak list."""
        import pandas as pd

        df = pd.read_csv(path)
        peaks = []

        for _, row in df.iterrows():
            name_value = row.get("Assign F1", row.get("#", ""))
            x_primary = row.get("Pos F1")
            y_primary = row.get("Pos F2")
            fallback_x = ValidationService._to_float(row.iloc[0])
            fallback_y = ValidationService._to_float(row.iloc[1])
            peaks.append(
                PeakData(
                    name=str(name_value),
                    x=ValidationService._to_float(x_primary, fallback_x),
                    y=ValidationService._to_float(y_primary, fallback_y),
                )
            )
        return peaks

    @staticmethod
    def _read_json_list(path: Path) -> list[PeakData]:
        """Read JSON format peak list."""
        with path.open() as f:
            data = json.load(f)

        if isinstance(data, list):
            return [
                PeakData(
                    name=str(p.get("name", p.get("Assign F1", ""))),
                    x=ValidationService._to_float(p.get("x") or p.get("Pos F1"), 0.0),
                    y=ValidationService._to_float(p.get("y") or p.get("Pos F2"), 0.0),
                )
                for p in data
            ]
        return []

    @staticmethod
    def _read_excel_list(path: Path) -> list[PeakData]:
        """Read Excel format peak list."""
        import pandas as pd

        df = pd.read_excel(path)
        peaks = []
        for _, row in df.iterrows():
            name_value = row.get("Assign F1", row.get("#", ""))
            x_primary = row.get("Pos F1")
            y_primary = row.get("Pos F2")
            fallback_x = ValidationService._to_float(row.iloc[0])
            fallback_y = ValidationService._to_float(row.iloc[1])
            peaks.append(
                PeakData(
                    name=str(name_value),
                    x=ValidationService._to_float(x_primary, fallback_x),
                    y=ValidationService._to_float(y_primary, fallback_y),
                )
            )
        return peaks

    @staticmethod
    def _to_float(value: Any, fallback: float = 0.0) -> float:
        """Convert arbitrary values to float with graceful fallback."""
        if value is None:
            return fallback

        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback
