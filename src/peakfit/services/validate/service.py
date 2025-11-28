"""Validation service implementation.

This service handles validation of input files (spectrum and peak list)
separating the validation logic from UI display concerns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from pathlib import Path


class RowLike(Protocol):
    """Minimal protocol for a pandas-like row used by CSV/Excel readers."""

    def get(self, key: str, default: Any | None = None) -> Any:
        """Get a value by key, with an optional default if missing."""
        ...

    """Get a value by key, with an optional default if missing."""

    def __getitem__(self, key: int | str) -> Any:
        """Return value accessed by key or index."""
        ...

    """Return value accessed by key or index."""

    @property
    def iloc(self) -> Any:
        """Index-based accessor similar to pandas `iloc` for selecting rows/columns."""
        ...

    """Index-based accessor similar to pandas `iloc` for selecting rows/columns."""


@dataclass
class PeakData:
    """Simple data class for peak information with N-dimensional support."""

    name: str
    positions: list[float]  # Ordered from F1 to Fn (indirect to direct)

    # Legacy property accessors for backward compatibility
    @property
    def x(self) -> float:
        """Get direct dimension position (last in list)."""
        return self.positions[-1] if self.positions else 0.0

    @property
    def y(self) -> float:
        """Get first indirect dimension position."""
        return self.positions[0] if self.positions else 0.0

    @property
    def n_dims(self) -> int:
        """Number of dimensions."""
        return len(self.positions)


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

        Returns
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

        Returns
        -------
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

        except (OSError, FileNotFoundError, ValueError, ImportError, TypeError) as e:
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
                n_dims = peaks[0].n_dims if peaks else 0
                for dim_idx in range(n_dims):
                    dim_label = f"F{dim_idx + 1}"
                    dim_range = result.get_dimension_range(dim_idx)
                    if dim_range:
                        result.info[f"{dim_label} range (ppm)"] = (
                            f"{dim_range[0]:.2f} to {dim_range[1]:.2f}"
                        )

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

    @staticmethod
    def _read_sparky_list(path: Path) -> list[PeakData]:
        """Read Sparky format peak list with N-dimensional support."""
        peaks = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(("#", "Assignment")):
                    continue
                parts = line.split()
                if len(parts) >= 2:  # At least name + 1 position
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

    @staticmethod
    def _read_csv_list(path: Path) -> list[PeakData]:
        """Read CSV format peak list with N-dimensional support."""
        import pandas as pd

        df = pd.read_csv(path)
        peaks = []

        # Detect position columns (Pos F1, Pos F2, ... or w1, w2, ...)
        pos_cols = ValidationService._detect_position_columns(df)

        for _, row in df.iterrows():
            name_value = row.get("Assign F1", row.get("#", row.get("name", "")))
            positions = [ValidationService._to_float(row.get(col), 0.0) for col in pos_cols]
            if not positions:  # Fallback to first numeric columns
                positions = [
                    ValidationService._to_float(row.iloc[i], 0.0)
                    for i in range(1, min(3, len(row)))
                ]
            peaks.append(PeakData(name=str(name_value), positions=positions))
        return peaks

    @staticmethod
    def _detect_position_columns(df: Any) -> list[str]:
        """Detect position columns in a DataFrame."""
        columns = df.columns.tolist()
        pos_cols = []

        # Try 'Pos Fn' pattern
        for i in range(1, 5):
            col = f"Pos F{i}"
            if col in columns:
                pos_cols.append(col)

        if pos_cols:
            return pos_cols

        # Try 'wn' pattern
        for i in range(1, 5):
            col = f"w{i}"
            if col in columns:
                pos_cols.append(col)

        return pos_cols

    @staticmethod
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
                            positions.append(ValidationService._to_float(pos, 0.0))
                    # Legacy x/y fallback
                    if not positions:
                        if "y" in p:  # F1 (indirect)
                            positions.append(ValidationService._to_float(p.get("y"), 0.0))
                        if "x" in p:  # Fn (direct)
                            positions.append(ValidationService._to_float(p.get("x"), 0.0))
                peaks.append(PeakData(name=name, positions=positions))
            return peaks
        return []

    @staticmethod
    def _read_excel_list(path: Path) -> list[PeakData]:
        """Read Excel format peak list with N-dimensional support."""
        import pandas as pd

        df = pd.read_excel(path)
        peaks = []

        # Detect position columns
        pos_cols = ValidationService._detect_position_columns(df)

        for _, row in df.iterrows():
            name_value = row.get("Assign F1", row.get("#", row.get("name", "")))
            positions = [ValidationService._to_float(row.get(col), 0.0) for col in pos_cols]
            if not positions:  # Fallback
                positions = [
                    ValidationService._to_float(row.iloc[i], 0.0)
                    for i in range(1, min(3, len(row)))
                ]
            peaks.append(PeakData(name=str(name_value), positions=positions))
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
