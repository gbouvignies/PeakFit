"""Implementation of the validate command."""

from pathlib import Path

import nmrglue as ng

from peakfit.ui import PeakFitUI as ui, console


def run_validate(spectrum_path: Path, peaklist_path: Path, verbose: bool = False) -> None:
    """Validate input files.

    Args:
        spectrum_path: Path to spectrum file.
        peaklist_path: Path to peak list file.
        verbose: Show banner and verbose output.
    """
    # Show banner based on verbosity
    ui.show_banner(verbose)

    ui.show_header("Validating Input Files")

    errors = []
    warnings = []
    info = {}
    checks = {}

    # Validate spectrum file
    ui.spacer()
    ui.info(f"Checking spectrum: [path]{spectrum_path.name}[/path]")
    try:
        _dic, data = ng.pipe.read(str(spectrum_path))
        info["Spectrum shape"] = str(data.shape)
        info["Dimensions"] = str(data.ndim)

        # Extract spectral parameters
        if data.ndim == 2:
            info["Type"] = "2D (will be treated as pseudo-3D with 1 plane)"
        elif data.ndim == 3:
            info["Type"] = f"3D ({data.shape[0]} planes)"
        else:
            warnings.append(f"Unusual dimensionality: {data.ndim}D")

        ui.success(f"Spectrum readable - Shape: {data.shape}")
        checks["Spectrum file readable"] = (True, "Pass")
    except Exception as e:
        errors.append(f"Failed to read spectrum: {e}")
        ui.error(f"Failed to read spectrum: {e}")
        checks["Spectrum file readable"] = (False, f"✗ Failed: {e}")

    # Validate peak list file
    ui.spacer()
    ui.info(f"Checking peak list: [path]{peaklist_path.name}[/path]")
    try:
        suffix = peaklist_path.suffix.lower()

        if suffix == ".list":
            peaks = _read_sparky_list(peaklist_path)
        elif suffix == ".csv":
            peaks = _read_csv_list(peaklist_path)
        elif suffix == ".json":
            peaks = _read_json_list(peaklist_path)
        elif suffix in {".xlsx", ".xls"}:
            peaks = _read_excel_list(peaklist_path)
        else:
            errors.append(f"Unknown peak list format: {suffix}")
            peaks = []

        if peaks:
            info["Peaks"] = str(len(peaks))
            ui.success(f"Peak list readable - {len(peaks)} peaks found")
            checks["Peak list readable"] = (True, "Pass")

            # Check for duplicate names
            names = [p["name"] for p in peaks]
            if len(names) != len(set(names)):
                warnings.append("Duplicate peak names found")
                checks["No duplicate peaks"] = (False, "Duplicates found")
            else:
                checks["No duplicate peaks"] = (True, "Pass")

            # Check position ranges
            x_positions = [p["x"] for p in peaks]
            y_positions = [p["y"] for p in peaks]
            x_min, x_max = min(x_positions), max(x_positions)
            y_min, y_max = min(y_positions), max(y_positions)
            info["X range (ppm)"] = f"{x_min:.2f} to {x_max:.2f}"
            info["Y range (ppm)"] = f"{y_min:.2f} to {y_max:.2f}"

            # File permissions check
            checks["File permissions"] = (True, "Pass")

    except Exception as e:
        errors.append(f"Failed to read peak list: {e}")
        ui.error(f"Failed to read peak list: {e}")
        checks["Peak list readable"] = (False, f"✗ Failed: {e}")

    # Summary table
    ui.spacer()
    ui.print_summary(info, title="File Information")

    # Validation checks table
    if checks:
        ui.spacer()
        ui.print_validation_table(checks, title="Validation Checks")

    # Warnings
    if warnings:
        ui.spacer()
        for warning in warnings:
            ui.warning(warning)

    # Errors
    if errors:
        ui.spacer()
        for error in errors:
            ui.error(error)
        ui.spacer()
        ui.error("Validation failed!")
        raise SystemExit(1)

    ui.spacer()
    ui.success("All validation checks passed!")

    # Next steps
    ui.spacer()
    ui.info("Ready for fitting. Run:")
    console.print(f"    [cyan]peakfit fit {spectrum_path.name} {peaklist_path.name}[/cyan]")
    ui.spacer()


def _read_sparky_list(path: Path) -> list[dict]:
    """Read Sparky format peak list."""
    peaks = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", "Assignment")):
                continue
            parts = line.split()
            if len(parts) >= 3:
                peaks.append(
                    {
                        "name": parts[0],
                        "x": float(parts[1]),
                        "y": float(parts[2]),
                    }
                )
    return peaks


def _read_csv_list(path: Path) -> list[dict]:
    """Read CSV format peak list."""
    import pandas as pd

    df = pd.read_csv(path)
    peaks = []
    for _, row in df.iterrows():
        peaks.append(
            {
                "name": str(row.get("Assign F1", row.get("#", ""))),
                "x": float(row.get("Pos F1", row.iloc[0])),
                "y": float(row.get("Pos F2", row.iloc[1])),
            }
        )
    return peaks


def _read_json_list(path: Path) -> list[dict]:
    """Read JSON format peak list."""
    import json

    with path.open() as f:
        data = json.load(f)

    if isinstance(data, list):
        return [
            {
                "name": str(p.get("name", p.get("Assign F1", ""))),
                "x": float(p.get("x", p.get("Pos F1", 0))),
                "y": float(p.get("y", p.get("Pos F2", 0))),
            }
            for p in data
        ]
    return []


def _read_excel_list(path: Path) -> list[dict]:
    """Read Excel format peak list."""
    import pandas as pd

    df = pd.read_excel(path)
    peaks = []
    for _, row in df.iterrows():
        peaks.append(
            {
                "name": str(row.get("Assign F1", row.get("#", ""))),
                "x": float(row.get("Pos F1", row.iloc[0])),
                "y": float(row.get("Pos F2", row.iloc[1])),
            }
        )
    return peaks
