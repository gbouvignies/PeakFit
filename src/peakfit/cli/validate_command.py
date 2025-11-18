"""Implementation of the validate command."""

from pathlib import Path

import nmrglue as ng
from rich.console import Console
from rich.table import Table

console = Console()


def run_validate(spectrum_path: Path, peaklist_path: Path) -> None:
    """Validate input files.

    Args:
        spectrum_path: Path to spectrum file.
        peaklist_path: Path to peak list file.
    """
    console.print("[bold]Validating input files...[/bold]\n")

    errors = []
    warnings = []
    info = {}

    # Validate spectrum file
    console.print(f"[yellow]Checking spectrum:[/yellow] {spectrum_path}")
    try:
        _dic, data = ng.pipe.read(str(spectrum_path))
        info["spectrum_shape"] = data.shape
        info["spectrum_ndim"] = data.ndim

        # Extract spectral parameters
        if data.ndim == 2:
            info["spectrum_type"] = "2D (will be treated as pseudo-3D with 1 plane)"
        elif data.ndim == 3:
            info["spectrum_type"] = f"3D ({data.shape[0]} planes)"
        else:
            warnings.append(f"Unusual dimensionality: {data.ndim}D")

        console.print(f"  [green]OK[/green] - Shape: {data.shape}")
    except Exception as e:
        errors.append(f"Failed to read spectrum: {e}")
        console.print(f"  [red]ERROR[/red] - {e}")

    # Validate peak list file
    console.print(f"\n[yellow]Checking peak list:[/yellow] {peaklist_path}")
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
            info["n_peaks"] = len(peaks)
            console.print(f"  [green]OK[/green] - {len(peaks)} peaks found")

            # Check for duplicate names
            names = [p["name"] for p in peaks]
            if len(names) != len(set(names)):
                warnings.append("Duplicate peak names found")

            # Check position ranges
            x_positions = [p["x"] for p in peaks]
            y_positions = [p["y"] for p in peaks]
            info["x_range"] = (min(x_positions), max(x_positions))
            info["y_range"] = (min(y_positions), max(y_positions))

    except Exception as e:
        errors.append(f"Failed to read peak list: {e}")
        console.print(f"  [red]ERROR[/red] - {e}")

    # Summary table
    console.print("\n[bold]Summary:[/bold]")
    table = Table()
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    for key, value in info.items():
        table.add_row(key.replace("_", " ").title(), str(value))

    console.print(table)

    # Warnings
    if warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warning in warnings:
            console.print(f"  - {warning}")

    # Errors
    if errors:
        console.print("\n[red]Errors:[/red]")
        for error in errors:
            console.print(f"  - {error}")
        console.print("\n[red]Validation failed![/red]")
        raise SystemExit(1)
    console.print("\n[green]Validation passed![/green]")


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
