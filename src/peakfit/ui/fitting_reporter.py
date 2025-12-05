"""Reporting utilities for the fitting pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from peakfit.ui import console, create_table, spacer

if TYPE_CHECKING:
    from peakfit.core.domain.spectrum import Spectra


def print_configuration(output_dir: Path, workers: int) -> None:
    """Print configuration information in a consolidated table."""
    spacer()

    config_table = create_table("Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="white", justify="right")

    config_table.add_row("Backend", "numpy")

    # Show parallel processing status
    if workers == 1:
        parallel_str = "disabled (sequential)"
    elif workers == -1:
        import os

        cpu_count = os.cpu_count() or 1
        parallel_str = f"enabled (all {cpu_count} CPUs)"
    else:
        parallel_str = f"enabled ({workers} workers)"
    config_table.add_row("Parallel processing", parallel_str)

    config_table.add_row("Output directory", str(output_dir.name))

    console.print(config_table)
    spacer()


def print_spectrum_info(
    spectrum_path: Path,
    spectra: Spectra,
    shape_names: list[str],
    noise: float,
    noise_source: str,
    contour_level: float,
) -> None:
    """Print consolidated spectrum information table."""
    spacer()

    spectrum_table = create_table(f"Spectrum: {spectrum_path.name}")
    spectrum_table.add_column("Property", style="cyan")
    spectrum_table.add_column("Value", style="white", justify="right")

    shape = spectra.data.shape
    n_spectral = spectra.n_spectral_dims

    # Build dimension string with Fn labels
    if n_spectral >= 1:
        dim_parts = []
        for dim in spectra.dimensions:
            size = dim.size
            label = dim.label
            nucleus = f" ({dim.nucleus})" if dim.nucleus else ""
            dim_parts.append(f"{label}{nucleus}: {size} pts")
        dim_str = ", ".join(dim_parts)
        # Also show shape
        shape_str = " × ".join(str(s) for s in reversed(shape[1:]))
        dim_str = f"{shape_str} ({dim_str})"
    else:
        dim_str = str(shape)

    spectrum_table.add_row("Spectral dimensions", str(n_spectral))
    spectrum_table.add_row("Dimension sizes", dim_str)
    spectrum_table.add_row("Number of planes", str(len(spectra.z_values)))

    lineshape_str = ", ".join(shape_names) if isinstance(shape_names, list) else str(shape_names)
    spectrum_table.add_row("Lineshapes", lineshape_str)
    spectrum_table.add_row("Noise level", f"{noise:.2f} ({noise_source})")
    spectrum_table.add_row("Contour level", f"{contour_level:.2f}")

    console.print(spectrum_table)
    spacer()


def print_peaklist_info(peaklist_path: Path, z_values_path: Path | None, n_peaks: int) -> None:
    """Print consolidated peak list information table."""
    spacer()

    peaklist_table = create_table(f"Peak List: {peaklist_path.name}")
    peaklist_table.add_column("Property", style="cyan")
    peaklist_table.add_column("Value", style="white", justify="right")

    suffix = peaklist_path.suffix.lower()
    if suffix == ".list":
        format_str = "Sparky/NMRPipe"
    elif suffix == ".csv":
        format_str = "CSV"
    elif suffix == ".json":
        format_str = "JSON"
    elif suffix in {".xlsx", ".xls"}:
        format_str = "Excel"
    else:
        format_str = "Unknown"

    peaklist_table.add_row("Format", format_str)
    peaklist_table.add_row("Number of peaks", str(n_peaks))

    if z_values_path:
        peaklist_table.add_row("Z-values file", z_values_path.name)
    else:
        peaklist_table.add_row("Z-values file", "auto-detected")

    console.print(peaklist_table)
    spacer()
