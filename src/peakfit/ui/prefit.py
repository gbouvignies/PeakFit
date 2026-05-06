"""Pre-Fit "Manifest" UI component.

Displays a boxed, static manifest summarizing the run configuration
before any computation starts. Establishes trust by proving the program
understands the inputs.

Design Philosophy:
- Fit entirely on one terminal screen
- Readable at a glance
- No "PASS" or "OK" indicators unless something is actually wrong
"""

import multiprocessing
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from peakfit.ui.console import VERSION, console, icon

if TYPE_CHECKING:
    from peakfit.engine.domain.config import PeakFitConfig
    from peakfit.fit.fitting import LoadedData


def _get_version() -> str:
    """Get package version dynamically."""
    try:
        return version("peakfit")
    except Exception:
        return VERSION or "dev"


def _format_workers(workers: int) -> str:
    """Format workers count for display."""
    if workers == -1:
        cpu_count = multiprocessing.cpu_count()
        return f"{cpu_count} (Parallel)"
    if workers == 1:
        return "1 (Sequential)"
    return f"{workers} (Parallel)"


def _format_noise_source(noise_source: str) -> str:
    """Format noise source for display."""
    if noise_source.lower() in ("estimated", "auto"):
        return "Estimated"
    return "User-specified"


def _summarize_spectra(loaded_data: LoadedData) -> tuple[str, int]:
    """Return (shape_type, n_series) for display."""
    data = loaded_data.spectra.data
    n_series = data.shape[0]
    return "Pseudo-ND", n_series


def _describe_method(optimizer: str) -> str:
    opt = optimizer.lower()
    if opt == "varpro":
        return "VARPRO (Variable Projection)"
    if opt == "basin_hopping":
        return "Basin Hopping (Global)"
    if opt == "differential_evolution":
        return "Differential Evolution (Global)"
    return optimizer.upper()


def _format_contour(config: PeakFitConfig, noise_val: float) -> str:
    if config.clustering.contour_level is not None:
        return f"{config.clustering.contour_level:.2e}"

    factor = config.clustering.contour_factor
    contour_val = factor * noise_val
    return f"{contour_val:.2e} (Auto: {factor} × noise)"


def _relative_path(path: Path) -> str:
    cwd = Path.cwd().resolve()
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(cwd))
    except Exception:
        return str(resolved)


def _build_prefit_panel(
    *,
    version: str,
    author: str,
    method_str: str,
    contour_str: str,
    n_clusters: int,
    refine_str: str,
    worker_str: str,
    spectrum_path: Path,
    peaklist_path: Path | None,
    n_series: int,
    shape_type: str,
    n_peaks: int,
    noise_val: float,
    noise_source: str,
    rel_output: str,
) -> Panel:
    # --- Build Content ---
    content_table = Table.grid(padding=(0, 2))
    content_table.add_column(width=14, style="bold")
    content_table.add_column()

    section_header = Text("Input Manifest", style="bold underline")

    input_table = Table.grid(padding=(0, 2))
    input_table.add_column(width=2)
    input_table.add_column(width=12, style="cyan")
    input_table.add_column()

    input_table.add_row(icon("bullet"), "Method:", f"{method_str}")
    input_table.add_row(icon("bullet"), "Contour:", contour_str)
    input_table.add_row(
        icon("bullet"), "Clusters:", f"{n_clusters} (Segmentation based on contour)"
    )
    input_table.add_row(icon("bullet"), "Refine:", refine_str)
    input_table.add_row(icon("bullet"), "Workers:", worker_str)

    files_header = Text("Input Files", style="bold underline")
    files_table = Table.grid(padding=(0, 2))
    files_table.add_column(width=2)
    files_table.add_column(width=12, style="cyan")
    files_table.add_column()

    files_table.add_row(
        icon("bullet"),
        "Spectrum:",
        f"{spectrum_path.name} ({n_series} spectra, {shape_type})",
    )
    if peaklist_path is None:
        files_table.add_row(icon("bullet"), "Peak List:", f"Auto-detected ({n_peaks} peaks)")
    else:
        files_table.add_row(icon("bullet"), "Peak List:", f"{peaklist_path.name} ({n_peaks} peaks)")
    files_table.add_row(
        icon("bullet"),
        "Noise:",
        f"{noise_val:.2e} ({_format_noise_source(noise_source)})",
    )

    output_header = Text("Output", style="bold underline")
    output_table = Table.grid(padding=(0, 2))
    output_table.add_column(width=2)
    output_table.add_column(width=12, style="cyan")
    output_table.add_column()
    output_table.add_row(icon("bullet"), "Directory:", rel_output)

    panel_content = Table.grid(padding=(0, 0))
    panel_content.add_column()
    panel_content.add_row(Text("Command: Fitting", style="dim"))
    panel_content.add_row(Text(f"Author: {author}", style="dim"))
    panel_content.add_row(Text(""))
    panel_content.add_row(section_header)
    panel_content.add_row(input_table)
    panel_content.add_row(Text(""))
    panel_content.add_row(files_header)
    panel_content.add_row(files_table)
    panel_content.add_row(Text(""))
    panel_content.add_row(output_header)
    panel_content.add_row(output_table)

    return Panel(
        panel_content,
        title=f"[header]PeakFit v{version}[/header]",
        title_align="left",
        border_style="panel.border",
        box=box.HEAVY,
        padding=(1, 2),
    )


def show_prefit_check(
    loaded_data: LoadedData,
    output_dir: Path,
    optimizer: str,
    config: PeakFitConfig,
    spectrum_path: Path,
    peaklist_path: Path | None,
    workers: int | str = 1,
) -> None:
    """Display the pre-fit manifest panel.

    Shows a boxed summary of:
    - Program name and version
    - Author credit
    - Input files with metadata
    - Configuration parameters
    - Output directory

    This is an informational display only; fitting starts immediately after.
    """
    version = _get_version()
    author = "Guillaume Bouvignies"

    shape_type, n_series = _summarize_spectra(loaded_data)
    n_peaks = len(loaded_data.peaks)
    n_clusters = len(loaded_data.clusters) if loaded_data.clusters else 0
    noise_val = loaded_data.noise
    noise_source = loaded_data.noise_source

    contour_str = _format_contour(config, noise_val)
    method_str = _describe_method(optimizer)

    refine = config.fitting.refine_iterations
    refine_str = f"{refine} iteration{'s' if refine != 1 else ''}"

    worker_int = int(workers) if isinstance(workers, int) else -1
    worker_str = _format_workers(worker_int)

    rel_output = _relative_path(output_dir)

    panel = _build_prefit_panel(
        version=version,
        author=author,
        method_str=method_str,
        contour_str=contour_str,
        n_clusters=n_clusters,
        refine_str=refine_str,
        worker_str=worker_str,
        spectrum_path=spectrum_path,
        peaklist_path=peaklist_path,
        n_series=n_series,
        shape_type=shape_type,
        n_peaks=n_peaks,
        noise_val=noise_val,
        noise_source=noise_source,
        rel_output=rel_output,
    )

    console.print()
    console.print(panel)
    console.print()
