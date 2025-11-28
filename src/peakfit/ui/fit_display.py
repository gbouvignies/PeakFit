"""UI components for displaying fit results.

This module provides specialized display functions for showing
fitting results, cluster information, and data summaries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from pathlib import Path

from .console import console
from .tables import create_table

__all__ = [
    "create_cluster_status",
    "print_cluster_info",
    "print_data_summary",
    "print_file_item",
    "print_fit_report",
    "print_fit_summary",
    "print_optimization_settings",
    "print_peaks_panel",
]


class HasName(Protocol):
    """Protocol for objects with a name attribute."""

    name: str


def create_cluster_status(
    cluster_index: int,
    total_clusters: int,
    peak_names: list[str],
    status: str = "fitting",
    result: Any | None = None,
) -> Panel:
    """Create a renderable cluster status panel for live display.

    Args:
        cluster_index: Current cluster number (1-based)
        total_clusters: Total number of clusters
        peak_names: List of peak names in this cluster
        status: Status message ("fitting", "optimizing", "done")
        result: Optional optimization result to display

    Returns
    -------
        Panel object that can be rendered
    """
    peaks_str = ", ".join(peak_names)

    # Build content
    content = Text()
    content.append(f"Cluster {cluster_index}/{total_clusters}\n", style="bold cyan")
    content.append("Peaks: ", style="dim")
    content.append(f"{peaks_str}\n\n", style="green")

    if status == "fitting":
        content.append("Status: ", style="dim")
        content.append("● Fitting...", style="bold yellow")
    elif status == "optimizing":
        content.append("Status: ", style="dim")
        content.append("● Optimizing...", style="bold yellow")
    elif status == "done" and result:
        if hasattr(result, "success") and result.success:
            content.append("Status: ", style="dim")
            content.append("✓ Complete", style="bold green")
        else:
            content.append("Status: ", style="dim")
            content.append("⚠ Complete (with issues)", style="bold yellow")

        # Add key statistics
        if hasattr(result, "nfev"):
            content.append(f"\nEvaluations: {result.nfev}", style="dim")
        if hasattr(result, "cost"):
            content.append(f" │ Cost: {result.cost:.2e}", style="dim")

    return Panel(
        content,
        border_style="cyan" if status != "done" else "green",
        padding=(0, 2),
    )


def print_cluster_info(cluster_index: int, total_clusters: int, peak_names: list[str]) -> None:
    """Display information about a cluster being processed.

    Args:
        cluster_index: Current cluster number (1-based)
        total_clusters: Total number of clusters
        peak_names: List of peak names in this cluster
    """
    # Add visual separation before cluster
    console.print()
    console.print("[dim]" + "─" * 60 + "[/dim]")

    peaks_str = ", ".join(peak_names)
    console.print(
        f"[bold cyan]Cluster {cluster_index}/{total_clusters}[/] │ Peaks: [green]{peaks_str}[/]"
    )


def print_fit_report(result: Any | None) -> None:
    """Print fitting results in a styled panel.

    Args:
        result: scipy.optimize.OptimizeResult or similar object
    """
    # Create a table for fit statistics
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Property", style="cyan", width=18)
    table.add_column("Value", style="green")

    if result is not None and hasattr(result, "success"):
        status_style = "green" if result.success else "red"
        status_text = "✓ Success" if result.success else "✗ Failed"
        table.add_row("Status", f"[{status_style}]{status_text}[/]")

    if result is not None and hasattr(result, "message"):
        table.add_row("Message", str(result.message))

    if result is not None and hasattr(result, "nfev"):
        table.add_row("Function evals", str(result.nfev))

    if result is not None and hasattr(result, "njev") and result.njev:
        table.add_row("Jacobian evals", str(result.njev))

    if result is not None and hasattr(result, "cost"):
        table.add_row("Final cost", f"{result.cost:.6e}")

    if result is not None and hasattr(result, "optimality"):
        table.add_row("Optimality", f"{result.optimality:.6e}")

    if result is not None and hasattr(result, "x"):
        table.add_row("Parameters", str(len(result.x)))

    # Display in a panel
    border_style = (
        "green"
        if (result is not None and hasattr(result, "success") and result.success)
        else "yellow"
    )
    panel = Panel(
        table,
        title="[bold]Fit Statistics[/bold]",
        border_style=border_style,
        padding=(0, 1),
    )
    console.print(panel)


def print_peaks_panel(peaks: list[HasName]) -> None:
    """Display list of peaks in a panel.

    Args:
        peaks: List of Peak objects with .name attribute
    """
    peak_list = ", ".join(peak.name for peak in peaks)
    panel = Panel.fit(peak_list, title="Peaks", style="green")
    console.print(panel)


def print_data_summary(
    spectrum_shape: tuple,
    n_planes: int,
    n_peaks: int,
    n_clusters: int,
    noise_level: float,
    noise_source: str,
    contour_level: float,
) -> None:
    """Print a formatted summary of loaded data.

    Args:
        spectrum_shape: Shape of spectrum data
        n_planes: Number of planes (z-values)
        n_peaks: Number of peaks
        n_clusters: Number of clusters
        noise_level: Noise level value
        noise_source: Source of noise level ('user-provided' or 'estimated')
        contour_level: Contour level for clustering
    """
    table = create_table("Data Summary")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Spectrum shape", str(spectrum_shape))
    table.add_row("Number of planes", str(n_planes))
    table.add_row("Number of peaks", str(n_peaks))
    table.add_row("Number of clusters", str(n_clusters))
    table.add_row("Noise level", f"{noise_level:.2f} ({noise_source})")
    table.add_row("Contour level", f"{contour_level:.2f}")

    console.print()
    console.print(table)
    console.print()


def print_fit_summary(
    total_clusters: int,
    total_peaks: int,
    total_time: float,
    success_count: int,
) -> None:
    """Print a summary of fitting results in a panel.

    Args:
        total_clusters: Total number of clusters fitted
        total_peaks: Total number of peaks
        total_time: Total time elapsed
        success_count: Number of successful fits
    """
    console.print()
    panel_content = Text()
    panel_content.append("Fitting Complete\n\n", style="bold green")
    panel_content.append(f"  Clusters fitted:  {total_clusters}\n")
    panel_content.append(f"  Total peaks:      {total_peaks}\n")
    panel_content.append(f"  Successful:       {success_count}/{total_clusters}\n")
    panel_content.append(f"  Total time:       {total_time:.2f}s\n")

    if total_clusters > 0:
        avg_time = total_time / total_clusters
        panel_content.append(f"  Avg per cluster:  {avg_time:.3f}s")

    console.print(Panel(panel_content, border_style="green"))


def print_optimization_settings(ftol: float, xtol: float, max_nfev: int) -> None:
    """Print optimization settings in dim style.

    Args:
        ftol: Function tolerance
        xtol: Parameter tolerance
        max_nfev: Maximum function evaluations
    """
    console.print(f"[dim]Optimization: ftol={ftol:.0e}, xtol={xtol:.0e}, max_nfev={max_nfev}[/]")


def print_file_item(filepath: Path, indent: int = 2) -> None:
    """Print a file path as a bullet item.

    Args:
        filepath: Path to display
        indent: Indentation level
    """
    spaces = "  " * indent
    console.print(f"{spaces}[cyan]‣[/cyan] [path]{filepath}[/path]")


def export_html(filepath: Path) -> None:
    """Export console output to an HTML file.

    Args:
        filepath: Path to save HTML file
    """
    filepath.write_text(console.export_html())
