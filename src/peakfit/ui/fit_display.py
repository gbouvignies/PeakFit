"""UI components for displaying fit results.

This module provides specialized display functions for exporting HTML logs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from .console import console

__all__ = [
    "export_html",
]


def export_html(filepath: Path) -> None:
    """Export console output to an HTML file.

    Args:
        filepath: Path to save HTML file
    """
    filepath.write_text(console.export_html())


def print_fit_summary(
    clusters: list,
    peaks: list,
    total_time: float,
) -> None:
    """Print a summary of the fitting results.

    Args:
        clusters: List of fitted clusters
        peaks: List of peaks
        total_time: Total execution time in seconds
    """
    from .tables import create_table

    console.print()
    summary_table = create_table("Results Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="white", justify="right")

    summary_table.add_row("Total clusters", str(len(clusters)))
    summary_table.add_row("Total peaks", str(len(peaks)))

    # successful_clusters = sum(1 for c in clusters if all(p.value is not None for p in c.peaks))
    # For now, we rely on the caller to handle success metrics or pass them in.

    if total_time < 60:
        time_str = f"{total_time:.1f}s"
    else:
        time_str = f"{int(total_time // 60)}m {int(total_time % 60)}s"
    summary_table.add_row("Total time", time_str)

    avg_time_per_cluster = total_time / len(clusters) if len(clusters) > 0 else 0
    summary_table.add_row("Time per cluster", f"{avg_time_per_cluster:.1f}s")
    summary_table.add_row("Mode", "Automatic")

    console.print(summary_table)
    console.print()
