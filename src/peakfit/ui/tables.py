"""UI tables for displaying structured data.

This module provides functions for creating and displaying Rich tables
with consistent styling across the application.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from rich import box
from rich.table import Table

from .console import console

__all__ = [
    "create_table",
    "print_performance_summary",
    "print_summary",
    "print_validation_table",
]


def create_table(title: str | None = None, show_header: bool = True) -> Table:
    """Create a standard table with consistent styling.

    Args:
        title: Optional table title
        show_header: Whether to show table header

    Returns
    -------
        Configured Table instance
    """
    return Table(
        title=title,
        title_style="header" if title else None,
        box=box.ROUNDED,
        show_header=show_header,
        header_style="bold cyan",
        border_style="dim",
    )


def print_summary(items: dict[str, Any], title: str = "Summary") -> None:
    """Print a standard two-column summary table.

    Args:
        items: Dictionary of key-value pairs to display
        title: Table title
    """
    table = create_table(title, show_header=False)
    table.add_column("Item", style="metric")
    table.add_column("Value", style="value")

    for key, value in items.items():
        table.add_row(key, str(value))

    console.print(table)


def print_validation_table(
    checks: dict[str, tuple[bool, str]],
    title: str = "Input Validation",
) -> None:
    """Print a validation results table.

    Args:
        checks: Dictionary mapping check name to (passed, message) tuple
        title: Table title
    """
    table = create_table(title)
    table.add_column("Check", style="metric")
    table.add_column("Status", style="value", justify="center")

    for check_name, (passed, message) in checks.items():
        status = f"[success]✓[/success] {message}" if passed else f"[warning]⚠[/warning] {message}"
        table.add_row(check_name, status)

    console.print(table)


def print_performance_summary(
    total_time: float,
    n_items: int,
    item_name: str = "items",
    successful: int | None = None,
) -> None:
    """Print a performance summary table.

    Args:
        total_time: Total elapsed time in seconds
        n_items: Number of items processed
        item_name: Name of items being processed
        successful: Optional number of successful items
    """
    table = create_table("⏱️  Performance Summary", show_header=False)
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Value", style="green")

    table.add_row(f"Total {item_name}", str(n_items))
    if successful is not None:
        success_rate = (successful / n_items * 100) if n_items > 0 else 0
        table.add_row("Successful", f"{successful} ({success_rate:.1f}%)")
        if successful < n_items:
            failed = n_items - successful
            table.add_row("Failed", f"[red]{failed}[/]")

    # Format time nicely
    td = timedelta(seconds=total_time)
    if total_time < 60:
        time_str = f"{total_time:.2f}s"
    elif total_time < 3600:
        time_str = f"{td.seconds // 60}m {td.seconds % 60}s"
    else:
        time_str = str(td)

    table.add_row("Total time", time_str)

    if n_items > 0:
        avg_time = total_time / n_items
        avg_str = f"{avg_time * 1000:.0f}ms" if avg_time < 1 else f"{avg_time:.3f}s"
        table.add_row(f"Average per {item_name.rstrip('s')}", avg_str)

    console.print()
    console.print(table)
    console.print()
