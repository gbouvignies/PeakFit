"""UI tables for displaying structured data.

This module provides functions for creating and displaying Rich tables
with consistent styling across the application.
"""

from typing import Any

from rich import box
from rich.table import Table

from .console import console, icon

__all__ = [
    "create_live_metrics_table",
    "create_metadata_grid",
    "create_table",
    "print_summary",
    "print_validation_table",
]


def create_metadata_grid(metadata: dict[str, Any]) -> Table:
    """Create a grid for displaying metadata.

    Args:
        metadata: Dictionary of key-value pairs to display.
    """
    table = Table(box=None, show_header=False, expand=True)
    table.add_column(style="key", justify="right")
    table.add_column(style="value")
    for key, value in metadata.items():
        table.add_row(f"{key}:", str(value))
    return table


def create_live_metrics_table(metrics: dict[str, Any]) -> Table:
    """Create a compact table for live metrics.

    Args:
        metrics: Dictionary of metric name -> (value, style) or just value.

    Returns:
    -------
        A Table for displaying live metrics.
    """
    table = Table(box=box.SIMPLE, show_header=False, show_edge=False, pad_edge=False)

    row_values = []
    for label, val in metrics.items():
        table.add_column(label, justify="center")

        style = "value"
        display_val = str(val)

        # Handle (value, style) tuple
        if isinstance(val, tuple):
            display_val = str(val[0])
            style = val[1]

        row_values.append(f"[key]{label}:[/key] [{style}]{display_val}[/{style}]")

    table.add_row(*row_values)
    return table


def create_table(
    title: str | None = None,
    caption: str | None = None,
    show_header: bool = True,
) -> Table:
    """Create a standard table with consistent styling.

    Args:
        title: Optional table title
        caption: Optional table caption
        show_header: Whether to show table header

    Returns:
    -------
        Configured Table instance
    """
    return Table(
        title=title,
        caption=caption,
        title_style="panel.title" if title else None,
        caption_style="neutral",
        box=box.ROUNDED,
        show_header=show_header,
        header_style="subheader",
        border_style="box.border",
        expand=False,
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
        if passed:
            status = f"[success]{icon('check')}[/success] {message}"
        else:
            status = f"[warning]{icon('warn')}[/warning] {message}"
        table.add_row(check_name, status)

    console.print(table)
