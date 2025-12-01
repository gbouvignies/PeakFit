"""UI tables for displaying structured data.

This module provides functions for creating and displaying Rich tables
with consistent styling across the application.
"""

from __future__ import annotations

from typing import Any

from rich import box
from rich.table import Table

from .console import console

__all__ = [
    "create_table",
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
