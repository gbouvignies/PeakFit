"""UI panels for displaying boxed content.

This module provides functions for creating and displaying Rich panels
with consistent styling across the application.
"""

from __future__ import annotations

from rich import box
from rich.panel import Panel

from .console import console

__all__ = [
    "create_panel",
    "print_panel",
]


def create_panel(
    content: str,
    title: str | None = None,
    style: str = "info",
) -> Panel:
    """Create a standard panel with consistent styling.

    Args:
        content: Panel content
        title: Optional panel title
        style: Border style (info, success, warning, error)

    Returns:
        Configured Panel instance
    """
    return Panel(
        content,
        title=title,
        title_align="left",
        border_style=style,
        box=box.ROUNDED,
    )


def print_panel(
    content: str,
    title: str | None = None,
    style: str = "info",
) -> None:
    """Print a standard panel.

    Args:
        content: Panel content
        title: Optional panel title
        style: Border style (info, success, warning, error)
    """
    panel = create_panel(content, title, style)
    console.print(panel)
