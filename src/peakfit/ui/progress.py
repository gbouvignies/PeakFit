"""UI progress indicators.

This module provides functions for creating progress bars
with consistent styling across the application.
"""

from __future__ import annotations

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from .console import console

__all__ = [
    "create_progress",
]


def create_progress() -> Progress:
    """Create a standard progress bar with consistent styling.

    Returns:
        Configured Progress instance
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="cyan", finished_style="green"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    )
