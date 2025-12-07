"""UI progress indicators.

This module provides functions for creating progress bars
with consistent styling across the application.
"""

from __future__ import annotations

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .console import console, icon

__all__ = [
    "create_progress",
]


def create_progress(transient: bool = False) -> Progress:
    """Create a standard progress bar with consistent styling.

    Args:
        transient: Whether the progress bar should disappear when complete

    Returns
    -------
        Configured Progress instance
    """
    return Progress(
        SpinnerColumn(finished_text=f"[success]{icon('check')}[/success]", spinner_name="dots"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="cyan", finished_style="green"),
        TaskProgressColumn(show_speed=True),
        MofNCompleteColumn(),
        TextColumn("[dim]•[/dim]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=transient,
    )
