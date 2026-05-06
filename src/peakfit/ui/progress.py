"""UI progress indicators.

This module provides functions for creating progress bars
with consistent styling across the application.
"""

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
    "create_mcmc_progress",
    "create_progress",
]


def create_progress(transient: bool = False) -> Progress:
    """Create a standard progress bar with consistent styling.

    Args:
        transient: Whether the progress bar should disappear when complete

    Returns:
    -------
        Configured Progress instance
    """
    return Progress(
        SpinnerColumn(finished_text=f"[success]{icon('check')}[/success]", spinner_name="dots"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="progress.percentage", finished_style="success"),
        TaskProgressColumn(show_speed=True),
        MofNCompleteColumn(),
        TextColumn("[neutral]•[/neutral]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=transient,
    )


def create_mcmc_progress(transient: bool = False) -> Progress:
    """Create a specialized progress bar for MCMC sampling.

    Includes columns for acceptance rate and step tracking.
    """
    return Progress(
        SpinnerColumn(finished_text=f"[success]{icon('check')}[/success]", spinner_name="dots"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="progress.percentage", finished_style="success"),
        TaskProgressColumn(),
        TextColumn("[neutral]•[/neutral]"),
        MofNCompleteColumn(),
        TextColumn("[neutral]•[/neutral]"),
        TextColumn("{task.fields[stats]}"),
        TextColumn("[neutral]•[/neutral]"),
        TimeRemainingColumn(),
        console=console,
        transient=transient,
    )
