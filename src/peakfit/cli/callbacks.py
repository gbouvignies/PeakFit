"""Typer callbacks for CLI."""

import typer

from peakfit.ui import PeakFitUI


def version_callback(value: bool | None) -> None:
    """Show version information and exit."""
    if value:
        PeakFitUI.show_version()
        raise typer.Exit
