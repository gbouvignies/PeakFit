"""Typer callbacks for CLI."""

import typer

from peakfit.ui.branding import show_version


def version_callback(value: bool | None) -> None:
    """Show version information and exit."""
    if value:
        show_version()
        raise typer.Exit
