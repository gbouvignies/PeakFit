"""Typer callbacks for CLI."""

from typing import Optional

import typer

from peakfit import __version__


def version_callback(value: Optional[bool]) -> None:
    """Show version information and exit."""
    if value:
        from rich.console import Console

        console = Console()
        console.print(f"[bold]PeakFit[/bold] version {__version__}")
        raise typer.Exit()
