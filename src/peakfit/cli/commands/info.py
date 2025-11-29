"""Info command implementation."""

from __future__ import annotations

from typing import Annotated

import typer  # Required at runtime by Typer

from peakfit.ui import console


def info_command(
    benchmark: Annotated[
        bool,
        typer.Option(
            "--benchmark",
            help="Run performance benchmark (deprecated)",
        ),
    ] = False,
) -> None:
    """Show system information.

    Display details about the PeakFit installation and system capabilities.
    """
    import sys

    import numpy as np

    from peakfit import __version__

    console.print("[bold]PeakFit System Information[/bold]\n")

    # Version info
    console.print(f"[green]PeakFit version:[/green] {__version__}")
    console.print(f"[green]Python version:[/green] {sys.version}")
    console.print(f"[green]NumPy version:[/green] {np.__version__}")

    # Parallelization removed
    console.print("\n[green]Parallelization:[/green] Disabled (single-threaded execution)")

    # Note about backends
    console.print(
        "\n[dim]Note: PeakFit uses NumPy for computations and focuses on optimized, maintainable implementations.[/dim]"
    )

    # Benchmark
    if benchmark:
        console.print("\n[yellow]Benchmark option is deprecated and has been removed.[/yellow]")
