"""Main Typer application for PeakFit.

This module provides a thin orchestration layer that:
1. Creates the main Typer application
2. Imports commands from the commands/ subpackage
3. Registers commands and sub-applications

Target: <150 LOC as per architecture refactoring plan.
"""

from typing import Annotated

import typer

from peakfit.cli.callbacks import version_callback
from peakfit.cli.commands import (
    analyze_command,
    benchmark_command,
    fit_command,
    info_command,
    init_command,
    plot_app,
    validate_command,
)

# Create main application
app = typer.Typer(
    name="peakfit",
    help="PeakFit - Lineshape fitting for pseudo-3D NMR spectra",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """PeakFit - Modern lineshape fitting for pseudo-3D NMR spectra.

    Perform automated peak integration and lineshape analysis on NMR data.
    """


# Register commands
app.command(name="fit")(fit_command)
app.command(name="validate")(validate_command)
app.command(name="init")(init_command)
app.command(name="info")(info_command)
app.command(name="analyze")(analyze_command)
app.command(name="benchmark")(benchmark_command)

# Register sub-applications
app.add_typer(plot_app, name="plot")
