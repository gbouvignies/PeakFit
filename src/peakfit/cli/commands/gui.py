"""Launch the desktop application without importing Qt during ordinary CLI use."""

from pathlib import Path  # noqa: TC003 - Typer resolves annotations at runtime
from typing import Annotated

import typer


def gui_command(
    spectrum: Annotated[
        Path | None,
        typer.Argument(exists=True, dir_okay=False, help="2D or pseudo-ND NMRPipe spectrum"),
    ] = None,
    peaks: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False, help="Existing peak list to display")
    ] = None,
    z_values: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Plane values file (one value per plane)"),
    ] = None,
) -> None:
    """Open the responsive spectrum viewer, or start with an empty window."""
    if spectrum is None and (peaks is not None or z_values is not None):
        raise typer.BadParameter("Provide a spectrum when using --peaks or --z-values.")
    from peakfit.gui.app import launch  # noqa: PLC0415

    try:
        launch(spectrum, peaks, z_values)
    except (OSError, ValueError, KeyError) as exc:
        typer.echo(f"Could not open spectrum: {exc}", err=True)
        raise typer.Exit(1) from exc
