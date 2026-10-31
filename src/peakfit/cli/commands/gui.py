"""Launch the desktop application without importing Qt during ordinary CLI use."""

from enum import StrEnum
from pathlib import Path  # noqa: TC003 - Typer resolves annotations at runtime
from typing import Annotated

import typer


class PeakAxisOrder(StrEnum):
    """Spectral axes of a peak list's first and second coordinate columns."""

    INDIRECT_DIRECT = "indirect-direct"
    DIRECT_INDIRECT = "direct-indirect"


def gui_command(
    path: Annotated[
        Path | None,
        typer.Argument(
            exists=True,
            help="2D or pseudo-ND NMRPipe spectrum, or a saved .peakfit analysis",
        ),
    ] = None,
    peaks: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Peak list to import with the spectrum"),
    ] = None,
    z_values: Annotated[
        Path | None,
        typer.Option(
            "--z-values",
            "-z",
            exists=True,
            dir_okay=False,
            help="Plane values file (one value per plane)",
        ),
    ] = None,
    peak_axis_order: Annotated[
        PeakAxisOrder | None,
        typer.Option(
            help="Spectral axes of the peak list's two coordinate columns; required with --peaks"
        ),
    ] = None,
) -> None:
    """Open the PeakFit desktop application, optionally with a dataset or saved analysis."""
    companions = peaks is not None or z_values is not None or peak_axis_order is not None
    if path is None and companions:
        raise typer.BadParameter(
            "Provide a spectrum when using --peaks, --z-values or --peak-axis-order."
        )
    if path is not None and path.suffix == ".peakfit":
        if not path.is_dir():
            raise typer.BadParameter(f"{path} is not a .peakfit analysis directory.")
        if companions:
            raise typer.BadParameter(
                "A saved .peakfit analysis records its own inputs; "
                "do not combine it with --peaks, --z-values or --peak-axis-order."
            )
    elif path is not None and path.is_dir():
        raise typer.BadParameter(f"{path} is a directory, not an NMRPipe spectrum file.")
    if peaks is not None and peak_axis_order is None:
        raise typer.BadParameter(
            "--peaks requires --peak-axis-order; confirm the coordinate order from your "
            "peak list's metadata. No axes are guessed."
        )
    if peaks is None and peak_axis_order is not None:
        raise typer.BadParameter("--peak-axis-order applies only with --peaks.")
    from peakfit.gui.editor.window import launch  # noqa: PLC0415

    launch(
        path,
        peaks,
        z_values,
        swap_peak_axes=peak_axis_order is PeakAxisOrder.DIRECT_INDIRECT,
    )
