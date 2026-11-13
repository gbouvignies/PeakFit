"""The replicates command: check a CEST run's profile σ against its baseline planes."""

from __future__ import annotations

from collections import Counter
from pathlib import Path  # noqa: TC003 - needed at runtime for Typer
from typing import Annotated

import numpy as np
import typer

from peakfit.engine.scientific.replicates import (
    BASELINE_HZ,
    BaselineScatter,
    run_baseline_scatter,
)
from peakfit.io.scientific import read_scientific
from peakfit.ui.branding import show_command_summary
from peakfit.ui.console import Verbosity, console, display_path, set_verbosity
from peakfit.ui.messages import info, show_error_with_details, success, warning
from peakfit.ui.tables import create_table


def replicates_command(
    results: Annotated[
        Path,
        typer.Argument(help="Results directory", exists=True, file_okay=False, resolve_path=True),
    ],
    ref: Annotated[
        list[int] | None,
        typer.Option(
            "--ref",
            "-r",
            help="Reference point indices, in offset order; omit for |offset| ≥ 10 kHz",
        ),
    ] = None,
    window: Annotated[
        float,
        typer.Option(
            "--window", help="Planes closer than this to a peak's own offset (Hz) are left out"
        ),
    ] = BASELINE_HZ,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="List every peak"),
    ] = False,
) -> None:
    """Check a CEST run's profile σ against the scatter of its baseline planes.

    Far from a peak's own offset, CEST planes are approximate replicates. The robust
    scatter of their second differences across offsets is compared with the published
    profile σ: observed / stated is 1 when the errors are calibrated. The run is read
    from scientific.npz, never refitted. Plane values must be offsets in Hz from the
    carrier of the first spectral axis.

    This measures repeatability, not accuracy: an error common to every baseline plane,
    such as a lineshape misfit, is invisible. Drift and minor-state dips can only
    raise the observed scatter.

    Examples:
        peakfit replicates Fits/20240101_120000/
    """
    set_verbosity(Verbosity.VERBOSE if verbose else Verbosity.NORMAL)
    show_command_summary(
        "Check Profile σ Against CEST Baseline Replicates",
        [
            (
                "Inputs",
                {
                    "Results": display_path(results),
                    "Reference points": ", ".join(str(i) for i in ref)
                    if ref
                    else "|offset| ≥ 10 kHz",
                    "Peak window": f"±{window:g} Hz",
                },
            )
        ],
    )
    archive = results / "scientific.npz"
    try:
        if not archive.is_file():
            raise FileNotFoundError(
                f"No scientific.npz in {results}: the check reads the archive PeakFit "
                "writes with every fit. Refit to produce one."
            )
        outcome, _samples, context = read_scientific(archive)
        if not context:
            raise ValueError(f"{archive} records no plane values or spectral axes.")
        peaks = run_baseline_scatter(outcome, context, reference=ref, window=window)
    except Exception as e:
        show_error_with_details("checking baseline replicates", e)
        raise typer.Exit(1) from e

    checked = {name: r for name, r in peaks.items() if isinstance(r, BaselineScatter)}
    if verbose and checked:
        table = create_table("Baseline replicates")
        for column in ("Peak", "Planes", "Relative scatter", "Observed / stated"):
            table.add_column(column, justify="left" if column == "Peak" else "right")
        for name, r in sorted(checked.items()):
            table.add_row(name, str(r.planes), f"{r.observed / r.level:.4f}", f"{r.ratio:.2f}")
        console.print(table)
    for reason, count in sorted(Counter(r for r in peaks.values() if isinstance(r, str)).items()):
        warning(f"Skipped {count} peak{'s' if count > 1 else ''}: {reason}")
    if not checked:
        warning("No peak has enough baseline planes to check.")
        raise typer.Exit(1)
    ratios = np.array([r.ratio for r in checked.values()])
    low, high = np.percentile(ratios, [10, 90])
    scatter = np.median([r.observed / r.level for r in checked.values()])
    info(f"Relative baseline scatter: median {scatter:.4f}")
    success(
        f"Observed / stated profile σ: median {np.median(ratios):.2f} "
        f"(10–90%: {low:.2f}–{high:.2f}), {ratios.size} peaks"
    )
