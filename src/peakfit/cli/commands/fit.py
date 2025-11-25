"""Fit command implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from peakfit.core.domain.config import (
    ClusterConfig,
    FitConfig,
    LineshapeName,
    OutputConfig,
    PeakFitConfig,
)
from peakfit.io.config import load_config


def fit_command(
    spectrum: Annotated[
        Path,
        typer.Argument(
            help="Path to NMRPipe spectrum file (.ft2, .ft3)",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    peaklist: Annotated[
        Path,
        typer.Argument(
            help="Path to peak list file (.list, .csv, .json, .xlsx)",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    z_values: Annotated[
        Path | None,
        typer.Option(
            "--z-values",
            "-z",
            help="Path to Z-dimension values file",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for results",
            file_okay=False,
            resolve_path=True,
        ),
    ] = Path("Fits"),
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to TOML configuration file",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    lineshape: Annotated[
        LineshapeName,
        typer.Option(
            "--lineshape",
            "-l",
            help="Lineshape model: auto, gaussian, lorentzian, pvoigt, sp1, sp2, no_apod",
        ),
    ] = "auto",
    refine: Annotated[
        int,
        typer.Option(
            "--refine",
            "-r",
            help="Number of refinement iterations",
            min=0,
            max=20,
        ),
    ] = 1,
    contour_level: Annotated[
        float | None,
        typer.Option(
            "--contour",
            "-t",
            help="Contour level for segmentation (default: 5 * noise)",
        ),
    ] = None,
    noise: Annotated[
        float | None,
        typer.Option(
            "--noise",
            "-n",
            help="Manual noise level (auto-detected if not specified)",
        ),
    ] = None,
    fixed: Annotated[
        bool,
        typer.Option(
            "--fixed/--no-fixed",
            help="Fix peak positions during fitting",
        ),
    ] = False,
    jx: Annotated[
        bool,
        typer.Option(
            "--jx/--no-jx",
            help="Fit J-coupling constant",
        ),
    ] = False,
    phx: Annotated[
        bool,
        typer.Option(
            "--phx/--no-phx",
            help="Fit phase correction in X",
        ),
    ] = False,
    phy: Annotated[
        bool,
        typer.Option(
            "--phy/--no-phy",
            help="Fit phase correction in Y",
        ),
    ] = False,
    exclude: Annotated[
        list[int] | None,
        typer.Option(
            "--exclude",
            "-e",
            help="Plane indices to exclude (can be specified multiple times)",
        ),
    ] = None,
    optimizer: Annotated[
        str,
        typer.Option(
            "--optimizer",
            help="Optimization algorithm: leastsq (fast), basin-hopping (global), "
            "differential-evolution (global)",
        ),
    ] = "leastsq",
    save_state: Annotated[
        bool,
        typer.Option(
            "--save-state/--no-save-state",
            help="Save fitting state for later analysis (enables 'peakfit analyze' command)",
        ),
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Show banner and verbose output",
        ),
    ] = False,
) -> None:
    """Fit lineshapes to peaks in pseudo-3D NMR spectrum.

    Example:
        peakfit fit spectrum.ft2 peaks.list --output results --refine 2

    For difficult peaks with overlaps, use global optimization:
        peakfit fit spectrum.ft2 peaks.list --optimizer basin-hopping

    To perform uncertainty analysis later:
        peakfit fit spectrum.ft2 peaks.list --save-state
        peakfit analyze mcmc Fits/  # Compute MCMC uncertainties
    """
    from peakfit.cli.fit_command import run_fit

    # Load config from file or create from CLI options
    if config is not None:
        fit_config = load_config(config)
        # Override with CLI options where explicitly set
        fit_config.output.directory = output
    else:
        fit_config = PeakFitConfig(
            fitting=FitConfig(
                lineshape=lineshape,
                refine_iterations=refine,
                fix_positions=fixed,
                fit_j_coupling=jx,
                fit_phase_x=phx,
                fit_phase_y=phy,
            ),
            clustering=ClusterConfig(contour_level=contour_level),
            output=OutputConfig(directory=output),
            noise_level=noise,
            exclude_planes=exclude or [],
        )

    run_fit(
        spectrum_path=spectrum,
        peaklist_path=peaklist,
        z_values_path=z_values,
        config=fit_config,
        optimizer=optimizer,
        save_state=save_state,
        verbose=verbose,
    )
