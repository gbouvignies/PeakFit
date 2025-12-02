"""Fit command implementation."""

from __future__ import annotations

import pathlib  # Required at runtime by Typer
from typing import Annotated, cast, get_args

import typer

import peakfit.core.domain.config as domain_config  # Required at runtime by Typer  # noqa: TC001
from peakfit.core.domain.config import (
    ClusterConfig,
    FitConfig,
    OutputConfig,
    OutputFormat,
    PeakFitConfig,
)
from peakfit.io.config import load_config

# Valid output formats for CLI validation
VALID_OUTPUT_FORMATS = get_args(OutputFormat)  # ("csv", "json", "txt")


def fit_command(
    spectrum: Annotated[
        pathlib.Path,
        typer.Argument(
            help="Path to NMRPipe spectrum file (.ft2, .ft3)",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    peaklist: Annotated[
        pathlib.Path,
        typer.Argument(
            help="Path to peak list file (.list, .csv, .json, .xlsx)",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    z_values: Annotated[
        pathlib.Path | None,
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
        pathlib.Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for results",
            file_okay=False,
            resolve_path=True,
        ),
    ] = None,
    config: Annotated[
        pathlib.Path | None,
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
        domain_config.LineshapeName,
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
    formats: Annotated[
        list[str] | None,
        typer.Option(
            "--format",
            "-f",
            help="Output format(s): json, csv, txt. Can be specified multiple times. "
            "Default: json,csv,txt (all formats)",
        ),
    ] = None,
    output_verbosity: Annotated[
        domain_config.OutputVerbosity,
        typer.Option(
            "--output-verbosity",
            help="Output verbosity level: minimal (essential), standard (default), full (all)",
        ),
    ] = "standard",
    include_legacy: Annotated[
        bool,
        typer.Option(
            "--legacy/--no-legacy",
            help="Generate legacy .out files in legacy/ subdirectory (for backward compatibility)",
        ),
    ] = False,
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            "-w",
            help="Number of parallel workers for cluster fitting. "
            "Use -1 for all CPUs, 1 for sequential (default: -1)",
            min=-1,
        ),
    ] = -1,
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
    from peakfit.services.fit.pipeline import FitPipeline

    # Validate and process format options
    if formats:
        invalid_formats = [f for f in formats if f not in VALID_OUTPUT_FORMATS]
        if invalid_formats:
            msg = f"Invalid format(s): {', '.join(invalid_formats)}. Valid formats: {', '.join(VALID_OUTPUT_FORMATS)}"
            raise typer.BadParameter(msg)
        output_formats = cast("list[OutputFormat]", formats)  # Type-safe after validation
    else:
        # Default formats: generate all outputs (JSON, CSV, and legacy txt)
        # Ensure proper typing for mypy: cast to list[OutputFormat]
        output_formats = cast("list[OutputFormat]", ["json", "csv", "txt"])

    # Load config from file or create from CLI options
    if config is not None:
        fit_config = load_config(config)
        # Override with CLI options only where explicitly set
        if output is not None:
            fit_config.output.directory = output
        # Only override formats if explicitly provided via --format
        if formats is not None:
            fit_config.output.formats = output_formats
        # Note: verbosity and include_legacy are always applied from CLI
        # since there's no way to detect if they were explicitly set
        # (they have non-None defaults). Users should set these in config.
        fit_config.output.verbosity = output_verbosity
        fit_config.output.include_legacy = include_legacy
    else:
        # Build output config - use CLI output if provided, otherwise use default
        output_config = OutputConfig(
            formats=output_formats,
            verbosity=output_verbosity,
            include_legacy=include_legacy,
        )
        if output is not None:
            output_config.directory = output

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
            output=output_config,
            noise_level=noise,
            exclude_planes=exclude or [],
        )

    FitPipeline.run(
        spectrum_path=spectrum,
        peaklist_path=peaklist,
        z_values_path=z_values,
        config=fit_config,
        optimizer=optimizer,
        save_state=save_state,
        verbose=verbose,
        workers=workers,
    )
