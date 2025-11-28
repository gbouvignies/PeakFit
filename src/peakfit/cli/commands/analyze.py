"""Analyze subcommands for PeakFit CLI.

This module contains all analyze-related commands for uncertainty estimation.
It creates a Typer sub-application with commands for:
"""

from __future__ import annotations

from pathlib import Path  # Required at runtime by Typer  # noqa: TC003
from typing import Annotated

import typer  # Required at runtime by Typer

from peakfit.ui import info

# Create analyze sub-application
analyze_app = typer.Typer(
    help="Uncertainty analysis commands for PeakFit results",
    no_args_is_help=True,
)


# ==================== MCMC COMMAND ====================


@analyze_app.command("mcmc")
def analyze_mcmc(
    results: Annotated[
        Path,
        typer.Argument(
            help="Path to results directory from 'peakfit fit'",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ],
    peaks: Annotated[
        list[str] | None,
        typer.Option(
            "--peaks",
            help="Peak names to analyze (default: all)",
        ),
    ] = None,
    n_walkers: Annotated[
        int,
        typer.Option(
            "--walkers",
            "--chains",
            help="Number of MCMC walkers/chains",
            min=4,
        ),
    ] = 32,
    n_steps: Annotated[
        int,
        typer.Option(
            "--steps",
            "--samples",
            help="Number of MCMC steps/samples per walker",
            min=100,
        ),
    ] = 1000,
    burn_in: Annotated[
        int | None,
        typer.Option(
            "--burn-in",
            help="MCMC burn-in steps (manual override; default: auto-determined using R-hat)",
            min=0,
        ),
    ] = None,
    auto_burnin: Annotated[
        bool,
        typer.Option(
            "--auto-burnin/--no-auto-burnin",
            help="Automatically determine burn-in using R-hat convergence monitoring",
        ),
    ] = True,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file for results",
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
) -> None:
    """Run MCMC sampling for uncertainty estimation.

    MCMC sampling provides full posterior distributions for fitted parameters.

    Examples
    --------
        peakfit analyze mcmc Fits/
        peakfit analyze mcmc Fits/ --chains 64 --samples 2000
        peakfit analyze mcmc Fits/ --walkers 64 --steps 2000
        peakfit analyze mcmc Fits/ --peaks 2N-H --peaks 3N-H
    """
    from peakfit.cli.analyze_command import run_mcmc

    # Handle manual override: if --burn-in is specified, disable auto-burnin
    if burn_in is not None and auto_burnin:
        info("Manual burn-in specified; disabling auto-burnin")
        auto_burnin = False

    run_mcmc(
        results_dir=results,
        n_walkers=n_walkers,
        n_steps=n_steps,
        burn_in=burn_in,
        auto_burnin=auto_burnin,
        peaks=peaks,
        output_file=output,
        verbose=False,
    )


# ==================== PROFILE COMMAND ====================


@analyze_app.command("profile")
def analyze_profile(
    results: Annotated[
        Path,
        typer.Argument(
            help="Path to results directory from 'peakfit fit'",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ],
    param: Annotated[
        str | None,
        typer.Option(
            "--param",
            "-p",
            help="Parameter to profile: exact name, peak name, or parameter type (default: all)",
        ),
    ] = None,
    n_points: Annotated[
        int,
        typer.Option(
            "--points",
            help="Number of profile likelihood points",
            min=5,
        ),
    ] = 20,
    confidence: Annotated[
        float,
        typer.Option(
            "--confidence",
            help="Confidence level (0.68 or 0.95)",
            min=0.5,
            max=0.999,
        ),
    ] = 0.95,
    plot: Annotated[
        bool,
        typer.Option(
            "--plot/--no-plot",
            help="Plot profile likelihood curve",
        ),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file for results",
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
) -> None:
    """Compute profile likelihood confidence intervals.

    Profile likelihood gives accurate confidence intervals for parameters,
    especially when the likelihood surface is non-quadratic.

    Examples
    --------
        peakfit analyze profile Fits/                    # All parameters
        peakfit analyze profile Fits/ --param 2N-H       # All params for peak 2N-H
        peakfit analyze profile Fits/ --param x0         # All x0 parameters
        peakfit analyze profile Fits/ --param 2N-H_x0    # Specific parameter
    """
    from peakfit.cli.analyze_command import run_profile_likelihood

    run_profile_likelihood(
        results_dir=results,
        param_name=param,
        n_points=n_points,
        confidence_level=confidence,
        plot=plot,
        output_file=output,
        verbose=False,
    )


# ==================== UNCERTAINTY COMMAND ====================


@analyze_app.command("uncertainty")
def analyze_uncertainty(
    results: Annotated[
        Path,
        typer.Argument(
            help="Path to results directory from 'peakfit fit'",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file for uncertainty summary",
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
) -> None:
    """Display parameter uncertainties from fitting results.

    Shows the covariance-based uncertainties computed during fitting.
    This is a quick way to review uncertainties without running
    additional analysis.

    Examples
    --------
        peakfit analyze uncertainty Fits/
        peakfit analyze uncertainty Fits/ --output uncertainties.txt
    """
    from peakfit.cli.analyze_command import run_uncertainty

    run_uncertainty(
        results_dir=results,
        output_file=output,
        verbose=False,
    )
