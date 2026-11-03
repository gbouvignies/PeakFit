"""Plot subcommands for PeakFit CLI."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - needed at runtime for Typer
from typing import TYPE_CHECKING, Annotated

import typer

from peakfit.io.readers.results import ResultsLoader
from peakfit.plot.outputs import (
    PlotOutput,
    generate_cest_plots,
    generate_cpmg_plots,
    generate_intensity_plots,
    generate_mcmc_diagnostics,
)
from peakfit.ui.branding import show_command_summary
from peakfit.ui.console import Verbosity, display_path, set_verbosity
from peakfit.ui.messages import show_error_with_details, success, warning
from peakfit.ui.reporter import ConsoleReporter

if TYPE_CHECKING:
    from collections.abc import Callable


_MAX_REF_POINTS_SHOWN = 6
# Create plot sub-application
plot_app = typer.Typer(
    help="Plotting commands for PeakFit results",
    no_args_is_help=True,
)


def _configure_plot_ui(
    verbose: bool,
    title: str,
    sections: list[tuple[str, dict[str, str]]],
) -> None:
    """Configure terminal verbosity and print a compact command summary."""
    set_verbosity(Verbosity.VERBOSE if verbose else Verbosity.NORMAL)
    show_command_summary(title, sections)


def _print_plot_success(out: PlotOutput, label: str) -> None:
    """Print a consistent success message for generated plot artifacts."""
    success(f"Saved {out.n_plots} {label} plot(s) to [path]{display_path(out.path)}[/path]")


def _run_plot_generation(
    label: str,
    error_context: str,
    generate: Callable[[], PlotOutput],
) -> None:
    """Run a plot generator with consistent success and error handling."""
    try:
        out = generate()
        _print_plot_success(out, label)
    except Exception as e:
        show_error_with_details(error_context, e)
        raise typer.Exit(1) from e


def _format_bool(flag: bool) -> str:
    """Format boolean values as concise Yes/No labels."""
    return "Yes" if flag else "No"


def _format_reference_indices(ref: list[int] | None) -> str:
    """Format CEST reference point indices for compact display."""
    if not ref:
        return "Auto"
    if len(ref) <= _MAX_REF_POINTS_SHOWN:
        return ", ".join(str(i) for i in ref)
    shown = ", ".join(str(i) for i in ref[:_MAX_REF_POINTS_SHOWN])
    return f"{shown}, ... ({len(ref)} total)"


@plot_app.command("cest")
def plot_cest(
    results: Annotated[
        Path,
        typer.Argument(help="Results directory", exists=True, resolve_path=True),
    ],
    ref: Annotated[
        list[int] | None,
        typer.Option("--ref", "-r", help="Reference point indices; omit for auto"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output PDF file", dir_okay=False, resolve_path=True),
    ] = None,
    show: Annotated[
        bool,
        typer.Option("--show/--no-show", help="Display interactively"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Verbose output"),
    ] = False,
) -> None:
    """Plot CEST profiles as normalized intensity vs B1 offset."""
    output_path = output or (results / "cest_profiles.pdf")
    _configure_plot_ui(
        verbose,
        "Plot CEST Profiles",
        sections=[
            (
                "Inputs",
                {
                    "Results": display_path(results),
                    "Reference points": _format_reference_indices(ref),
                },
            ),
            (
                "Output",
                {
                    "PDF": display_path(output_path),
                    "Show interactively": _format_bool(show),
                },
            ),
        ],
    )
    _run_plot_generation(
        "CEST",
        "generating CEST plots",
        lambda: generate_cest_plots(
            results,
            output_path=output_path,
            reference_indices=ref,
            show=show,
            reporter=ConsoleReporter(),
        ),
    )


@plot_app.command("intensity")
def plot_intensity(
    results: Annotated[
        Path,
        typer.Argument(help="Results directory", exists=True, resolve_path=True),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output PDF file", dir_okay=False, resolve_path=True),
    ] = None,
    show: Annotated[
        bool,
        typer.Option("--show/--no-show", help="Display interactively"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Verbose output"),
    ] = False,
) -> None:
    """Plot intensity profiles vs plane or Z-value.

    Examples:
        peakfit plot intensity Fits/20240101_120000/
        peakfit plot intensity results/ -o profiles.pdf
    """
    output_path = output or (results / "intensity_profiles.pdf")
    _configure_plot_ui(
        verbose,
        "Plot Intensity Profiles",
        sections=[
            ("Inputs", {"Results": display_path(results)}),
            (
                "Output",
                {
                    "PDF": display_path(output_path),
                    "Show interactively": _format_bool(show),
                },
            ),
        ],
    )
    _run_plot_generation(
        "intensity",
        "generating plots",
        lambda: generate_intensity_plots(
            results,
            output_path=output_path,
            show=show,
            reporter=ConsoleReporter(),
        ),
    )


@plot_app.command("cpmg")
def plot_cpmg(
    results: Annotated[
        Path,
        typer.Argument(help="Results directory", exists=True, resolve_path=True),
    ],
    time_t2: Annotated[
        float,
        typer.Option("--time-t2", "-t", help="T2 relaxation time in seconds"),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output PDF", dir_okay=False, resolve_path=True),
    ] = None,
    show: Annotated[
        bool,
        typer.Option("--show/--no-show", help="Display interactively"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Verbose output"),
    ] = False,
) -> None:
    """Plot CPMG relaxation dispersion as R2eff vs nuCPMG."""
    output_path = output or (results / "cpmg_profiles.pdf")
    _configure_plot_ui(
        verbose,
        "Plot CPMG Profiles",
        sections=[
            (
                "Inputs",
                {
                    "Results": display_path(results),
                    "T2 time (s)": f"{time_t2:g}",
                },
            ),
            (
                "Output",
                {
                    "PDF": display_path(output_path),
                    "Show interactively": _format_bool(show),
                },
            ),
        ],
    )
    _run_plot_generation(
        "CPMG",
        "generating CPMG plots",
        lambda: generate_cpmg_plots(
            results,
            time_t2=time_t2,
            output_path=output_path,
            show=show,
            reporter=ConsoleReporter(),
        ),
    )


@plot_app.command("mcmc")
def plot_mcmc(
    results: Annotated[
        Path,
        typer.Argument(help="Results directory", exists=True, resolve_path=True),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output PDF", dir_okay=False, resolve_path=True),
    ] = None,
    burn_in: Annotated[
        int,
        typer.Option("--burn-in", "-b", help="Burn-in samples"),
    ] = 0,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Verbose output"),
    ] = False,
) -> None:
    """Plot MCMC diagnostic traces and corner plots.

    Examples:
        peakfit plot mcmc Fits/20240101_120000/
        peakfit plot mcmc results/ --burn-in 100
    """
    output_path = output or (results / "mcmc_diagnostics.pdf")
    _configure_plot_ui(
        verbose,
        "Plot MCMC Diagnostics",
        sections=[
            (
                "Inputs",
                {
                    "Results": display_path(results),
                    "Burn-in samples": str(burn_in),
                },
            ),
            ("Output", {"PDF": display_path(output_path)}),
        ],
    )

    try:
        loader = ResultsLoader(results)
        chains = loader.load_mcmc_chains()

        if not chains:
            warning(
                f"No MCMC chains found in [path]{display_path(results)}[/path]. "
                "Run [code]peakfit mcmc[/code] first."
            )
            raise typer.Exit(1)

        out = generate_mcmc_diagnostics(chains, output_path=output_path, burn_in=burn_in)
        _print_plot_success(out, "MCMC diagnostic")
    except typer.Exit:
        raise
    except Exception as e:
        show_error_with_details("generating MCMC diagnostics", e)
        raise typer.Exit(1) from e
