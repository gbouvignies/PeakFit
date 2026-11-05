"""Fit command - main fitting workflow with mandatory validation."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from peakfit.cli.commands.fit_setup import build_fit_config, resolve_output_dir
from peakfit.engine.diagnostics.bounds import at_bound
from peakfit.engine.domain.config import LineshapeName  # noqa: TC001 - Typer evaluates annotations.
from peakfit.engine.results import HIGH_REDUCED_CHI_SQUARED, FitResult
from peakfit.fit.fitting import (
    ClusterReview,
    FitRun,
    ProgressStart,
    find_review_clusters,
    load_data,
    run_fit,
    write_fit_run_outputs,
)
from peakfit.fit.run_models import CorrectionUpdate, PassStart
from peakfit.fit.validation import validate_inputs
from peakfit.ui.branding import show_command_summary
from peakfit.ui.console import Verbosity, console, display_path, set_verbosity
from peakfit.ui.messages import bullet, error, show_error_with_details
from peakfit.ui.prefit import show_prefit_check
from peakfit.ui.reporter import ConsoleReporter
from peakfit.ui.views import display_post_fit_summary, live_fit_display

if TYPE_CHECKING:
    from peakfit.engine.domain.config import PeakFitConfig
    from peakfit.fit.fitting import LoadedData


def fit_command(
    spectrum: Annotated[
        Path,
        typer.Argument(
            help="NMRPipe spectrum file (.ft2, .ft3)",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    peaklist: Annotated[
        Path,
        typer.Argument(
            help="Peak list file (.list, .csv, .json, .xlsx, .xls)",
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
            help="Plane values file for a pseudo-ND spectrum",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output directory", file_okay=False, resolve_path=True),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="TOML config file",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    lineshape: Annotated[
        LineshapeName,
        typer.Option("--lineshape", "-l", help="Lineshape model"),
    ] = "auto",
    refine: Annotated[
        int,
        typer.Option(
            "--refine",
            "-r",
            help="Optimizer passes; corrections update between passes",
            min=1,
            max=20,
        ),
    ] = 2,
    contour_level: Annotated[
        float | None,
        typer.Option("--contour", "-t", help="Contour level (default: 5×noise)"),
    ] = None,
    noise: Annotated[
        float | None,
        typer.Option("--noise", "-n", help="Manual noise level"),
    ] = None,
    phx: Annotated[
        bool,
        typer.Option("--phx/--no-phx", help="Fit X phase"),
    ] = False,
    phy: Annotated[
        bool,
        typer.Option("--phy/--no-phy", help="Fit Y phase"),
    ] = False,
    exclude: Annotated[
        list[int] | None,
        typer.Option("--exclude", "-e", help="Exclude plane indices"),
    ] = None,
    formats: Annotated[
        list[str] | None,
        typer.Option("--format", "-f", help="Output formats: json, csv, txt"),
    ] = None,
    detailed_evidence: Annotated[
        bool,
        typer.Option(
            "--detailed-evidence",
            help="Include observations and predictions for portable review and continuation.",
        ),
    ] = False,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", help="Parallel workers (-1 = all CPUs)", min=-1),
    ] = -1,
    headless: Annotated[
        bool | None,
        typer.Option("--headless/--interactive", help="Disable live UI"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Verbose output"),
    ] = False,
) -> None:
    """Fit lineshapes to peaks in a pseudo-ND NMR spectrum.

    Runs validation automatically before fitting. If validation fails,
    the fit will not proceed.

    Examples:
        peakfit fit spectrum.ft2 peaks.list
        peakfit fit spectrum.ft2 peaks.list -z z_values.txt -o results/
        peakfit fit spectrum.ft2 peaks.list --config settings.toml
    """
    start_time = datetime.datetime.now(datetime.UTC)
    set_verbosity(Verbosity.VERBOSE if verbose else Verbosity.NORMAL)
    reporter = ConsoleReporter()

    validation = validate_inputs(spectrum, peaklist)
    if validation.errors:
        error("Validation failed. Fix the errors below before fitting.")
        for e in validation.errors:
            bullet(str(e), style="error")
        raise typer.Exit(1)

    fit_config = build_fit_config(
        config=config,
        output=output,
        formats=formats,
        headless=headless,
        noise=noise,
        contour_level=contour_level,
        lineshape=lineshape,
        refine=refine,
        phx=phx,
        phy=phy,
        exclude=exclude,
    )
    if detailed_evidence:
        fit_config.output.detailed_evidence = True

    if fit_config.output.headless:
        _show_headless_command_summary(
            spectrum=spectrum,
            peaklist=peaklist,
            z_values=z_values,
            fit_config=fit_config,
            workers=workers,
        )

    try:
        with console.status("[info]Loading data...[/info]", spinner="dots"):
            data = load_data(spectrum, peaklist, z_values, fit_config)
    except (OSError, ValueError) as e:
        show_error_with_details("Loading data", e)
        raise typer.Exit(1) from e

    output_dir = resolve_output_dir(fit_config)

    if not fit_config.output.headless:
        show_prefit_check(data, output_dir, fit_config, spectrum, peaklist, workers)

    try:
        if fit_config.output.headless:
            result = run_fit(
                data,
                fit_config,
                output_dir,
                workers=workers,
                reporter=reporter,
            )
        else:
            result = _run_interactive_fit(data, fit_config, output_dir, workers)

        duration = (datetime.datetime.now(datetime.UTC) - start_time).total_seconds()

        if not fit_config.output.headless:
            reviews = find_review_clusters(result.outcome)
            clusters_to_review = _format_review_clusters(reviews)
            display_post_fit_summary(
                total_time=duration,
                total_clusters=result.summary.n_clusters,
                successful_fits=result.summary.n_converged,
                usable_non_converged=result.summary.n_usable_non_converged,
                unusable=result.summary.n_unusable,
                chi_sq_stats={
                    "Mean": result.summary.mean_redchi,
                    "Std Dev": result.summary.std_redchi,
                    "Median": result.summary.median_redchi,
                },
                clusters_to_review=clusters_to_review,
                output_dir=display_path(result.output_dir),
            )

        with console.status("[info]Writing results...[/info]", spinner="dots"):
            input_paths = {"spectrum": spectrum, "peaklist": peaklist}
            if z_values is not None:
                input_paths["z_values"] = z_values
            write_fit_run_outputs(result, fit_config, input_paths, reporter)

    except Exception as e:
        show_error_with_details("Fitting", e)
        raise typer.Exit(1) from e


def _show_headless_command_summary(
    *,
    spectrum: Path,
    peaklist: Path,
    z_values: Path | None,
    fit_config: PeakFitConfig,
    workers: int,
) -> None:
    """Show the compact header used when the live pre-fit panel is disabled."""
    show_command_summary(
        "Fitting",
        sections=[
            (
                "Inputs",
                {
                    "Spectrum": display_path(spectrum),
                    "Peak list": display_path(peaklist),
                    "Plane values": display_path(z_values) if z_values is not None else "None",
                },
            ),
            (
                "Fitting",
                {
                    "Optimizer": "varpro",
                    "Lineshape": str(fit_config.fitting.lineshape),
                    "Optimizer passes": str(fit_config.fitting.refine_iterations),
                    "Workers": "All CPUs" if workers == -1 else str(workers),
                },
            ),
            (
                "Output",
                {
                    "Base directory": display_path(fit_config.output.directory or Path("./")),
                    "Formats": ", ".join(fit_config.output.formats),
                },
            ),
        ],
    )


def _run_interactive_fit(
    data: LoadedData,
    fit_config: PeakFitConfig,
    output_dir: Path,
    workers: int,
) -> FitRun:
    """Run fitting with interactive live display."""
    # Use a mutable container for the live display context
    live_ctx: dict[str, Any] = {"update": None}

    def progress_callback(item: Any) -> None:
        """Handle progress events from pipeline."""
        if isinstance(item, ProgressStart):
            # Start the live display with the pipeline total
            live_ctx["display"] = live_fit_display(total_steps=item.total_steps)
            live_ctx["update"] = live_ctx["display"].__enter__()
        elif isinstance(item, PassStart):
            if live_ctx["update"]:
                status = f"[bold blue]Iteration {item.iteration}/{item.iterations}[/]"
                live_ctx["update"](status_message=status, advance=0)
        elif isinstance(item, CorrectionUpdate):
            if live_ctx["update"]:
                live_ctx["update"](
                    status_message="[dim]Correcting data with neighbors...[/]", advance=0
                )
        elif isinstance(item, FitResult):
            if live_ctx["update"]:
                cluster_id = str(item.cluster_id)
                warnings = []
                bounds_hit = [p for p in item.params.values() if at_bound(p)]
                if bounds_hit:
                    warnings.append(f"Bounds: {', '.join(p.name for p in bounds_hit)}")
                if item.message and "Maximum iterations reached" in item.message:
                    warnings.append("Max iterations")

                live_ctx["update"](
                    cluster_id=cluster_id,
                    success=item.success,
                    chisqr=item.redchi,
                    fit_time=item.metadata.get("fit_time", 0.0),
                    message=item.message,
                    warnings=warnings,
                    advance=1,
                    peak_names=item.metadata.get("peak_names", []),
                )

    try:
        return run_fit(
            data,
            fit_config,
            output_dir,
            workers=workers,
            progress_callback=progress_callback,
        )
    finally:
        # Clean up the live display context
        if "display" in live_ctx:
            live_ctx["display"].__exit__(None, None, None)


# Threshold for formatting high chi-squared
_MAX_PEAK_LABELS = 3


def _format_review_clusters(reviews: list[ClusterReview]) -> list[dict[str, Any]]:
    """Format ClusterReview objects for display."""
    formatted = []
    for review in reviews:
        if review.reason == "unusable":
            status, status_color = "Unusable", "metric.bad"
            details = review.unusable_reason or "Numerical evaluation is unusable"
            if review.termination_message:
                details = f"{details}; optimizer: {review.termination_message}"
        elif review.reason == "not_converged":
            status, status_color = "Not converged", "metric.warn"
            details = review.termination_message or "Optimizer did not converge"
        elif review.reason == "uncertainty":
            status, status_color = "Unreliable uncertainties", "metric.warn"
            details = "Inspect overlapping parameters, bounds and alternative fits"
        elif review.reason == "high_chi":
            status, status_color = "High χ²", "metric.warn"
            assert review.redchi is not None
            details = f"Red. χ² = {review.redchi:.2f}"
        else:
            status, status_color = "At Bounds", "metric.warn"
            details = ", ".join(review.at_bounds)

        label = (
            f"{review.peak_names[0]}..."
            if len(review.peak_names) > _MAX_PEAK_LABELS
            else ", ".join(review.peak_names)
            if review.peak_names
            else review.cluster_id
        )

        formatted.append(
            {
                "id": review.cluster_id,
                "label": label,
                "status": status,
                "status_color": status_color,
                "chi_sq": review.redchi,
                "chi_sq_color": (
                    "metric.bad"
                    if review.redchi is not None and review.redchi > HIGH_REDUCED_CHI_SQUARED
                    else "metric.warn"
                ),
                "details": details,
            }
        )

    return formatted
