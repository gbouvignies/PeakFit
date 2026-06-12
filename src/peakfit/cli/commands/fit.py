"""Fit command - main fitting workflow with mandatory validation."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast, get_args

import typer

from peakfit.engine.domain.config import (
    ClusterConfig,
    FitConfig,
    LineshapeName,
    OutputConfig,
    OutputFormat,
    PeakFitConfig,
)
from peakfit.engine.results import FitResult
from peakfit.fit.fitting import (
    ClusterReview,
    ProgressStart,
    ServiceResult,
    find_review_clusters,
    load_data,
    run_fit,
    write_service_results,
)
from peakfit.fit.validation import validate_inputs
from peakfit.io.config import load_config
from peakfit.io.paths import resolve_output_path
from peakfit.shared.constants import BASIN_HOPPING_NITER, DIFF_EVOLUTION_MAXITER
from peakfit.ui.branding import show_command_summary
from peakfit.ui.console import Verbosity, console, display_path, set_verbosity
from peakfit.ui.messages import bullet, error, show_error_with_details
from peakfit.ui.prefit import show_prefit_check
from peakfit.ui.reporter import ConsoleReporter
from peakfit.ui.views import display_post_fit_summary, live_fit_display

if TYPE_CHECKING:
    from peakfit.engine.domain.peaks import Peak
    from peakfit.fit.fitting import LoadedData
    from peakfit.shared.reporter import Reporter
    from peakfit.ui.auto_pick_stepper import AutoPickStepController

VALID_OUTPUT_FORMATS = get_args(OutputFormat)


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
        Path | None,
        typer.Argument(
            help="Peak list file (.list, .csv). Omit to enable automatic peak picking.",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    z_values: Annotated[
        Path | None,
        typer.Option(
            "--z-values",
            "-z",
            help="Z-dimension values file",
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
        typer.Option("--refine", "-r", help="Refinement iterations", min=0, max=20),
    ] = 2,
    contour_level: Annotated[
        float | None,
        typer.Option("--contour", "-t", help="Contour level (default: 5×noise)"),
    ] = None,
    noise: Annotated[
        float | None,
        typer.Option("--noise", "-n", help="Manual noise level"),
    ] = None,
    fixed: Annotated[
        bool,
        typer.Option("--fixed/--no-fixed", help="Fix peak positions"),
    ] = False,
    jx: Annotated[
        bool,
        typer.Option("--jx/--no-jx", help="Fit J-coupling"),
    ] = False,
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
    optimizer: Annotated[
        str,
        typer.Option("--optimizer", help="Optimizer: varpro, basin-hopping"),
    ] = "varpro",
    formats: Annotated[
        list[str] | None,
        typer.Option("--format", "-f", help="Output formats: json, csv, txt"),
    ] = None,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", help="Parallel workers (-1 = all CPUs)", min=-1),
    ] = -1,
    headless: Annotated[
        bool | None,
        typer.Option("--headless/--interactive", help="Disable live UI"),
    ] = None,
    auto_pick_step: Annotated[
        bool,
        typer.Option(
            "--auto-pick-step/--no-auto-pick-step",
            help=(
                "When no peak list is provided, open an interactive GUI to manually "
                "add/remove peaks per ROI and jump to the next cluster."
            ),
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Verbose output"),
    ] = False,
) -> None:
    """Fit lineshapes to peaks in a pseudo-3D NMR spectrum.

    Runs validation automatically before fitting. If validation fails,
    the fit will not proceed.

    Examples:
        peakfit fit spectrum.ft2
        peakfit fit spectrum.ft2 peaks.list
        peakfit fit spectrum.ft2 --auto-pick-step
        peakfit fit spectrum.ft2 peaks.list -z z_values.txt -o results/
        peakfit fit spectrum.ft2 peaks.list --config settings.toml
    """
    start_time = datetime.datetime.now()
    set_verbosity(Verbosity.VERBOSE if verbose else Verbosity.NORMAL)
    reporter = ConsoleReporter()

    # === 1. MANDATORY VALIDATION ===
    validation = validate_inputs(spectrum, peaklist)
    if validation.errors:
        error("Validation failed. Fix the errors below before fitting.")
        for e in validation.errors:
            bullet(str(e), style="error")
        raise typer.Exit(1)

    # === 2. BUILD CONFIG ===
    fit_config = _build_config(
        config=config,
        output=output,
        formats=formats,
        headless=headless,
        noise=noise,
        contour_level=contour_level,
        lineshape=lineshape,
        refine=refine,
        fixed=fixed,
        jx=jx,
        phx=phx,
        phy=phy,
        exclude=exclude,
        optimizer=optimizer,
    )

    # Interactive fits show the richer pre-fit setup panel, so avoid a redundant header.
    if fit_config.output.headless:
        show_command_summary(
            "Fitting",
            sections=[
                (
                    "Inputs",
                    {
                        "Spectrum": display_path(spectrum),
                        "Peak list": display_path(peaklist) if peaklist is not None else "Auto",
                        "Z values": display_path(z_values) if z_values is not None else "None",
                    },
                ),
                (
                    "Fitting",
                    {
                        "Optimizer": optimizer,
                        "Lineshape": str(fit_config.fitting.lineshape),
                        "Refine iterations": str(fit_config.fitting.refine_iterations),
                        "Auto-pick step mode": "Yes" if auto_pick_step else "No",
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

    # === 3. LOAD DATA ===
    data = _load_fit_data(
        spectrum=spectrum,
        peaklist=peaklist,
        z_values=z_values,
        fit_config=fit_config,
        auto_pick_step=auto_pick_step,
        reporter=reporter,
    )

    # === 4. RESOLVE OUTPUT ===
    output_dir = _resolve_output(fit_config)

    if not fit_config.output.headless:
        show_prefit_check(data, output_dir, optimizer, fit_config, spectrum, peaklist, workers)

    # === 5. FIT ===
    try:
        if fit_config.output.headless:
            # Headless mode: use reporter for progress
            result = run_fit(
                data,
                fit_config,
                output_dir,
                optimizer=optimizer,
                workers=workers,
                reporter=reporter,
            )
        else:
            # Interactive mode: use live display with callback
            result = _run_interactive_fit(data, fit_config, output_dir, optimizer, workers)

        duration = (datetime.datetime.now() - start_time).total_seconds()

        # === 6. POST-FIT SUMMARY ===
        if not fit_config.output.headless:
            reviews = find_review_clusters(result)
            clusters_to_review = _format_review_clusters(reviews)
            display_post_fit_summary(
                total_time=duration,
                total_clusters=len(result.results),
                successful_fits=result.summary.n_converged,
                chi_sq_stats={
                    "Mean": result.summary.mean_redchi,
                    "Std Dev": result.summary.std_redchi,
                    "Median": result.summary.median_redchi,
                },
                clusters_to_review=clusters_to_review,
                output_dir=display_path(result.output_dir),
            )

        # === 7. WRITE OUTPUTS ===
        with console.status("[info]Writing results...[/info]", spinner="dots"):
            input_paths = {"spectrum": spectrum}
            if peaklist is not None:
                input_paths["peaklist"] = peaklist
            else:
                auto_peaklist = _write_autopicked_peaklist(result.output_dir, data.peaks)
                input_paths["peaklist"] = auto_peaklist
            if z_values:
                input_paths["z_values"] = z_values

            if result.spectra is None:
                raise ValueError("No spectra in result")

            write_service_results(result, result.spectra, fit_config, input_paths, reporter)

    except Exception as e:
        show_error_with_details("Fitting", e)
        raise typer.Exit(1) from e


# === HELPERS ===


def _load_fit_data(
    *,
    spectrum: Path,
    peaklist: Path | None,
    z_values: Path | None,
    fit_config: PeakFitConfig,
    auto_pick_step: bool,
    reporter: Reporter,
) -> LoadedData:
    """Load fitting inputs with optional step-wise auto-pick GUI control."""
    stepper: AutoPickStepController | None = None
    if auto_pick_step:
        if peaklist is not None:
            reporter.warning("--auto-pick-step ignored because a peak list was provided.")
        elif fit_config.output.headless:
            error("--auto-pick-step requires interactive mode (disable --headless).")
            raise typer.Exit(1)
        else:
            from peakfit.ui.auto_pick_stepper import AutoPickStepController  # noqa: PLC0415

            stepper = AutoPickStepController()

    callback_builder = stepper.bind if stepper is not None else None

    try:
        if peaklist is None:
            return load_data(
                spectrum,
                peaklist,
                z_values,
                fit_config,
                reporter=reporter,
                auto_pick_callback_builder=callback_builder,
            )

        with console.status("[info]Loading data...[/info]", spinner="dots"):
            return load_data(
                spectrum,
                peaklist,
                z_values,
                fit_config,
                reporter=reporter,
            )
    finally:
        if stepper is not None:
            stepper.close()


def _build_config(
    config: Path | None,
    output: Path | None,
    formats: list[str] | None,
    headless: bool | None,
    noise: float | None,
    contour_level: float | None,
    lineshape: LineshapeName,
    refine: int,
    fixed: bool,
    jx: bool,
    phx: bool,
    phy: bool,
    exclude: list[int] | None,
    optimizer: str,
) -> PeakFitConfig:
    """Build config from CLI args or file."""
    normalized_formats = _normalize_output_formats(formats)

    if config is not None:
        cfg = load_config(config)
        if output:
            cfg.output.directory = output
        if normalized_formats is not None:
            cfg.output.formats = normalized_formats
        if headless is not None:
            cfg.output.headless = headless
        if noise is not None:
            cfg.noise_level = noise
        if contour_level is not None:
            cfg.clustering.contour_level = contour_level
        return cfg

    # Build from scratch
    fit_phase = []
    if phx:
        fit_phase.append("F3")
    if phy:
        fit_phase.append("F2")

    default_formats: list[OutputFormat] = ["json", "csv"]
    output_config = OutputConfig(
        formats=normalized_formats if normalized_formats is not None else default_formats,
        headless=headless if headless is not None else False,
    )
    if output:
        output_config.directory = output

    cfg = PeakFitConfig(
        fitting=FitConfig(
            lineshape=lineshape,
            refine_iterations=refine,
            fix_positions=fixed,
            fit_j_coupling=jx,
            fit_phase=fit_phase,
        ),
        clustering=ClusterConfig(contour_level=contour_level),
        output=output_config,
        noise_level=noise,
        exclude_planes=exclude or [],
    )

    if optimizer == "differential_evolution":
        cfg.fitting.max_iterations = DIFF_EVOLUTION_MAXITER
    elif optimizer == "basin_hopping":
        cfg.fitting.max_iterations = BASIN_HOPPING_NITER

    return cfg


def _normalize_output_formats(formats: list[str] | None) -> list[OutputFormat] | None:
    """Validate and normalize repeated --format values."""
    if not formats:
        return None

    allowed = set(VALID_OUTPUT_FORMATS)
    normalized = [fmt.lower() for fmt in formats]
    invalid = sorted({fmt for fmt in normalized if fmt not in allowed})
    if invalid:
        allowed_text = ", ".join(VALID_OUTPUT_FORMATS)
        invalid_text = ", ".join(invalid)
        raise typer.BadParameter(
            f"Unknown output format(s): {invalid_text}. Choose from: {allowed_text}."
        )

    deduped = list(dict.fromkeys(normalized))
    return cast("list[OutputFormat]", deduped)


def _resolve_output(fit_config: PeakFitConfig) -> Path:
    """Resolve output directory with timestamp."""
    base = fit_config.output.directory or Path("./")
    output_dir = resolve_output_path(base, include_timestamp=fit_config.output.include_timestamp)
    fit_config.output.directory = output_dir
    fit_config.output.include_timestamp = False
    return output_dir


def _write_autopicked_peaklist(output_dir: Path, peaks: list[Peak]) -> Path:
    """Write auto-detected peaks to Sparky-like list format for reproducibility."""
    peaklist_dir = output_dir / "metadata"
    peaklist_dir.mkdir(parents=True, exist_ok=True)
    peaklist_path = peaklist_dir / "autopicked.list"

    columns = [f"w{i + 1}" for i in range(len(peaks[0].positions))] if peaks else []
    header = "Assignment" if not columns else f"Assignment {' '.join(columns)}"

    lines = [header]
    for peak in peaks:
        positions = " ".join(f"{float(pos):.6f}" for pos in peak.positions)
        lines.append(f"{peak.name} {positions}")

    peaklist_path.write_text("\n".join(lines) + "\n")
    return peaklist_path


def _run_interactive_fit(
    data: LoadedData,
    fit_config: PeakFitConfig,
    output_dir: Path,
    optimizer: str,
    workers: int,
) -> ServiceResult:
    """Run fitting with interactive live display."""
    # Use a mutable container for the live display context
    live_ctx: dict[str, Any] = {"update": None}

    def progress_callback(item: Any) -> None:
        """Handle progress events from pipeline."""
        if isinstance(item, ProgressStart):
            # Start the live display with total from service
            live_ctx["display"] = live_fit_display(total_steps=item.total_steps)
            live_ctx["update"] = live_ctx["display"].__enter__()
        elif isinstance(item, tuple) and item[0] == "status":
            if live_ctx["update"]:
                live_ctx["update"](status_message=item[1], advance=0)
        elif isinstance(item, FitResult):
            if live_ctx["update"]:
                cluster_id = str(item.metadata.get("cluster_id", "??"))
                warnings = []
                bounds_hit = [p for p in item.params.values() if p.is_at_boundary()]
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
            optimizer=optimizer,
            workers=workers,
            progress_callback=progress_callback,
        )
    finally:
        # Clean up the live display context
        if "display" in live_ctx:
            live_ctx["display"].__exit__(None, None, None)


# Threshold for formatting high chi-squared
_HIGH_REDCHI = 5.0
_MAX_PEAK_LABELS = 3


def _format_review_clusters(reviews: list[ClusterReview]) -> list[dict[str, Any]]:
    """Format ClusterReview objects for display."""
    formatted = []
    for review in reviews:
        if review.reason == "diverged":
            status, status_color, details = "Diverged", "metric.bad", "Optimizer failed"
        elif review.reason == "high_chi":
            status, status_color = "High χ²", "metric.warn"
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
                "chi_sq_color": "metric.bad" if review.redchi > _HIGH_REDCHI else "metric.warn",
                "details": details,
            }
        )

    return formatted
