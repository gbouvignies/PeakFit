"""Fitting pipeline coordinating CLI input with core algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from peakfit.core.algorithms.clustering import create_clusters
from peakfit.core.algorithms.noise import prepare_noise_level
from peakfit.core.domain.config import PeakFitConfig
from peakfit.core.domain.peaks_io import read_list
from peakfit.core.domain.spectrum import Spectra, get_shape_names, read_spectra
from peakfit.core.domain.state import FittingState
from peakfit.core.fitting.strategies import STRATEGIES
from peakfit.core.shared.constants import (
    LEAST_SQUARES_FTOL,
    LEAST_SQUARES_MAX_NFEV,
    LEAST_SQUARES_XTOL,
)
from peakfit.core.shared.events import Event, EventDispatcher, EventType
from peakfit.io.output import write_profiles, write_shifts
from peakfit.io.state import StateRepository
from peakfit.services.fit.fitting import fit_all_clusters
from peakfit.services.fit.writer import write_new_format_outputs, write_simulated_spectra
from peakfit.ui import (
    close_logging,
    console,
    create_table,
    error,
    export_html,
    info,
    log,
    log_dict,
    log_section,
    setup_logging,
    show_banner,
    show_footer,
    show_header,
    show_run_info,
    spacer,
    success,
    warning,
)


@dataclass
class FitArguments:
    """Arguments for fitting process."""

    path_spectra: Path = field(default_factory=Path)
    path_list: Path = field(default_factory=Path)
    path_z_values: Path | None = None
    path_output: Path = field(default_factory=lambda: Path("Fits"))
    contour_level: float | None = None
    noise: float | None = None
    refine_nb: int = 1
    fixed: bool = False
    jx: bool = False
    phx: bool = False
    phy: bool = False
    exclude: list[int] = field(default_factory=list)
    pvoigt: bool = False
    lorentzian: bool = False
    gaussian: bool = False


def config_to_fit_args(
    config: PeakFitConfig,
    spectrum_path: Path,
    peaklist_path: Path,
    z_values_path: Path | None,
) -> FitArguments:
    """Convert modern config to FitArguments."""
    return FitArguments(
        path_spectra=spectrum_path,
        path_list=peaklist_path,
        path_z_values=z_values_path,
        contour_level=config.clustering.contour_level,
        noise=config.noise_level,
        path_output=config.output.directory,
        refine_nb=config.fitting.refine_iterations,
        fixed=config.fitting.fix_positions,
        jx=config.fitting.fit_j_coupling,
        phx=config.fitting.fit_phase_x,
        phy=config.fitting.fit_phase_y,
        exclude=config.exclude_planes,
        pvoigt=config.fitting.lineshape == "pvoigt",
        lorentzian=config.fitting.lineshape == "lorentzian",
        gaussian=config.fitting.lineshape == "gaussian",
    )


class FitPipeline:
    """High-level orchestrator for running the fitting workflow."""

    @staticmethod
    def run(
        spectrum_path: Path,
        peaklist_path: Path,
        z_values_path: Path | None,
        config: PeakFitConfig,
        *,
        optimizer: str = "leastsq",
        save_state: bool = True,
        verbose: bool = False,
        dispatcher: EventDispatcher | None = None,
    ) -> None:
        FitPipeline._run_fit(
            spectrum_path,
            peaklist_path,
            z_values_path,
            config,
            optimizer=optimizer,
            save_state=save_state,
            verbose=verbose,
            dispatcher=dispatcher,
        )

    @staticmethod
    def _run_fit(
        spectrum_path: Path,
        peaklist_path: Path,
        z_values_path: Path | None,
        config: PeakFitConfig,
        *,
        optimizer: str,
        save_state: bool,
        verbose: bool,
        dispatcher: EventDispatcher | None,
    ) -> None:
        """Run the fitting process with the given configuration."""
        start_time_dt = datetime.now()
        _dispatch_event(
            dispatcher,
            Event(
                event_type=EventType.FIT_STARTED,
                data={
                    "spectrum": str(spectrum_path),
                    "peaklist": str(peaklist_path),
                    "optimizer": optimizer,
                },
            ),
        )

        # Setup logging
        log_file = config.output.directory / "peakfit.log"
        setup_logging(log_file=log_file, verbose=False)

        if verbose:
            show_banner(verbose)
        else:
            show_run_info(start_time_dt)

        # NOTE: we do not mutate the configuration object to set a runtime
        # 'verbose' flag; instead, pass the `verbose` value explicitly to any
        # components that require it. Avoid setting private attributes on the
        # Pydantic model to keep types and model schema stable.

        valid_optimizers = sorted(STRATEGIES.keys())
        if optimizer not in valid_optimizers:
            error(f"Invalid optimizer: {optimizer}")
            info(f"Valid options: {', '.join(valid_optimizers)}")
            raise SystemExit(1)

        if optimizer != "leastsq":
            warning(f"Using global optimizer: {optimizer}")
            console.print("  [dim]This may take significantly longer than standard fitting[/dim]")

        _print_configuration(config.output.directory)

        console.print()
        show_header("Loading Input Files")

        clargs = config_to_fit_args(config, spectrum_path, peaklist_path, z_values_path)

        with console.status("[cyan]Reading spectrum...[/cyan]", spinner="dots"):
            spectra = read_spectra(clargs.path_spectra, clargs.path_z_values, clargs.exclude)

        log_dict(
            {
                "Spectrum": str(spectrum_path),
                "Dimensions": str(spectra.data.shape),
                "Size": f"{spectrum_path.stat().st_size / 1024 / 1024:.1f} MB",
                "Data type": str(spectra.data.dtype),
            }
        )

        log_section("Noise Estimation")
        noise_was_provided = clargs.noise is not None and clargs.noise > 0.0
        clargs.noise = prepare_noise_level(clargs, spectra)
        if clargs.noise is None:
            raise ValueError("Noise must be set by prepare_noise_level")
        noise_value: float = float(clargs.noise)
        noise_source = "user-provided" if noise_was_provided else "estimated"
        log(
            f"Method: {'User-provided' if noise_was_provided else 'Median Absolute Deviation (MAD)'}"
        )
        log(f"Noise level: {noise_value:.2f} ({noise_source})")

        log_section("Lineshape Detection")
        shape_names = get_shape_names(clargs, spectra)
        log(f"Selected lineshape: {shape_names}")

        clargs.contour_level = clargs.contour_level or 5.0 * clargs.noise

        _print_spectrum_info(
            spectrum_path, spectra, shape_names, clargs.noise, noise_source, clargs.contour_level
        )

        log_section("Peak List")
        peaks = read_list(spectra, shape_names, clargs)
        log_dict(
            {
                "Peak list": str(peaklist_path),
                "Format": "Sparky/NMRPipe",
                "Peaks": len(peaks),
            }
        )

        _print_peaklist_info(peaklist_path, z_values_path, len(peaks))

        console.print()
        show_header("Clustering Peaks")
        log_section("Clustering")
        log("Algorithm: DBSCAN")
        if clargs.noise > 0:
            noise_multiplier = f"{clargs.contour_level / clargs.noise:.1f} * noise"
        else:
            # Noise can be zero when estimated from silent regions; avoid division by zero
            noise_multiplier = "n/a (noise level is 0)"

        log(f"Contour level: {clargs.contour_level:.2f} ({noise_multiplier})")

        with console.status(
            "[cyan]Segmenting spectra and clustering peaks...[/cyan]", spinner="dots"
        ):
            clusters = create_clusters(spectra, peaks, clargs.contour_level)

        log(f"Identified {len(clusters)} clusters")
        cluster_sizes = [len(c.peaks) for c in clusters]
        log_dict(
            {
                "Min": f"{min(cluster_sizes)} peak" if cluster_sizes else "N/A",
                "Max": f"{max(cluster_sizes)} peaks" if cluster_sizes else "N/A",
                "Median": f"{sorted(cluster_sizes)[len(cluster_sizes) // 2]} peaks"
                if cluster_sizes
                else "N/A",
            }
        )

        min_peaks = min(cluster_sizes) if cluster_sizes else 0
        max_peaks = max(cluster_sizes) if cluster_sizes else 0
        if min_peaks == max_peaks:
            cluster_desc = f"{min_peaks} peak{'s' if min_peaks != 1 else ''} per cluster"
        else:
            cluster_desc = f"{min_peaks}-{max_peaks} peaks per cluster"

        success(f"Identified {len(clusters)} clusters ({cluster_desc})")

        console.print()
        show_header("Fitting Clusters")
        log_section("Fitting")
        log(f"Optimizer: {optimizer}")
        log("Backend: numpy (default)")
        if optimizer == "leastsq":
            log(f"Tolerances: ftol={LEAST_SQUARES_FTOL:.0e}, xtol={LEAST_SQUARES_XTOL:.0e}")
            log(f"Max iterations: {LEAST_SQUARES_MAX_NFEV}")
        log("")

        if optimizer != "leastsq":
            info(f"Using {optimizer} optimizer...")

        params = fit_all_clusters(
            clargs,
            clusters,
            optimizer=optimizer,
            verbose=verbose,
            dispatcher=dispatcher,
        )

        console.print()
        show_header("Fitting Complete")

        end_time_dt = datetime.now()
        total_time = (end_time_dt - start_time_dt).total_seconds()

        log_section("Output Files")
        config.output.directory.mkdir(parents=True, exist_ok=True)

        # Write legacy .out files only if --legacy flag is set
        if config.output.include_legacy:
            legacy_dir = config.output.directory / "legacy"
            legacy_dir.mkdir(parents=True, exist_ok=True)

            with console.status("[cyan]Writing legacy profiles...[/cyan]", spinner="dots"):
                write_profiles(legacy_dir, spectra.z_values, clusters, params, clargs)
            success(
                f"Legacy profiles: {config.output.directory.name}/legacy/{len(peaks)} *.out files"
            )
            log(f"Legacy profile files: {len(peaks)} *.out files in legacy/")

            with console.status("[cyan]Writing legacy shifts...[/cyan]", spinner="dots"):
                write_shifts(peaks, params, legacy_dir / "shifts.list")
            success(f"Legacy shifts: {config.output.directory.name}/legacy/shifts.list")
            log("Legacy shifts file: legacy/shifts.list")

        if config.output.save_simulated:
            write_simulated_spectra(config.output.directory, spectra, clusters, params)
            success(f"Simulated spectra: {config.output.directory.name}/simulated_*.ft*")

        if config.output.save_html_report:
            with console.status("[cyan]Generating HTML report...[/cyan]", spinner="dots"):
                export_html(config.output.directory / "logs.html")
            success(f"HTML report: {config.output.directory.name}/logs.html")
            log(f"HTML report: {config.output.directory / 'logs.html'}")

        if save_state:
            with console.status("[cyan]Saving fitting state...[/cyan]", spinner="dots"):
                state_file = StateRepository.default_path(config.output.directory)
                state = FittingState(
                    clusters=clusters, params=params, noise=clargs.noise, peaks=peaks
                )
                StateRepository.save(state_file, state)
            success(f"Fitting state: {config.output.directory.name}/cache/state.pkl")
            log(f"State file: {state_file}")

        # Write structured outputs (JSON, CSV, Markdown)
        # These are in addition to the legacy .out files
        if "json" in config.output.formats or "csv" in config.output.formats:
            input_files = {
                "spectrum": spectrum_path,
                "peaklist": peaklist_path,
            }
            if z_values_path:
                input_files["z_values"] = z_values_path

            write_new_format_outputs(
                output_dir=config.output.directory,
                spectra=spectra,
                clusters=clusters,
                params=params,
                noise=clargs.noise,
                config=config.model_dump(),
                input_files=input_files,
                verbosity=config.output.verbosity,
                include_legacy=config.output.include_legacy,
            )

        log(f"Log file: {log_file}")

        log_section("Results Summary")
        log(f"Total clusters: {len(clusters)}")
        log(f"Total peaks: {len(peaks)}")
        log(f"Total time: {total_time:.0f}s ({total_time / 60:.1f}m)")
        log(f"Average time per cluster: {total_time / len(clusters):.1f}s")

        console.print()
        summary_table = create_table("Results Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white", justify="right")

        summary_table.add_row("Total clusters", str(len(clusters)))
        summary_table.add_row("Total peaks", str(len(peaks)))

        successful_clusters = len(clusters)
        if len(clusters) > 0:
            success_pct = successful_clusters / len(clusters) * 100
            summary_table.add_row(
                "Successful fits",
                f"{successful_clusters}/{len(clusters)} ({success_pct:.0f}%)",
            )
        else:
            summary_table.add_row("Successful fits", "0/0 (0%)")

        if total_time < 60:
            time_str = f"{total_time:.1f}s"
        else:
            time_str = f"{int(total_time // 60)}m {int(total_time % 60)}s"
        summary_table.add_row("Total time", time_str)

        avg_time_per_cluster = total_time / len(clusters) if len(clusters) > 0 else 0
        summary_table.add_row("Time per cluster", f"{avg_time_per_cluster:.1f}s")
        summary_table.add_row("Mode", "Automatic")

        console.print(summary_table)
        console.print()

        output_dir_name = config.output.directory.name
        spectrum_name = spectrum_path.name

        _dispatch_event(
            dispatcher,
            Event(
                event_type=EventType.FIT_COMPLETED,
                data={
                    "clusters": len(clusters),
                    "peaks": len(peaks),
                    "total_time_sec": total_time,
                    "output_dir": str(config.output.directory),
                },
            ),
        )

        show_header("Next Steps")
        console.print("1. View intensity profiles:")
        console.print(f"   [cyan]peakfit plot intensity {output_dir_name}/[/cyan]")
        console.print("2. View fitted spectra:")
        console.print(
            f"   [cyan]peakfit plot spectra {output_dir_name}/ --spectrum {spectrum_name}[/cyan]"
        )
        console.print("3. Uncertainty analysis:")
        console.print(f"   [cyan]peakfit analyze mcmc {output_dir_name}/[/cyan]")
        console.print("4. Check log file:")
        console.print(f"   [cyan]less {output_dir_name}/peakfit.log[/cyan]")
        console.print()

        show_footer(start_time_dt, end_time_dt)
        close_logging()


def _dispatch_event(dispatcher: EventDispatcher | None, event: Event) -> None:
    """Safely dispatch an event when a dispatcher is provided."""

    if dispatcher is None:
        return

    dispatcher.dispatch(event)


def _print_configuration(output_dir: Path) -> None:
    """Print configuration information in a consolidated table."""
    spacer()

    config_table = create_table("Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="white", justify="right")

    config_table.add_row("Backend", "numpy")
    config_table.add_row("Parallel processing", "disabled")
    config_table.add_row("Output directory", str(output_dir.name))

    console.print(config_table)
    spacer()


def _print_spectrum_info(
    spectrum_path: Path,
    spectra: Spectra,
    shape_names: list[str],
    noise: float,
    noise_source: str,
    contour_level: float,
) -> None:
    """Print consolidated spectrum information table."""
    spacer()

    spectrum_table = create_table(f"Spectrum: {spectrum_path.name}")
    spectrum_table.add_column("Property", style="cyan")
    spectrum_table.add_column("Value", style="white", justify="right")

    shape = spectra.data.shape
    n_spectral = spectra.n_spectral_dims

    # Build dimension string with Fn labels
    if n_spectral >= 1:
        dim_parts = []
        for dim in spectra.dimensions:
            size = dim.size
            label = dim.label
            nucleus = f" ({dim.nucleus})" if dim.nucleus else ""
            dim_parts.append(f"{label}{nucleus}: {size} pts")
        dim_str = ", ".join(dim_parts)
        # Also show shape
        shape_str = " × ".join(str(s) for s in reversed(shape[1:]))
        dim_str = f"{shape_str} ({dim_str})"
    else:
        dim_str = str(shape)

    spectrum_table.add_row("Spectral dimensions", str(n_spectral))
    spectrum_table.add_row("Dimension sizes", dim_str)
    spectrum_table.add_row("Number of planes", str(len(spectra.z_values)))

    lineshape_str = ", ".join(shape_names) if isinstance(shape_names, list) else str(shape_names)
    spectrum_table.add_row("Lineshapes", lineshape_str)
    spectrum_table.add_row("Noise level", f"{noise:.2f} ({noise_source})")
    spectrum_table.add_row("Contour level", f"{contour_level:.2f}")

    console.print(spectrum_table)
    spacer()


def _print_peaklist_info(peaklist_path: Path, z_values_path: Path | None, n_peaks: int) -> None:
    """Print consolidated peak list information table."""
    spacer()

    peaklist_table = create_table(f"Peak List: {peaklist_path.name}")
    peaklist_table.add_column("Property", style="cyan")
    peaklist_table.add_column("Value", style="white", justify="right")

    suffix = peaklist_path.suffix.lower()
    if suffix == ".list":
        format_str = "Sparky/NMRPipe"
    elif suffix == ".csv":
        format_str = "CSV"
    elif suffix == ".json":
        format_str = "JSON"
    elif suffix in {".xlsx", ".xls"}:
        format_str = "Excel"
    else:
        format_str = "Unknown"

    peaklist_table.add_row("Format", format_str)
    peaklist_table.add_row("Number of peaks", str(n_peaks))

    if z_values_path:
        peaklist_table.add_row("Z-values file", z_values_path.name)
    else:
        peaklist_table.add_row("Z-values file", "auto-detected")

    console.print(peaklist_table)
    spacer()
