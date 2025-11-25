"""Fitting pipeline coordinating CLI input with core algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from importlib import import_module
from pathlib import Path

import numpy as np  # type: ignore[import-not-found]
from scipy.optimize import least_squares  # type: ignore[import-not-found]
from threadpoolctl import threadpool_limits  # type: ignore[import-not-found]

from peakfit.core.algorithms.clustering import create_clusters
from peakfit.core.algorithms.noise import prepare_noise_level
from peakfit.core.domain.config import PeakFitConfig
from peakfit.core.domain.peaks import create_params
from peakfit.core.domain.peaks_io import read_list
from peakfit.core.domain.spectrum import get_shape_names, read_spectra
from peakfit.core.domain.state import FittingState
from peakfit.core.fitting.computation import residuals, update_cluster_corrections
from peakfit.core.fitting.parameters import Parameters
from peakfit.core.fitting.simulation import simulate_data
from peakfit.core.shared.constants import (
    LEAST_SQUARES_FTOL,
    LEAST_SQUARES_MAX_NFEV,
    LEAST_SQUARES_XTOL,
)
from peakfit.infra.state import StateRepository
from peakfit.io.output import write_profiles, write_shifts
from peakfit.ui import PeakFitUI, console

ui = PeakFitUI


@dataclass
class FitArguments:
    """Arguments for fitting process."""

    path_spectra: Path = field(default_factory=Path)
    path_list: Path = field(default_factory=Path)
    path_z_values: Path | None = None
    path_output: Path = field(default_factory=lambda: Path("Fits"))
    contour_level: float | None = None
    noise: float = 0.0
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
        noise=float(config.noise_level or 0.0),
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
    ) -> None:
        FitPipeline._run_fit(
            spectrum_path,
            peaklist_path,
            z_values_path,
            config,
            optimizer=optimizer,
            save_state=save_state,
            verbose=verbose,
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
    ) -> None:
        """Run the fitting process with the given configuration."""
        start_time_dt = datetime.now()

        # Setup logging
        log_file = config.output.directory / "peakfit.log"
        ui.setup_logging(log_file=log_file, verbose=False)

        if verbose:
            ui.show_banner(verbose)
        else:
            ui.show_run_info(start_time_dt)

        config._verbose = verbose  # type: ignore[attr-defined]

        valid_optimizers = ["leastsq", "basin-hopping", "differential-evolution"]
        if optimizer not in valid_optimizers:
            ui.error(f"Invalid optimizer: {optimizer}")
            ui.info(f"Valid options: {', '.join(valid_optimizers)}")
            raise SystemExit(1)

        if optimizer != "leastsq":
            ui.warning(f"Using global optimizer: {optimizer}")
            console.print("  [dim]This may take significantly longer than standard fitting[/dim]")

        _print_configuration(config.output.directory)

        console.print()
        ui.show_header("Loading Input Files")

        clargs = config_to_fit_args(config, spectrum_path, peaklist_path, z_values_path)

        with console.status("[cyan]Reading spectrum...[/cyan]", spinner="dots"):
            spectra = read_spectra(clargs.path_spectra, clargs.path_z_values, clargs.exclude)

        ui.log_dict(
            {
                "Spectrum": str(spectrum_path),
                "Dimensions": str(spectra.data.shape),
                "Size": f"{spectrum_path.stat().st_size / 1024 / 1024:.1f} MB",
                "Data type": str(spectra.data.dtype),
            }
        )

        ui.log_section("Noise Estimation")
        noise_was_provided = clargs.noise is not None and clargs.noise > 0.0
        clargs.noise = prepare_noise_level(clargs, spectra)
        noise_source = "user-provided" if noise_was_provided else "estimated"
        ui.log(
            f"Method: {'User-provided' if noise_was_provided else 'Median Absolute Deviation (MAD)'}"
        )
        ui.log(f"Noise level: {clargs.noise:.2f} ({noise_source})")

        ui.log_section("Lineshape Detection")
        shape_names = get_shape_names(clargs, spectra)
        ui.log(f"Selected lineshape: {shape_names}")

        clargs.contour_level = clargs.contour_level or 5.0 * clargs.noise

        _print_spectrum_info(
            spectrum_path, spectra, shape_names, clargs.noise, noise_source, clargs.contour_level
        )

        ui.log_section("Peak List")
        peaks = read_list(spectra, shape_names, clargs)
        ui.log_dict(
            {
                "Peak list": str(peaklist_path),
                "Format": "Sparky/NMRPipe",
                "Peaks": len(peaks),
            }
        )

        _print_peaklist_info(peaklist_path, z_values_path, len(peaks))

        console.print()
        ui.show_header("Clustering Peaks")
        ui.log_section("Clustering")
        ui.log("Algorithm: DBSCAN")
        if clargs.noise > 0:
            noise_multiplier = f"{clargs.contour_level / clargs.noise:.1f} * noise"
        else:
            # Noise can be zero when estimated from silent regions; avoid division by zero
            noise_multiplier = "n/a (noise level is 0)"

        ui.log(f"Contour level: {clargs.contour_level:.2f} ({noise_multiplier})")

        with console.status(
            "[cyan]Segmenting spectra and clustering peaks...[/cyan]", spinner="dots"
        ):
            clusters = create_clusters(spectra, peaks, clargs.contour_level)

        ui.log(f"Identified {len(clusters)} clusters")
        cluster_sizes = [len(c.peaks) for c in clusters]
        ui.log_dict(
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

        ui.success(f"Identified {len(clusters)} clusters ({cluster_desc})")

        console.print()
        ui.show_header("Fitting Clusters")
        ui.log_section("Fitting")
        ui.log(f"Optimizer: {optimizer}")
        ui.log("Backend: numpy (default)")
        ui.log(f"Tolerances: ftol={LEAST_SQUARES_FTOL:.0e}, xtol={LEAST_SQUARES_XTOL:.0e}")
        ui.log(f"Max iterations: {LEAST_SQUARES_MAX_NFEV}")
        ui.log("")

        if optimizer != "leastsq":
            ui.info(f"Using {optimizer} optimizer...")
            params = _fit_clusters_global(clargs, clusters, optimizer, verbose)
        else:
            params = _fit_clusters(clargs, clusters, verbose)

        console.print()
        ui.show_header("Fitting Complete")

        end_time_dt = datetime.now()
        total_time = (end_time_dt - start_time_dt).total_seconds()

        ui.log_section("Output Files")
        config.output.directory.mkdir(parents=True, exist_ok=True)

        with console.status("[cyan]Writing profiles...[/cyan]", spinner="dots"):
            write_profiles(config.output.directory, spectra.z_values, clusters, params, clargs)
        ui.success(f"Peak profiles: {config.output.directory.name}/{len(peaks)} *.out files")
        ui.log(f"Profile files: {len(peaks)} *.out files")

        with console.status("[cyan]Writing shifts...[/cyan]", spinner="dots"):
            write_shifts(peaks, params, config.output.directory / "shifts.list")
        ui.success(f"Chemical shifts: {config.output.directory.name}/shifts.list")
        ui.log(f"Shifts file: {config.output.directory / 'shifts.list'}")

        if config.output.save_simulated:
            with console.status("[cyan]Writing simulated spectra...[/cyan]", spinner="dots"):
                _write_spectra(config.output.directory, spectra, clusters, params)
            ui.success(f"Simulated spectra: {config.output.directory.name}/simulated_*.ft*")

        if config.output.save_html_report:
            with console.status("[cyan]Generating HTML report...[/cyan]", spinner="dots"):
                ui.export_html(config.output.directory / "logs.html")
            ui.success(f"HTML report: {config.output.directory.name}/logs.html")
            ui.log(f"HTML report: {config.output.directory / 'logs.html'}")

        if save_state:
            with console.status("[cyan]Saving fitting state...[/cyan]", spinner="dots"):
                state_file = StateRepository.default_path(config.output.directory)
                state = FittingState(
                    clusters=clusters, params=params, noise=clargs.noise, peaks=peaks
                )
                StateRepository.save(state_file, state)
            ui.success(f"Fitting state: {config.output.directory.name}/.peakfit_state.pkl")
            ui.log(f"State file: {state_file}")

        ui.log(f"Log file: {log_file}")

        ui.log_section("Results Summary")
        ui.log(f"Total clusters: {len(clusters)}")
        ui.log(f"Total peaks: {len(peaks)}")
        ui.log(f"Total time: {total_time:.0f}s ({total_time / 60:.1f}m)")
        ui.log(f"Average time per cluster: {total_time / len(clusters):.1f}s")

        console.print()
        summary_table = ui.create_table("Results Summary")
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

        ui.show_header("Next Steps")
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

        ui.show_footer(start_time_dt, end_time_dt)
        ui.close_logging()


def _residual_wrapper(x: np.ndarray, params: Parameters, cluster, noise: float) -> np.ndarray:
    """Wrapper to convert array to Parameters for residual calculation."""
    vary_names = params.get_vary_names()
    for i, name in enumerate(vary_names):
        params[name].value = x[i]
    return residuals(params, cluster, noise)


def _fit_clusters(clargs: FitArguments, clusters: list, verbose: bool = False) -> Parameters:
    """Fit all clusters and return parameters."""
    params_all = Parameters()

    with threadpool_limits(limits=1, user_api="blas"):
        for index in range(clargs.refine_nb + 1):
            if index == 0:
                ui.subsection_header("Initial Fit")
            else:
                ui.subsection_header(f"Refining Parameters (Iteration {index})")
                ui.log_section(f"Refinement Iteration {index}")
                update_cluster_corrections(params_all, clusters)

            for cluster_idx, cluster in enumerate(clusters, 1):
                import time as time_module

                cluster_start = time_module.time()
                peak_names = [peak.name for peak in cluster.peaks]
                peaks_str = ", ".join(peak_names)
                n_peaks = len(cluster.peaks)

                if cluster_idx > 1:
                    console.print()
                console.print(
                    f"[bold cyan]Cluster {cluster_idx}/{len(clusters)}[/bold cyan] [dim]│[/dim] "
                    f"{peaks_str} [dim][{n_peaks} peak{'s' if n_peaks != 1 else ''}][/dim]"
                )

                ui.log("")
                ui.log(f"Cluster {cluster_idx}/{len(clusters)}: {peaks_str}")
                ui.log(f"  - Peaks: {len(cluster.peaks)}")

                params = create_params(cluster.peaks, fixed=clargs.fixed)
                params = _update_params(params, params_all)

                vary_names = params.get_vary_names()
                ui.log(f"  - Varying parameters: {len(vary_names)}")

                x0 = params.get_vary_values()
                bounds_lower = np.array([params[name].min for name in vary_names])
                bounds_upper = np.array([params[name].max for name in vary_names])

                scipy_verbose = 2 if verbose else 0
                result = least_squares(
                    _residual_wrapper,
                    x0,
                    args=(params, cluster, clargs.noise),
                    bounds=(bounds_lower, bounds_upper),
                    ftol=LEAST_SQUARES_FTOL,
                    xtol=LEAST_SQUARES_XTOL,
                    max_nfev=LEAST_SQUARES_MAX_NFEV,
                    verbose=scipy_verbose,
                )

                for i, name in enumerate(vary_names):
                    params[name].value = result.x[i]

                if result.jac is not None and len(result.fun) > len(vary_names):
                    ndata = len(result.fun)
                    nvarys = len(vary_names)
                    redchi = np.sum(result.fun**2) / max(1, ndata - nvarys)
                    try:
                        jtj = result.jac.T @ result.jac
                        cov = np.linalg.inv(jtj) * redchi
                        stderr = np.sqrt(np.diag(cov))
                        for i, name in enumerate(vary_names):
                            params[name].stderr = float(stderr[i])
                    except np.linalg.LinAlgError:
                        pass

                cluster_time = time_module.time() - cluster_start

                if result.success:
                    console.print(
                        f"[green]✓[/green] Converged [dim]│[/dim] "
                        f"χ² = [cyan]{result.cost:.2e}[/cyan] [dim]│[/dim] "
                        f"{result.nfev} evaluations [dim]│[/dim] "
                        f"{cluster_time:.1f}s"
                    )
                    ui.log("  - Status: Converged", level="info")
                else:
                    console.print(
                        f"[yellow]⚠[/yellow] {result.message} [dim]│[/dim] "
                        f"χ² = [cyan]{result.cost:.2e}[/cyan] [dim]│[/dim] "
                        f"{result.nfev} evaluations [dim]│[/dim] "
                        f"{cluster_time:.1f}s"
                    )
                    ui.log(f"  - Status: {result.message}", level="warning")

                ui.log(f"  - Cost: {result.cost:.3e}")
                ui.log(f"  - Function evaluations: {result.nfev}")
                ui.log(f"  - Time: {cluster_time:.1f}s")

                params_all.update(params)

    return params_all


def _fit_clusters_global(
    clargs: FitArguments, clusters: list, optimizer: str, verbose: bool = False
) -> Parameters:
    """Fit all clusters using global optimization."""
    from peakfit.core.fitting.advanced import fit_basin_hopping, fit_differential_evolution

    params_all = Parameters()

    with threadpool_limits(limits=1, user_api="blas"):
        for index in range(clargs.refine_nb + 1):
            if index == 0:
                ui.subsection_header("Initial Fit")
            else:
                ui.subsection_header(f"Refining Parameters (Iteration {index})")
                ui.log_section(f"Refinement Iteration {index}")
                update_cluster_corrections(params_all, clusters)

            for cluster_idx, cluster in enumerate(clusters, 1):
                peak_names = [peak.name for peak in cluster.peaks]
                peaks_str = ", ".join(peak_names)

                if cluster_idx > 1:
                    console.print()
                console.print(
                    f"[bold cyan]Cluster {cluster_idx}/{len(clusters)}[/bold cyan] [dim]│[/dim] {peaks_str}"
                )

                params = create_params(cluster.peaks, fixed=clargs.fixed)
                params = _update_params(params, params_all)

                if optimizer == "basin-hopping":
                    result = fit_basin_hopping(
                        params,
                        cluster,
                        clargs.noise,
                        n_iterations=50,
                        temperature=1.0,
                        step_size=0.5,
                    )
                elif optimizer == "differential-evolution":
                    result = fit_differential_evolution(
                        params,
                        cluster,
                        clargs.noise,
                        max_iterations=500,
                        population_size=15,
                        polish=True,
                    )
                else:
                    msg = f"Unknown optimizer: {optimizer}"
                    raise ValueError(msg)

                success = result.success if hasattr(result, "success") else True
                chisqr = result.chisqr if hasattr(result, "chisqr") else result.cost
                nfev = result.nfev if hasattr(result, "nfev") else "N/A"

                if success:
                    console.print(
                        f"[green]✓ Converged[/green] [dim]│[/dim] Cost: [cyan]{chisqr:.3e}[/cyan] [dim]│[/dim] Evaluations: [cyan]{nfev}[/cyan]"
                    )
                else:
                    message = result.message if hasattr(result, "message") else "Did not converge"
                    console.print(
                        f"[yellow]⚠ {message}[/yellow] [dim]│[/dim] Cost: [cyan]{chisqr:.3e}[/cyan] [dim]│[/dim] Evaluations: [cyan]{nfev}[/cyan]"
                    )

                params_all.update(result.params)

    return params_all


def _update_params(params: Parameters, params_all: Parameters) -> Parameters:
    """Update parameters with global parameters."""
    for key in params:
        if key in params_all:
            params[key] = params_all[key]
    return params


def _write_spectra(path: Path, spectra, clusters, params: Parameters) -> None:
    """Write simulated spectra to file."""

    try:
        ng = import_module("nmrglue")
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
        msg = (
            "Writing simulated spectra requires the optional 'nmrglue' dependency. "
            "Install it with `uv pip install nmrglue`."
        )
        raise RuntimeError(msg) from exc

    ui.action("Writing simulated spectra...")

    data_simulated = simulate_data(params, clusters, spectra.data)

    if spectra.pseudo_dim_added:
        data_simulated = np.squeeze(data_simulated, axis=0)

    ng.pipe.write(
        str(path / f"simulated.ft{data_simulated.ndim}"),
        spectra.dic,
        data_simulated.astype(np.float32),
        overwrite=True,
    )


def _print_configuration(output_dir: Path) -> None:
    """Print configuration information in a consolidated table."""
    ui.spacer()

    config_table = ui.create_table("Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="white", justify="right")

    config_table.add_row("Backend", "numpy")
    config_table.add_row("Parallel processing", "disabled")
    config_table.add_row("Output directory", str(output_dir.name))

    console.print(config_table)
    ui.spacer()


def _print_spectrum_info(
    spectrum_path: Path,
    spectra,
    shape_names: list,
    noise: float,
    noise_source: str,
    contour_level: float,
) -> None:
    """Print consolidated spectrum information table."""
    ui.spacer()

    spectrum_table = ui.create_table(f"Spectrum: {spectrum_path.name}")
    spectrum_table.add_column("Property", style="cyan")
    spectrum_table.add_column("Value", style="white", justify="right")

    shape = spectra.data.shape
    if len(shape) == 3:
        dim_str = f"{shape[2]} × {shape[1]} × {shape[0]}"
    else:
        dim_str = str(shape)

    spectrum_table.add_row("Dimensions", dim_str)
    spectrum_table.add_row("Number of planes", str(len(spectra.z_values)))

    lineshape_str = ", ".join(shape_names) if isinstance(shape_names, list) else str(shape_names)
    spectrum_table.add_row("Lineshapes", lineshape_str)
    spectrum_table.add_row("Noise level", f"{noise:.2f} ({noise_source})")
    spectrum_table.add_row("Contour level", f"{contour_level:.2f}")

    console.print(spectrum_table)
    ui.spacer()


def _print_peaklist_info(peaklist_path: Path, z_values_path: Path | None, n_peaks: int) -> None:
    """Print consolidated peak list information table."""
    ui.spacer()

    peaklist_table = ui.create_table(f"Peak List: {peaklist_path.name}")
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
    ui.spacer()
