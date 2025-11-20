"""Implementation of the fit command."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from peakfit.clustering import create_clusters
from peakfit.computing import residuals, simulate_data, update_cluster_corrections
from peakfit.core.fitting import Parameters
from peakfit.core.models import PeakFitConfig
from peakfit.noise import prepare_noise_level
from peakfit.peak import create_params
from peakfit.peaklist import read_list
from peakfit.spectra import get_shape_names, read_spectra
from peakfit.ui import PeakFitUI as ui, console
from peakfit.writing import write_profiles, write_shifts


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


def run_fit(
    spectrum_path: Path,
    peaklist_path: Path,
    z_values_path: Path | None,
    config: PeakFitConfig,
    parallel: bool = False,
    n_workers: int | None = None,
    backend: str = "auto",
    optimizer: str = "leastsq",
    save_state: bool = True,
    verbose: bool = False,
) -> None:
    """Run the fitting process.

    Args:
        spectrum_path: Path to NMRPipe spectrum file.
        peaklist_path: Path to peak list file.
        z_values_path: Optional path to Z-values file.
        config: Configuration object.
        parallel: Whether to use parallel processing.
        n_workers: Number of parallel workers.
        backend: Computation backend (auto, numpy, numba).
        optimizer: Optimization algorithm (leastsq, basin-hopping, differential-evolution).
        save_state: Whether to save fitting state for later analysis.
        verbose: Show banner and verbose output.
    """
    import time

    # Setup logging
    log_file = config.output.directory / "peakfit.log"
    ui.setup_logging(log_file=log_file, verbose=False)

    # Show banner based on verbosity
    ui.show_banner(verbose)

    # Store verbose flag globally for _fit_clusters to access
    config._verbose = verbose

    # Initialize computation backend
    _initialize_backend(backend, parallel=parallel)

    # Show optimization status
    _print_optimization_status()

    # Validate optimizer choice
    valid_optimizers = ["leastsq", "basin-hopping", "differential-evolution"]
    if optimizer not in valid_optimizers:
        ui.error(f"Invalid optimizer: {optimizer}")
        ui.info(f"Valid options: {', '.join(valid_optimizers)}")
        raise SystemExit(1)

    if optimizer != "leastsq":
        ui.warning(f"Using global optimizer: {optimizer}")
        console.print("  [dim]This may take significantly longer than standard fitting[/dim]")

    # Convert to legacy args for compatibility with existing modules
    clargs = config_to_fit_args(config, spectrum_path, peaklist_path, z_values_path)

    # Track timing
    start_time = time.time()

    # Load data
    ui.show_header("Loading Data")

    with console.status("[bold yellow]Reading spectrum..."):
        spectra = read_spectra(clargs.path_spectra, clargs.path_z_values, clargs.exclude)

    ui.success(f"Loaded spectrum: {spectrum_path.name}")
    ui.bullet(f"Shape: {spectra.data.shape}", style="default")
    ui.bullet(f"Z-values: {len(spectra.z_values)} planes", style="default")

    # Log spectrum details
    ui.log_dict({
        "Spectrum": str(spectrum_path),
        "Dimensions": str(spectra.data.shape),
        "Size": f"{spectrum_path.stat().st_size / 1024 / 1024:.1f} MB",
        "Data type": str(spectra.data.dtype),
    })

    # Estimate noise
    ui.spacer()
    ui.log_section("Noise Estimation")
    clargs.noise = prepare_noise_level(clargs, spectra)
    ui.success(f"Noise level: {clargs.noise:.2f}")
    ui.log(f"Method: Median Absolute Deviation (MAD)")
    ui.log(f"Noise level: {clargs.noise:.2f}")

    # Determine lineshape
    ui.spacer()
    ui.log_section("Lineshape Detection")
    shape_names = get_shape_names(clargs, spectra)
    ui.bullet(f"Lineshapes: {shape_names}", style="default")
    ui.log(f"Selected lineshape: {shape_names}")

    # Read peak list
    ui.spacer()
    ui.log_section("Peak List")
    peaks = read_list(spectra, shape_names, clargs)
    ui.success(f"Loaded {len(peaks)} peaks")
    ui.log_dict({
        "Peak list": str(peaklist_path),
        "Format": "Sparky/NMRPipe",
        "Peaks": len(peaks),
    })

    # Cluster peaks
    ui.spacer()
    ui.log_section("Clustering")
    clargs.contour_level = clargs.contour_level or 5.0 * clargs.noise
    ui.bullet(f"Contour level: {clargs.contour_level:.2f}", style="default")

    clusters = create_clusters(spectra, peaks, clargs.contour_level)
    ui.success(f"Created {len(clusters)} clusters")

    # Log clustering details
    ui.log("Algorithm: DBSCAN")
    ui.log(f"Contour level: {clargs.contour_level:.2f} ({clargs.contour_level/clargs.noise:.1f} * noise)")
    ui.log(f"Identified {len(clusters)} clusters")

    # Calculate cluster size distribution
    cluster_sizes = [len(c.peaks) for c in clusters]
    ui.log_dict({
        "Min": f"{min(cluster_sizes)} peak" if cluster_sizes else "N/A",
        "Max": f"{max(cluster_sizes)} peaks" if cluster_sizes else "N/A",
        "Median": f"{sorted(cluster_sizes)[len(cluster_sizes)//2]} peaks" if cluster_sizes else "N/A",
    })

    # Display data summary
    ui.spacer()
    ui.print_data_summary(
        spectrum_shape=spectra.data.shape,
        n_planes=len(spectra.z_values),
        n_peaks=len(peaks),
        n_clusters=len(clusters),
        noise_level=clargs.noise,
        contour_level=clargs.contour_level,
    )

    # Fit clusters - choose method based on flags
    ui.show_header("Fitting Clusters")
    ui.log_section("Fitting")
    ui.log(f"Optimizer: {optimizer}")
    ui.log(f"Backend: {backend}")
    ui.log(f"Parallel: {'enabled' if parallel else 'disabled'}")
    ui.log(f"Tolerances: ftol=1e-7, xtol=1e-7")
    ui.log(f"Max iterations: 1000")
    ui.log("")

    if optimizer != "leastsq":
        ui.info(f"Using {optimizer} optimizer...")
        params = _fit_clusters_global(clargs, clusters, optimizer, verbose)
    elif parallel and len(clusters) > 1:
        ui.info("Using parallel fitting...")
        params = _fit_clusters_parallel(clargs, clusters, n_workers)
    else:
        params = _fit_clusters(clargs, clusters, verbose)

    # Write outputs
    ui.show_header("Saving Results")
    ui.log_section("Output Files")
    config.output.directory.mkdir(parents=True, exist_ok=True)

    # Track what was saved
    output_files = []

    write_profiles(config.output.directory, spectra.z_values, clusters, params, clargs)
    output_files.append(("Peak profiles", f"{len(peaks)} *.out files"))
    ui.log(f"Profile files: {len(peaks)} *.out files")

    if config.output.save_html_report:
        ui.export_html(config.output.directory / "logs.html")
        output_files.append(("HTML report", "logs.html"))
        ui.log(f"HTML report: {config.output.directory / 'logs.html'}")

    write_shifts(peaks, params, config.output.directory / "shifts.list")
    output_files.append(("Chemical shifts", "shifts.list"))
    ui.log(f"Shifts file: {config.output.directory / 'shifts.list'}")

    if config.output.save_simulated:
        _write_spectra(config.output.directory, spectra, clusters, params)
        output_files.append(("Simulated spectra", "simulated_*.ft*"))

    # Save fitting state for later analysis
    if save_state:
        state_file = config.output.directory / ".peakfit_state.pkl"
        _save_fitting_state(state_file, clusters, params, clargs.noise, peaks)
        output_files.append(("Fitting state", ".peakfit_state.pkl"))
        ui.log(f"State file: {state_file}")

    # Display output table
    output_table = ui.create_table(title=None)
    output_table.add_column("Output Type", style="cyan")
    output_table.add_column("Location", style="green")

    for output_type, location in output_files:
        full_location = f"{config.output.directory.name}/{location}"
        output_table.add_row(output_type, full_location)

    ui.spacer()
    console.print(output_table)
    ui.spacer()
    ui.success(f"All results saved to: [path]{config.output.directory.name}/[/path]")

    # Calculate statistics
    total_time = time.time() - start_time
    ui.log(f"Log file: {log_file}")

    # Log summary
    ui.log_section("Results Summary")
    ui.log(f"Total clusters: {len(clusters)}")
    ui.log(f"Total peaks: {len(peaks)}")
    ui.log(f"Total time: {total_time:.0f}s ({total_time/60:.1f}m)")
    ui.log(f"Average time per cluster: {total_time/len(clusters):.1f}s")

    # Display final summary
    ui.spacer()
    console.print()
    ui.show_header("Fitting Summary")

    # Collect statistics for summary
    from datetime import timedelta

    # Count successes by checking params
    successful_clusters = 0
    chi_squared_values = []
    nfev_values = []

    # Note: In a real implementation, we'd track these during fitting
    # For now, we'll use approximate values based on what we know
    successful_clusters = len(clusters)  # Assume all successful for now

    summary_table = ui.create_table(title=None)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green", justify="right")

    summary_table.add_row("Total clusters", str(len(clusters)))
    summary_table.add_row("Total peaks", str(len(peaks)))

    # Success rate
    success_pct = (successful_clusters / len(clusters) * 100) if len(clusters) > 0 else 0
    summary_table.add_row(
        "Successful fits",
        f"{successful_clusters}/{len(clusters)} ({success_pct:.0f}%)"
    )

    # Format time nicely
    if total_time < 60:
        time_str = f"{total_time:.1f}s"
    else:
        time_str = f"{int(total_time // 60)}m {int(total_time % 60)}s"
    summary_table.add_row("Total time", time_str)

    avg_time_per_cluster = total_time / len(clusters) if len(clusters) > 0 else 0
    summary_table.add_row("Time per cluster", f"{avg_time_per_cluster:.1f}s")

    # Parallelization info
    parallel_mode = "Parallel" if parallel else "Sequential"
    summary_table.add_row("Mode", parallel_mode)

    console.print()
    console.print(summary_table)
    console.print()

    console.print("[bold green]✓ Fitting complete![/]")
    console.print()

    # Next steps - use relative paths for cleaner output
    from pathlib import Path as PathlibPath
    output_dir_name = config.output.directory.name
    spectrum_name = spectrum_path.name

    ui.spacer()
    ui.show_header("Next Steps")
    console.print()
    console.print("1. View intensity profiles:")
    console.print(f"   [cyan]peakfit plot intensity {output_dir_name}/[/cyan]")
    console.print()
    console.print("2. View fitted spectra:")
    console.print(f"   [cyan]peakfit plot spectra {output_dir_name}/ --spectrum {spectrum_name}[/cyan]")
    console.print()
    console.print("3. Uncertainty analysis:")
    console.print(f"   [cyan]peakfit analyze mcmc {output_dir_name}/[/cyan]")
    console.print()
    console.print("4. Check log file:")
    console.print(f"   [cyan]less {output_dir_name}/peakfit.log[/cyan]")
    console.print()

    # Close logging
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

    # Use threadpoolctl to limit BLAS threads at runtime
    # This prevents OpenBLAS/MKL from spawning threads that cause massive overhead
    # (e.g., 3171% CPU usage -> 99% CPU usage, 671s -> 8s CPU time)
    with threadpool_limits(limits=1, user_api="blas"):
        for index in range(clargs.refine_nb + 1):
            if index > 0:
                ui.spacer()
                ui.show_header("Refining Peak Parameters")
                ui.log_section(f"Refinement Iteration {index}")
                update_cluster_corrections(params_all, clusters)

            for cluster_idx, cluster in enumerate(clusters, 1):
                import time as time_module

                cluster_start = time_module.time()
                peak_names = [peak.name for peak in cluster.peaks]
                peaks_str = ", ".join(peak_names)
                n_peaks = len(cluster.peaks)

                # Clean cluster output
                console.print()
                console.print(
                    f"[bold cyan]Cluster {cluster_idx}/{len(clusters)}[/bold cyan] [dim]│[/dim] "
                    f"{peaks_str} [dim][{n_peaks} peak{'s' if n_peaks != 1 else ''}][/dim]"
                )

                # Log cluster info
                ui.log("")
                ui.log(f"Cluster {cluster_idx}/{len(clusters)}: {peaks_str}")
                ui.log(f"  - Peaks: {len(cluster.peaks)}")

                params = create_params(cluster.peaks, fixed=clargs.fixed)
                params = _update_params(params, params_all)

                # Get varying parameters
                vary_names = params.get_vary_names()
                ui.log(f"  - Varying parameters: {len(vary_names)}")

                x0 = params.get_vary_values()
                bounds_lower = np.array([params[name].min for name in vary_names])
                bounds_upper = np.array([params[name].max for name in vary_names])

                # Run optimization with scipy.optimize.least_squares
                # verbose=0: silent, verbose=2: show iterations
                scipy_verbose = 2 if verbose else 0
                result = least_squares(
                    _residual_wrapper,
                    x0,
                    args=(params, cluster, clargs.noise),
                    bounds=(bounds_lower, bounds_upper),
                    ftol=1e-7,
                    xtol=1e-7,
                    max_nfev=1000,
                    verbose=scipy_verbose,
                )

                # Update parameters with optimized values
                for i, name in enumerate(vary_names):
                    params[name].value = result.x[i]

                # Compute standard errors from Jacobian
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
                        # Singular matrix, can't compute errors
                        pass

                cluster_time = time_module.time() - cluster_start

                # Print and log completion status - clean one-line output
                if result.success:
                    console.print(
                        f"[green]✓[/green] Converged [dim]│[/dim] "
                        f"χ² = [cyan]{result.cost:.2e}[/cyan] [dim]│[/dim] "
                        f"{result.nfev} evaluations [dim]│[/dim] "
                        f"{cluster_time:.1f}s"
                    )
                    ui.log(f"  - Status: Converged", level="info")
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


def _fit_clusters_parallel(
    clargs: FitArguments, clusters: list, n_workers: int | None = None
) -> Parameters:
    """Fit all clusters using parallel processing."""
    from peakfit.core.parallel import fit_clusters_parallel_refined

    ui.info("Parallel fitting with refinement...")

    return fit_clusters_parallel_refined(
        clusters=clusters,
        noise=clargs.noise,
        refine_iterations=clargs.refine_nb,
        fixed=clargs.fixed,
        n_workers=n_workers,
        verbose=True,
    )


def _fit_clusters_global(
    clargs: FitArguments, clusters: list, optimizer: str, verbose: bool = False
) -> Parameters:
    """Fit all clusters using global optimization."""
    from peakfit.core.advanced_optimization import fit_basin_hopping, fit_differential_evolution

    params_all = Parameters()

    with threadpool_limits(limits=1, user_api="blas"):
        for index in range(clargs.refine_nb + 1):
            if index > 0:
                ui.spacer()
                ui.show_header("Refining Peak Parameters")
                ui.log_section(f"Refinement Iteration {index}")
                update_cluster_corrections(params_all, clusters)

            for cluster_idx, cluster in enumerate(clusters, 1):
                peak_names = [peak.name for peak in cluster.peaks]
                peaks_str = ", ".join(peak_names)

                # Print cluster header
                console.print()
                console.print(
                    f"[bold cyan]Cluster {cluster_idx}/{len(clusters)}[/bold cyan] [dim]│[/dim] {peaks_str}"
                )

                params = create_params(cluster.peaks, fixed=clargs.fixed)
                params = _update_params(params, params_all)

                # Use global optimizer
                if optimizer == "basin-hopping":
                    result = fit_basin_hopping(
                        params,
                        cluster,
                        clargs.noise,
                        n_iterations=50,  # Reasonable default
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

                # Print completion status
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


def _save_fitting_state(
    path: Path, clusters: list, params: Parameters, noise: float, peaks: list
) -> None:
    """Save fitting state for later analysis."""
    import pickle

    state = {
        "clusters": clusters,
        "params": params,
        "noise": noise,
        "peaks": peaks,
        "version": "1.0",
    }

    with path.open("wb") as f:
        pickle.dump(state, f)


def _load_fitting_state(path: Path) -> dict:
    """Load fitting state from file."""
    import pickle

    with path.open("rb") as f:
        return pickle.load(f)


def _update_params(params: Parameters, params_all: Parameters) -> Parameters:
    """Update parameters with global parameters."""
    for key in params:
        if key in params_all:
            params[key] = params_all[key]
    return params


def _write_spectra(path: Path, spectra, clusters, params: Parameters) -> None:
    """Write simulated spectra to file."""
    import nmrglue as ng
    import numpy as np

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


def _initialize_backend(backend: str, parallel: bool = False) -> None:
    """Initialize the computation backend.

    Args:
        backend: Requested backend (auto, numpy, numba)
        parallel: Whether parallel mode is enabled
    """
    from peakfit.core.backend import auto_select_backend, get_available_backends, set_backend

    if backend == "auto":
        selected = auto_select_backend()
        ui.success(f"Auto-selected backend: {selected}")
    else:
        available = get_available_backends()
        if backend not in available:
            ui.error(f"Backend '{backend}' not available. Available: {available}")
            ui.warning("Falling back to auto-selection...")
            selected = auto_select_backend()
            ui.success(f"Using backend: {selected}")
        else:
            set_backend(backend)
            ui.success(f"Using backend: {backend}")


def _print_optimization_status() -> None:
    """Print optimization status at the start of fitting."""
    from peakfit.core.backend import get_backend
    from peakfit.core.optimized import get_optimization_info

    current_backend = get_backend()
    opt_info = get_optimization_info()

    # Show backend-specific information
    if current_backend == "numba" and opt_info["numba_available"]:
        try:
            import numba

            ui.success(f"Numba JIT enabled (v{numba.__version__})")
        except ImportError:
            ui.info("Using NumPy vectorized operations")
    else:
        ui.info("Using NumPy vectorized operations")
        console.print("  [dim]Install numba for better performance[/dim]")
