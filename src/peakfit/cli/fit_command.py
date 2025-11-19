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
from peakfit.ui import PeakFitUI as ui
from peakfit.ui import console
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
    # Show banner based on verbosity
    ui.show_banner(verbose)

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

    # Load data
    ui.show_header("Loading Data")
    ui.action("Reading spectrum...")
    with console.status("[bold yellow]Processing..."):
        spectra = read_spectra(clargs.path_spectra, clargs.path_z_values, clargs.exclude)

    ui.success(f"Loaded spectrum: {spectrum_path.name}")
    ui.bullet(f"Shape: {spectra.data.shape}", style="default")
    ui.bullet(f"Z-values: {len(spectra.z_values)} planes", style="default")

    # Estimate noise
    ui.spacer()
    clargs.noise = prepare_noise_level(clargs, spectra)
    ui.success(f"Noise level: {clargs.noise:.2f}")

    # Determine lineshape
    shape_names = get_shape_names(clargs, spectra)
    ui.bullet(f"Lineshapes: {shape_names}", style="default")

    # Read peak list
    ui.spacer()
    ui.action("Reading peak list...")
    peaks = read_list(spectra, shape_names, clargs)
    ui.success(f"Loaded {len(peaks)} peaks")

    # Cluster peaks
    ui.spacer()
    clargs.contour_level = clargs.contour_level or 5.0 * clargs.noise
    ui.bullet(f"Contour level: {clargs.contour_level:.2f}", style="default")

    clusters = create_clusters(spectra, peaks, clargs.contour_level)
    ui.success(f"Created {len(clusters)} clusters")

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
    if optimizer != "leastsq":
        ui.info(f"Using {optimizer} optimizer...")
        params = _fit_clusters_global(clargs, clusters, optimizer)
    elif parallel and len(clusters) > 1:
        ui.info("Using parallel fitting...")
        params = _fit_clusters_parallel(clargs, clusters, n_workers)
    else:
        params = _fit_clusters(clargs, clusters)

    # Write outputs
    ui.show_header("Saving Results")
    config.output.directory.mkdir(parents=True, exist_ok=True)

    write_profiles(config.output.directory, spectra.z_values, clusters, params, clargs)
    ui.success(f"Profiles written")
    ui.bullet(f"{config.output.directory}/*.out", style="default")

    if config.output.save_html_report:
        ui.export_html(config.output.directory / "logs.html")
        ui.success("HTML report written")
        ui.bullet(f"{config.output.directory / 'logs.html'}", style="default")

    write_shifts(peaks, params, config.output.directory / "shifts.list")
    ui.success("Shifts written")
    ui.bullet(f"{config.output.directory / 'shifts.list'}", style="default")

    if config.output.save_simulated:
        _write_spectra(config.output.directory, spectra, clusters, params)
        ui.success("Simulated spectrum written")

    # Save fitting state for later analysis
    if save_state:
        ui.spacer()
        state_file = config.output.directory / ".peakfit_state.pkl"
        _save_fitting_state(state_file, clusters, params, clargs.noise, peaks)
        ui.success("Fitting state saved")
        ui.bullet(f"{state_file}", style="default")
        ui.bullet("Use 'peakfit analyze' to compute uncertainties", style="default")

    ui.spacer()
    console.print("[bold green]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]")
    console.print("[bold green]✓ Fitting complete![/]")
    console.print("[bold green]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]")
    ui.spacer()


def _residual_wrapper(x: np.ndarray, params: Parameters, cluster, noise: float) -> np.ndarray:
    """Wrapper to convert array to Parameters for residual calculation."""
    vary_names = params.get_vary_names()
    for i, name in enumerate(vary_names):
        params[name].value = x[i]
    return residuals(params, cluster, noise)


def _fit_clusters(clargs: FitArguments, clusters: list) -> Parameters:
    """Fit all clusters and return parameters."""
    from rich.live import Live
    from rich.console import Group
    from rich.text import Text

    params_all = Parameters()

    # Use threadpoolctl to limit BLAS threads at runtime
    # This prevents OpenBLAS/MKL from spawning threads that cause massive overhead
    # (e.g., 3171% CPU usage -> 99% CPU usage, 671s -> 8s CPU time)
    with threadpool_limits(limits=1, user_api="blas"):
        for index in range(clargs.refine_nb + 1):
            if index > 0:
                ui.spacer()
                ui.action(f"Refining peak parameters ({index}/{clargs.refine_nb})...")
                update_cluster_corrections(params_all, clusters)

            previous_result = None
            previous_cluster_info = None
            cluster_results = []  # Track all cluster results for HTML logging

            # Keep Live display open for entire cluster loop
            with Live(console=console, refresh_per_second=4, transient=False) as live:
                for cluster_idx, cluster in enumerate(clusters, 1):
                    peak_names = [peak.name for peak in cluster.peaks]

                    # Create current cluster panel
                    current_panel = ui.create_cluster_status(
                        cluster_idx, len(clusters), peak_names, status="fitting"
                    )

                    # Build display with previous result (if exists) and current cluster
                    if previous_result is not None and previous_cluster_info is not None:
                        prev_idx, prev_peaks, prev_res = previous_cluster_info
                        prev_panel = ui.create_cluster_status(
                            prev_idx, len(clusters), prev_peaks, status="done", result=prev_res
                        )
                        # Show both previous and current with spacing
                        display_group = Group(prev_panel, Text(""), current_panel)
                    else:
                        # First cluster, show only current
                        display_group = current_panel

                    # Update live display
                    live.update(display_group)

                    params = create_params(cluster.peaks, fixed=clargs.fixed)
                    params = _update_params(params, params_all)

                    # Get varying parameters
                    vary_names = params.get_vary_names()
                    x0 = params.get_vary_values()
                    bounds_lower = np.array([params[name].min for name in vary_names])
                    bounds_upper = np.array([params[name].max for name in vary_names])

                    # Run optimization with scipy.optimize.least_squares
                    result = least_squares(
                        _residual_wrapper,
                        x0,
                        args=(params, cluster, clargs.noise),
                        bounds=(bounds_lower, bounds_upper),
                        ftol=1e-7,
                        xtol=1e-7,
                        max_nfev=1000,
                        verbose=2,  # Show iteration progress
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

                    # Update display to show completion
                    done_panel = ui.create_cluster_status(
                        cluster_idx, len(clusters), peak_names, status="done", result=result
                    )
                    if previous_result is not None and previous_cluster_info is not None:
                        prev_idx, prev_peaks, prev_res = previous_cluster_info
                        prev_panel = ui.create_cluster_status(
                            prev_idx, len(clusters), prev_peaks, status="done", result=prev_res
                        )
                        # Show both previous and current (both done)
                        display_group = Group(prev_panel, Text(""), done_panel)
                    else:
                        # First cluster, show only current
                        display_group = done_panel
                    live.update(display_group)

                    # Store this result as "previous" for next iteration
                    previous_result = result
                    previous_cluster_info = (cluster_idx, peak_names, result)

                    # Track cluster result for HTML logging
                    cluster_results.append((cluster_idx, peak_names, result))

                    params_all.update(params)

            # After Live context exits, print cluster summaries for HTML logging
            ui.spacer()
            ui.show_subheader("Cluster Fitting Summary")
            for cluster_idx, peak_names, result in cluster_results:
                peaks_str = ", ".join(peak_names)
                ui.bullet(
                    f"Cluster {cluster_idx}/{len(clusters)} │ Peaks: {peaks_str} │ "
                    f"{'✓' if result.success else '✗'} │ "
                    f"Cost: {result.cost:.2e} │ nfev: {result.nfev}",
                    style="success" if result.success else "error"
                )

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


def _fit_clusters_global(clargs: FitArguments, clusters: list, optimizer: str) -> Parameters:
    """Fit all clusters using global optimization."""
    from rich.live import Live
    from rich.console import Group
    from rich.text import Text
    from peakfit.core.advanced_optimization import fit_basin_hopping, fit_differential_evolution

    params_all = Parameters()

    with threadpool_limits(limits=1, user_api="blas"):
        for index in range(clargs.refine_nb + 1):
            if index > 0:
                ui.spacer()
                ui.action(f"Refining peak parameters ({index}/{clargs.refine_nb})...")
                update_cluster_corrections(params_all, clusters)

            previous_result = None
            previous_cluster_info = None
            cluster_results = []  # Track all cluster results for HTML logging

            # Keep Live display open for entire cluster loop
            with Live(console=console, refresh_per_second=4, transient=False) as live:
                for cluster_idx, cluster in enumerate(clusters, 1):
                    peak_names = [peak.name for peak in cluster.peaks]

                    # Create current cluster panel
                    current_panel = ui.create_cluster_status(
                        cluster_idx, len(clusters), peak_names, status="optimizing"
                    )

                    # Build display with previous result (if exists) and current cluster
                    if previous_result is not None and previous_cluster_info is not None:
                        prev_idx, prev_peaks, prev_res = previous_cluster_info
                        prev_panel = ui.create_cluster_status(
                            prev_idx, len(clusters), prev_peaks, status="done", result=prev_res
                        )
                        # Show both previous and current with spacing
                        display_group = Group(prev_panel, Text(""), current_panel)
                    else:
                        # First cluster, show only current
                        display_group = current_panel

                    # Update live display
                    live.update(display_group)

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

                    # Update display to show completion
                    done_panel = ui.create_cluster_status(
                        cluster_idx, len(clusters), peak_names, status="done", result=result
                    )
                    if previous_result is not None and previous_cluster_info is not None:
                        prev_idx, prev_peaks, prev_res = previous_cluster_info
                        prev_panel = ui.create_cluster_status(
                            prev_idx, len(clusters), prev_peaks, status="done", result=prev_res
                        )
                        # Show both previous and current (both done)
                        display_group = Group(prev_panel, Text(""), done_panel)
                    else:
                        # First cluster, show only current
                        display_group = done_panel
                    live.update(display_group)

                    # Store this result as "previous" for next iteration
                    previous_result = result
                    previous_cluster_info = (cluster_idx, peak_names, result)

                    # Track cluster result for HTML logging
                    cluster_results.append((cluster_idx, peak_names, result))

                    params_all.update(result.params)

            # After Live context exits, print cluster summaries for HTML logging
            ui.spacer()
            ui.show_subheader("Cluster Fitting Summary")
            for cluster_idx, peak_names, result in cluster_results:
                peaks_str = ", ".join(peak_names)
                success = result.success if hasattr(result, "success") else True
                chisqr = result.chisqr if hasattr(result, "chisqr") else result.cost
                nfev = result.nfev if hasattr(result, "nfev") else "N/A"
                ui.bullet(
                    f"Cluster {cluster_idx}/{len(clusters)} │ Peaks: {peaks_str} │ "
                    f"{'✓' if success else '✗'} │ "
                    f"Cost: {chisqr:.2e} │ nfev: {nfev}",
                    style="success" if success else "error"
                )

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
