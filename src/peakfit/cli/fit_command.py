"""Implementation of the fit command."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rich.console import Console
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from peakfit.clustering import create_clusters
from peakfit.computing import residuals, simulate_data, update_cluster_corrections
from peakfit.core.fitting import Parameters
from peakfit.core.models import PeakFitConfig
from peakfit.messages import (
    export_html,
    print_fit_report,
    print_fitting,
    print_logo,
    print_peaks,
    print_refining,
    print_writing_spectra,
)
from peakfit.noise import prepare_noise_level
from peakfit.peak import create_params
from peakfit.peaklist import read_list
from peakfit.spectra import get_shape_names, read_spectra
from peakfit.writing import write_profiles, write_shifts

console = Console()


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
) -> None:
    """Run the fitting process.

    Args:
        spectrum_path: Path to NMRPipe spectrum file.
        peaklist_path: Path to peak list file.
        z_values_path: Optional path to Z-values file.
        config: Configuration object.
        parallel: Whether to use parallel processing.
        n_workers: Number of parallel workers.
        backend: Computation backend (auto, numpy, numba, jax).
        optimizer: Optimization algorithm (leastsq, basin-hopping, differential-evolution).
        save_state: Whether to save fitting state for later analysis.
    """
    print_logo()

    # Initialize computation backend
    _initialize_backend(backend, parallel=parallel)

    # Show optimization status
    _print_optimization_status()

    # Validate optimizer choice
    valid_optimizers = ["leastsq", "basin-hopping", "differential-evolution"]
    if optimizer not in valid_optimizers:
        console.print(f"[red]Invalid optimizer:[/red] {optimizer}")
        console.print(f"[yellow]Valid options:[/yellow] {', '.join(valid_optimizers)}")
        raise SystemExit(1)

    if optimizer != "leastsq":
        console.print(f"[yellow]Using global optimizer:[/yellow] {optimizer}")
        console.print("  [dim]This may take significantly longer than standard fitting[/dim]")

    # Convert to legacy args for compatibility with existing modules
    clargs = config_to_fit_args(config, spectrum_path, peaklist_path, z_values_path)

    # Load data
    with console.status("[bold yellow]Loading spectrum..."):
        spectra = read_spectra(clargs.path_spectra, clargs.path_z_values, clargs.exclude)

    console.print(f"[green]Loaded spectrum:[/green] {spectrum_path.name}")
    console.print(f"  Shape: {spectra.data.shape}")
    console.print(f"  Z-values: {len(spectra.z_values)} planes")

    # Estimate noise
    clargs.noise = prepare_noise_level(clargs, spectra)
    console.print(f"[green]Noise level:[/green] {clargs.noise:.2f}")

    # Determine lineshape
    shape_names = get_shape_names(clargs, spectra)
    console.print(f"[green]Lineshapes:[/green] {shape_names}")

    # Read peak list
    peaks = read_list(spectra, shape_names, clargs)
    console.print(f"[green]Loaded peaks:[/green] {len(peaks)} peaks")

    # Cluster peaks
    clargs.contour_level = clargs.contour_level or 5.0 * clargs.noise
    console.print(f"[green]Contour level:[/green] {clargs.contour_level:.2f}")

    clusters = create_clusters(spectra, peaks, clargs.contour_level)
    console.print(f"[green]Created clusters:[/green] {len(clusters)} clusters")

    # Fit clusters - choose method based on flags
    if optimizer != "leastsq":
        console.print(f"[yellow]Using {optimizer} optimizer...[/yellow]")
        params = _fit_clusters_global(clargs, clusters, optimizer)
    elif parallel and len(clusters) > 1:
        console.print("[yellow]Using parallel fitting...[/yellow]")
        params = _fit_clusters_parallel(clargs, clusters, n_workers)
    else:
        params = _fit_clusters(clargs, clusters)

    # Write outputs
    config.output.directory.mkdir(parents=True, exist_ok=True)

    write_profiles(config.output.directory, spectra.z_values, clusters, params, clargs)
    console.print(f"[green]Written profiles to:[/green] {config.output.directory}")

    if config.output.save_html_report:
        export_html(config.output.directory / "logs.html")
        console.print(f"[green]Written HTML report:[/green] {config.output.directory / 'logs.html'}")

    write_shifts(peaks, params, config.output.directory / "shifts.list")
    console.print(f"[green]Written shifts:[/green] {config.output.directory / 'shifts.list'}")

    if config.output.save_simulated:
        _write_spectra(config.output.directory, spectra, clusters, params)
        console.print("[green]Written simulated spectrum[/green]")

    # Save fitting state for later analysis
    if save_state:
        state_file = config.output.directory / ".peakfit_state.pkl"
        _save_fitting_state(state_file, clusters, params, clargs.noise, peaks)
        console.print(f"[green]Saved fitting state:[/green] {state_file}")
        console.print("  [dim]Use 'peakfit analyze' to compute uncertainties[/dim]")

    console.print("\n[bold green]Fitting complete![/bold green]")


def _residual_wrapper(
    x: np.ndarray, params: Parameters, cluster, noise: float
) -> np.ndarray:
    """Wrapper to convert array to Parameters for residual calculation."""
    vary_names = params.get_vary_names()
    for i, name in enumerate(vary_names):
        params[name].value = x[i]
    return residuals(params, cluster, noise)


def _fit_clusters(clargs: FitArguments, clusters: list) -> Parameters:
    """Fit all clusters and return parameters."""
    print_fitting()
    params_all = Parameters()

    # Use threadpoolctl to limit BLAS threads at runtime
    # This prevents OpenBLAS/MKL from spawning threads that cause massive overhead
    # (e.g., 3171% CPU usage -> 99% CPU usage, 671s -> 8s CPU time)
    with threadpool_limits(limits=1, user_api="blas"):
        for index in range(clargs.refine_nb + 1):
            if index > 0:
                print_refining(index, clargs.refine_nb)
                update_cluster_corrections(params_all, clusters)
            for cluster in clusters:
                print_peaks(cluster.peaks)
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
                    verbose=2,
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

                print_fit_report(result)
                params_all.update(params)

    return params_all


def _fit_clusters_parallel(
    clargs: FitArguments, clusters: list, n_workers: int | None = None
) -> Parameters:
    """Fit all clusters using parallel processing."""
    from peakfit.core.parallel import fit_clusters_parallel_refined

    console.print("[yellow]Parallel fitting with refinement...[/yellow]")

    return fit_clusters_parallel_refined(
        clusters=clusters,
        noise=clargs.noise,
        refine_iterations=clargs.refine_nb,
        fixed=clargs.fixed,
        n_workers=n_workers,
        verbose=True,
    )



def _fit_clusters_global(
    clargs: FitArguments, clusters: list, optimizer: str
) -> Parameters:
    """Fit all clusters using global optimization."""
    from peakfit.core.advanced_optimization import (
        fit_basin_hopping,
        fit_differential_evolution,
    )

    print_fitting()
    params_all = Parameters()

    with threadpool_limits(limits=1, user_api="blas"):
        for index in range(clargs.refine_nb + 1):
            if index > 0:
                print_refining(index, clargs.refine_nb)
                update_cluster_corrections(params_all, clusters)

            for cluster in clusters:
                print_peaks(cluster.peaks)
                params = create_params(cluster.peaks, fixed=clargs.fixed)
                params = _update_params(params, params_all)

                # Use global optimizer
                console.print(f"  [dim]Running {optimizer}...[/dim]")
                if optimizer == "basin-hopping":
                    result = fit_basin_hopping(
                        params, cluster, clargs.noise,
                        n_iterations=50,  # Reasonable default
                        temperature=1.0,
                        step_size=0.5,
                    )
                elif optimizer == "differential-evolution":
                    result = fit_differential_evolution(
                        params, cluster, clargs.noise,
                        max_iterations=500,
                        population_size=15,
                        polish=True,
                    )
                else:
                    msg = f"Unknown optimizer: {optimizer}"
                    raise ValueError(msg)

                console.print(
                    f"  [dim]χ²={result.chisqr:.2f}, "
                    f"reduced χ²={result.redchi:.4f}, "
                    f"nfev={result.nfev}[/dim]"
                )

                if not result.success:
                    console.print(f"  [yellow]Warning: {result.message}[/yellow]")

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

    print_writing_spectra()

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
        backend: Requested backend (auto, numpy, numba, jax)
        parallel: Whether parallel mode is enabled
    """
    from peakfit.core.backend import (
        auto_select_backend,
        get_available_backends,
        set_backend,
    )

    # JAX is incompatible with multiprocessing due to GPU device conflicts
    # When parallel mode is enabled, force numba or numpy backend
    if parallel and backend in ("auto", "jax"):
        available = get_available_backends()
        if "numba" in available:
            backend = "numba"
            console.print(
                "[yellow]⚠ Parallel mode: Using numba backend (JAX incompatible with multiprocessing)[/yellow]"
            )
        else:
            backend = "numpy"
            console.print(
                "[yellow]⚠ Parallel mode: Using numpy backend (JAX incompatible with multiprocessing)[/yellow]"
            )

    if backend == "auto":
        selected = auto_select_backend()
        console.print(f"[green]✓ Auto-selected backend:[/green] {selected}")
    else:
        available = get_available_backends()
        if backend not in available:
            console.print(
                f"[red]Backend '{backend}' not available. Available: {available}[/red]"
            )
            console.print("[yellow]Falling back to auto-selection...[/yellow]")
            selected = auto_select_backend()
            console.print(f"[green]✓ Using backend:[/green] {selected}")
        else:
            set_backend(backend)
            console.print(f"[green]✓ Using backend:[/green] {backend}")


def _print_optimization_status() -> None:
    """Print optimization status at the start of fitting."""
    from peakfit.core.backend import get_backend
    from peakfit.core.optimized import get_optimization_info

    current_backend = get_backend()
    opt_info = get_optimization_info()

    # Show backend-specific information
    if current_backend == "jax":
        try:
            import jax
            console.print(f"[green]✓ JAX backend active[/green] (v{jax.__version__})")
            devices = jax.devices()
            device_info = ", ".join([str(d).split(":")[0] for d in devices])
            console.print(f"  [dim]Devices: {device_info}[/dim]")
        except ImportError:
            pass
    elif current_backend == "numba" and opt_info["numba_available"]:
        try:
            import numba
            console.print(f"[green]✓ Numba JIT enabled[/green] (v{numba.__version__})")
        except ImportError:
            console.print("[yellow]• Using NumPy vectorized operations[/yellow]")
    else:
        console.print("[yellow]• Using NumPy vectorized operations[/yellow]")
        console.print("  [dim]Install numba or jax for better performance[/dim]")
