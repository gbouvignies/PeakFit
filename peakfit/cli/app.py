"""Main Typer application for PeakFit."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from peakfit.cli.callbacks import version_callback
from peakfit.core.models import PeakFitConfig
from peakfit.io.config import generate_default_config, load_config

app = typer.Typer(
    name="peakfit",
    help="PeakFit - Lineshape fitting for pseudo-3D NMR spectra",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

console = Console()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """PeakFit - Modern lineshape fitting for pseudo-3D NMR spectra.

    Perform automated peak integration and lineshape analysis on NMR data.
    """


@app.command()
def fit(
    spectrum: Annotated[
        Path,
        typer.Argument(
            help="Path to NMRPipe spectrum file (.ft2, .ft3)",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    peaklist: Annotated[
        Path,
        typer.Argument(
            help="Path to peak list file (.list, .csv, .json, .xlsx)",
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
            help="Path to Z-dimension values file",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for results",
            file_okay=False,
            resolve_path=True,
        ),
    ] = Path("Fits"),
    config: Annotated[
        Path | None,
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
        str,
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
    parallel: Annotated[
        bool,
        typer.Option(
            "--parallel/--no-parallel",
            help="Enable parallel fitting of clusters",
        ),
    ] = False,
    fast: Annotated[
        bool,
        typer.Option(
            "--fast/--no-fast",
            help="Use fast scipy optimization (bypasses lmfit overhead)",
        ),
    ] = False,
    workers: Annotated[
        int | None,
        typer.Option(
            "--workers",
            "-w",
            help="Number of parallel workers (default: number of CPUs)",
            min=1,
        ),
    ] = None,
) -> None:
    """Fit lineshapes to peaks in pseudo-3D NMR spectrum.

    Example:
        peakfit fit spectrum.ft2 peaks.list --output results --refine 2
    """
    from peakfit.cli.fit_command import run_fit

    # Load config from file or create from CLI options
    if config is not None:
        fit_config = load_config(config)
        # Override with CLI options where explicitly set
        fit_config.output.directory = output
    else:
        fit_config = PeakFitConfig(
            fitting={
                "lineshape": lineshape,
                "refine_iterations": refine,
                "fix_positions": fixed,
                "fit_j_coupling": jx,
                "fit_phase_x": phx,
                "fit_phase_y": phy,
            },
            clustering={
                "contour_level": contour_level,
            },
            output={
                "directory": output,
            },
            noise_level=noise,
            exclude_planes=exclude or [],
        )

    run_fit(
        spectrum_path=spectrum,
        peaklist_path=peaklist,
        z_values_path=z_values,
        config=fit_config,
        parallel=parallel,
        fast=fast,
        n_workers=workers,
    )


@app.command()
def validate(
    spectrum: Annotated[
        Path,
        typer.Argument(
            help="Path to NMRPipe spectrum file",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    peaklist: Annotated[
        Path,
        typer.Argument(
            help="Path to peak list file",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
) -> None:
    """Validate input files before fitting.

    Check that spectrum and peak list files are readable and compatible.
    """
    from peakfit.cli.validate_command import run_validate

    run_validate(spectrum, peaklist)


@app.command()
def init(
    path: Annotated[
        Path,
        typer.Argument(
            help="Path for new configuration file",
            dir_okay=False,
            resolve_path=True,
        ),
    ] = Path("peakfit.toml"),
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite existing file",
        ),
    ] = False,
) -> None:
    """Generate a default configuration file.

    Creates a TOML configuration file with default settings that can be customized.
    """
    if path.exists() and not force:
        console.print(f"[red]Error:[/red] File already exists: {path}")
        console.print("Use --force to overwrite")
        raise typer.Exit(1)

    config_content = generate_default_config()
    path.write_text(config_content)
    console.print(f"[green]Created configuration file:[/green] {path}")


@app.command()
def plot(
    results: Annotated[
        Path,
        typer.Argument(
            help="Path to results directory or specific result file",
            exists=True,
            resolve_path=True,
        ),
    ],
    spectrum: Annotated[
        Path | None,
        typer.Option(
            "--spectrum",
            "-s",
            help="Path to original spectrum for overlay",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file for plots (PDF)",
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    show: Annotated[
        bool,
        typer.Option(
            "--show/--no-show",
            help="Display plots interactively",
        ),
    ] = False,
    plot_type: Annotated[
        str,
        typer.Option(
            "--type",
            "-t",
            help="Plot type: intensity, cest, cpmg, spectra",
        ),
    ] = "intensity",
) -> None:
    """Generate plots from fitting results.

    Create publication-quality figures showing fitted intensities, spectra overlays, etc.
    """
    from peakfit.cli.plot_command import run_plot

    run_plot(results, spectrum, output, show, plot_type)


@app.command()
def info(
    benchmark: Annotated[
        bool,
        typer.Option(
            "--benchmark",
            "-b",
            help="Run performance benchmark to measure speedup",
        ),
    ] = False,
) -> None:
    """Show system information and optimization status.

    Display details about available optimizations including Numba JIT compilation,
    parallel processing capabilities, and optionally run a benchmark.
    """
    import multiprocessing as mp
    import sys

    import numpy as np

    from peakfit import __version__
    from peakfit.core.optimized import check_numba_available, get_optimization_info

    console.print("[bold]PeakFit System Information[/bold]\n")

    # Version info
    console.print(f"[green]PeakFit version:[/green] {__version__}")
    console.print(f"[green]Python version:[/green] {sys.version}")
    console.print(f"[green]NumPy version:[/green] {np.__version__}")

    # Numba status
    opt_info = get_optimization_info()
    numba_available = opt_info["numba_available"]

    console.print(f"\n[bold]Optimization Status:[/bold]")
    if numba_available:
        try:
            import numba

            console.print(f"[green]✓ Numba JIT enabled:[/green] {numba.__version__}")
            console.print(f"  Optimized functions: {', '.join(opt_info['optimizations'])}")
        except ImportError:
            console.print("[yellow]✗ Numba not available[/yellow]")
    else:
        console.print("[yellow]✗ Numba JIT not available[/yellow]")
        console.print("  Install with: pip install numba")
        console.print("  Or: pip install peakfit[performance]")
        console.print(f"  Using: {', '.join(opt_info['optimizations'])}")

    # Parallel processing
    n_cpus = mp.cpu_count()
    console.print(f"\n[green]Parallel processing:[/green] {n_cpus} CPU cores available")

    # Benchmark
    if benchmark:
        console.print("\n[bold]Running Performance Benchmark...[/bold]")
        _run_benchmark(numba_available)


def _run_benchmark(numba_available: bool) -> None:
    """Run performance benchmark comparing optimized vs pure NumPy."""
    import time

    import numpy as np

    from peakfit.core.optimized import gaussian_jit, lorentzian_jit, pvoigt_jit

    # Pure NumPy implementations for comparison (avoid circular imports)
    def gaussian_numpy(dx: np.ndarray, fwhm: float) -> np.ndarray:
        c = 4.0 * np.log(2.0) / (fwhm * fwhm)
        return np.exp(-dx * dx * c)

    def lorentzian_numpy(dx: np.ndarray, fwhm: float) -> np.ndarray:
        half_width_sq = (0.5 * fwhm) ** 2
        return half_width_sq / (dx * dx + half_width_sq)

    def pvoigt_numpy(dx: np.ndarray, fwhm: float, eta: float) -> np.ndarray:
        return (1.0 - eta) * gaussian_numpy(dx, fwhm) + eta * lorentzian_numpy(dx, fwhm)

    # Test parameters
    n_iterations = 1000
    dx = np.linspace(-100, 100, 10001)  # Large array
    fwhm = 15.0
    eta = 0.5

    console.print(f"  Array size: {len(dx):,} points")
    console.print(f"  Iterations: {n_iterations:,}")

    # Warmup JIT (if available)
    if numba_available:
        _ = gaussian_jit(dx, fwhm)
        _ = lorentzian_jit(dx, fwhm)
        _ = pvoigt_jit(dx, fwhm, eta)

    # Benchmark Gaussian
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = gaussian_numpy(dx, fwhm)
    numpy_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = gaussian_jit(dx, fwhm)
    jit_time = time.perf_counter() - start

    speedup = numpy_time / jit_time if jit_time > 0 else 1.0
    console.print(f"\n  [cyan]Gaussian:[/cyan]")
    console.print(f"    NumPy:     {numpy_time:.3f}s")
    console.print(f"    Optimized: {jit_time:.3f}s")
    if numba_available:
        console.print(f"    [green]Speedup: {speedup:.1f}x[/green]")

    # Benchmark Lorentzian
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = lorentzian_numpy(dx, fwhm)
    numpy_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = lorentzian_jit(dx, fwhm)
    jit_time = time.perf_counter() - start

    speedup = numpy_time / jit_time if jit_time > 0 else 1.0
    console.print(f"\n  [cyan]Lorentzian:[/cyan]")
    console.print(f"    NumPy:     {numpy_time:.3f}s")
    console.print(f"    Optimized: {jit_time:.3f}s")
    if numba_available:
        console.print(f"    [green]Speedup: {speedup:.1f}x[/green]")

    # Benchmark Pseudo-Voigt
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = pvoigt_numpy(dx, fwhm, eta)
    numpy_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = pvoigt_jit(dx, fwhm, eta)
    jit_time = time.perf_counter() - start

    speedup = numpy_time / jit_time if jit_time > 0 else 1.0
    console.print(f"\n  [cyan]Pseudo-Voigt:[/cyan]")
    console.print(f"    NumPy:     {numpy_time:.3f}s")
    console.print(f"    Optimized: {jit_time:.3f}s")
    if numba_available:
        console.print(f"    [green]Speedup: {speedup:.1f}x[/green]")


@app.command()
def benchmark(
    spectrum: Annotated[
        Path,
        typer.Argument(
            help="Path to NMRPipe spectrum file",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    peaklist: Annotated[
        Path,
        typer.Argument(
            help="Path to peak list file",
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
            help="Path to Z-dimension values file",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    iterations: Annotated[
        int,
        typer.Option(
            "--iterations",
            "-n",
            help="Number of benchmark iterations",
            min=1,
            max=10,
        ),
    ] = 1,
) -> None:
    """Benchmark fitting performance with different methods.

    Compare standard lmfit, fast scipy, and parallel fitting approaches
    to determine the optimal method for your data.

    Example:
        peakfit benchmark spectrum.ft2 peaks.list --iterations 3
    """
    import time

    from peakfit.cli_legacy import Arguments as LegacyArguments
    from peakfit.clustering import create_clusters
    from peakfit.core.fast_fit import fit_clusters_fast
    from peakfit.core.models import PeakFitConfig
    from peakfit.core.parallel import fit_clusters_parallel_refined
    from peakfit.noise import prepare_noise_level
    from peakfit.peaklist import read_list
    from peakfit.spectra import get_shape_names, read_spectra

    console.print("[bold]PeakFit Performance Benchmark[/bold]\n")

    # Load data
    with console.status("[yellow]Loading spectrum..."):
        # Create minimal legacy args for loading
        clargs = LegacyArguments()
        clargs.path_spectra = spectrum
        clargs.path_z_values = z_values
        clargs.path_list = peaklist
        clargs.exclude = []
        clargs.noise = None
        clargs.contour_level = None
        clargs.fixed = False
        clargs.jx = False
        clargs.phx = False
        clargs.phy = False
        clargs.pvoigt = False
        clargs.lorentzian = False
        clargs.gaussian = False

        spectra = read_spectra(spectrum, z_values, [])

    console.print(f"[green]Loaded spectrum:[/green] {spectrum.name}")
    console.print(f"  Shape: {spectra.data.shape}")

    # Prepare fitting
    clargs.noise = prepare_noise_level(clargs, spectra)
    shape_names = get_shape_names(clargs, spectra)
    peaks = read_list(spectra, shape_names, clargs)
    clargs.contour_level = 5.0 * clargs.noise
    clusters = create_clusters(spectra, peaks, clargs.contour_level)

    console.print(f"[green]Noise level:[/green] {clargs.noise:.2f}")
    console.print(f"[green]Clusters:[/green] {len(clusters)}")
    console.print(f"[green]Total peaks:[/green] {len(peaks)}")

    console.print(f"\n[bold]Running benchmark ({iterations} iteration(s))...[/bold]\n")

    # Benchmark fast sequential
    times_fast = []
    for i in range(iterations):
        start = time.perf_counter()
        fit_clusters_fast(
            clusters=list(clusters),
            noise=clargs.noise,
            refine_iterations=0,  # No refinement for speed comparison
            fixed=False,
            verbose=False,
        )
        times_fast.append(time.perf_counter() - start)

    avg_fast = sum(times_fast) / len(times_fast)
    console.print("[cyan]Fast Sequential:[/cyan]")
    console.print(f"  Average time: {avg_fast:.3f}s")
    if len(times_fast) > 1:
        console.print(f"  Min: {min(times_fast):.3f}s, Max: {max(times_fast):.3f}s")

    # Benchmark parallel (if enough clusters)
    if len(clusters) > 1:
        import multiprocessing as mp

        n_workers = mp.cpu_count()

        times_parallel = []
        for i in range(iterations):
            start = time.perf_counter()
            fit_clusters_parallel_refined(
                clusters=clusters,
                noise=clargs.noise,
                refine_iterations=0,
                fixed=False,
                n_workers=n_workers,
                verbose=False,
            )
            times_parallel.append(time.perf_counter() - start)

        avg_parallel = sum(times_parallel) / len(times_parallel)
        console.print(f"\n[cyan]Parallel ({n_workers} workers):[/cyan]")
        console.print(f"  Average time: {avg_parallel:.3f}s")
        if len(times_parallel) > 1:
            console.print(f"  Min: {min(times_parallel):.3f}s, Max: {max(times_parallel):.3f}s")

        speedup = avg_fast / avg_parallel if avg_parallel > 0 else 1.0
        console.print(f"  [green]Speedup: {speedup:.2f}x[/green]")

        # Recommendation
        console.print("\n[bold]Recommendation:[/bold]")
        if speedup > 1.2:
            console.print(
                f"  Use [green]--parallel[/green] for {speedup:.1f}x speedup"
            )
        else:
            console.print("  Use [green]--fast[/green] (parallel overhead exceeds benefit)")
    else:
        console.print("\n[yellow]Note:[/yellow] Only 1 cluster, parallel comparison skipped")
        console.print("\n[bold]Recommendation:[/bold]")
        console.print("  Use [green]--fast[/green] for optimal performance")


if __name__ == "__main__":
    app()
