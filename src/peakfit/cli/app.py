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
    workers: Annotated[
        int | None,
        typer.Option(
            "--workers",
            "-w",
            help="Number of parallel workers (default: number of CPUs)",
            min=1,
        ),
    ] = None,
    backend: Annotated[
        str,
        typer.Option(
            "--backend",
            "-b",
            help="Computation backend: auto, numpy, numba",
        ),
    ] = "auto",
    optimizer: Annotated[
        str,
        typer.Option(
            "--optimizer",
            help="Optimization algorithm: leastsq (fast), basin-hopping (global), "
            "differential-evolution (global)",
        ),
    ] = "leastsq",
    save_state: Annotated[
        bool,
        typer.Option(
            "--save-state/--no-save-state",
            help="Save fitting state for later analysis (enables 'peakfit analyze' command)",
        ),
    ] = True,
) -> None:
    """Fit lineshapes to peaks in pseudo-3D NMR spectrum.

    Example:
        peakfit fit spectrum.ft2 peaks.list --output results --refine 2

    For difficult peaks with overlaps, use global optimization:
        peakfit fit spectrum.ft2 peaks.list --optimizer basin-hopping

    To perform uncertainty analysis later:
        peakfit fit spectrum.ft2 peaks.list --save-state
        peakfit analyze mcmc Fits/  # Compute MCMC uncertainties
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
        n_workers=workers,
        backend=backend,
        optimizer=optimizer,
        save_state=save_state,
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
    All parameters are documented with inline comments explaining their purpose.

    Examples:
      Create default config:
        $ peakfit init

      Create config with custom name:
        $ peakfit init my_analysis.toml

      Overwrite existing config:
        $ peakfit init --force
    """
    from peakfit.messages import print_next_steps, print_success_message

    if path.exists() and not force:
        console.print(f"[red]✗ Error:[/red] File already exists: [yellow]{path}[/]")
        console.print("[dim]Use[/] [cyan]--force[/] [dim]to overwrite[/]")
        raise typer.Exit(1)

    config_content = generate_default_config()
    path.write_text(config_content)

    # Enhanced success message with details
    print_success_message(f"Created configuration file: {path}")

    console.print("\n[bold cyan]📄 Configuration includes:[/]")
    console.print("  • [green]Fitting parameters[/] (optimizer, lineshape, tolerances)")
    console.print("  • [green]Clustering settings[/] (algorithm, thresholds)")
    console.print("  • [green]Output preferences[/] (formats, directories)")
    console.print("  • [green]Advanced options[/] (parallel processing, backends)")

    # Suggest next steps
    print_next_steps([
        f"Review and customize: [cyan]{path}[/]",
        f"Run fitting: [cyan]peakfit fit spectrum.ft2 peaks.list --config {path}[/]",
        "Documentation: [cyan]https://github.com/gbouvignies/PeakFit[/]",
    ])


# Create a subapp for plot commands
plot_app = typer.Typer(help="Generate plots from fitting results", no_args_is_help=True)
app.add_typer(plot_app, name="plot")


@plot_app.command("intensity")
def plot_intensity(
    results: Annotated[
        Path,
        typer.Argument(
            help="Path to results directory or result file",
            exists=True,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output PDF file (default: intensity_profiles.pdf)",
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
) -> None:
    """Plot intensity profiles vs. plane index.

    Creates plots showing peak intensity decay/buildup across all planes in
    pseudo-3D spectra. Useful for visualizing CEST, CPMG, or T1/T2 relaxation data.

    Examples:
      Save all plots to PDF:
        $ peakfit plot intensity Fits/ --output intensity.pdf

      Interactive display (first 10 plots only):
        $ peakfit plot intensity Fits/ --show

      Plot single result file:
        $ peakfit plot intensity Fits/A45N-HN.out --show
    """
    from peakfit.cli.plot_command import plot_intensity_profiles

    plot_intensity_profiles(results, output, show)


@plot_app.command("cest")
def plot_cest(
    results: Annotated[
        Path,
        typer.Argument(
            help="Path to results directory or result file",
            exists=True,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output PDF file (default: cest_profiles.pdf)",
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
    ref: Annotated[
        list[int] | None,
        typer.Option(
            "--ref",
            "-r",
            help="Reference point indices (default: auto-detect using |offset| >= 10 kHz)",
        ),
    ] = None,
) -> None:
    """Plot CEST profiles (normalized intensity vs. B1 offset).

    Chemical Exchange Saturation Transfer (CEST) profiles show normalized peak
    intensities as a function of B1 offset frequency. Reference points (off-resonance)
    are used for normalization.

    By default, reference points are auto-detected as |offset| >= 10 kHz.
    Use --ref to manually specify reference point indices.

    Examples:
      Auto-detect reference points:
        $ peakfit plot cest Fits/ --output cest.pdf

      Manual reference selection (indices 0, 1, 2):
        $ peakfit plot cest Fits/ --ref 0 1 2

      Interactive display (first 10 plots):
        $ peakfit plot cest Fits/ --show

      Combine save and display:
        $ peakfit plot cest Fits/ --ref 0 1 --output my_cest.pdf --show
    """
    from peakfit.cli.plot_command import plot_cest_profiles

    plot_cest_profiles(results, output, show, ref or [-1])


@plot_app.command("cpmg")
def plot_cpmg(
    results: Annotated[
        Path,
        typer.Argument(
            help="Path to results directory or result file",
            exists=True,
            resolve_path=True,
        ),
    ],
    time_t2: Annotated[
        float,
        typer.Option(
            "--time-t2",
            "-t",
            help="T2 relaxation time in seconds (required)",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output PDF file (default: cpmg_profiles.pdf)",
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
) -> None:
    """Plot CPMG relaxation dispersion (R2eff vs. νCPMG).

    Carr-Purcell-Meiboom-Gill (CPMG) relaxation dispersion experiments probe
    microsecond-millisecond dynamics. This command converts cycle counts to
    CPMG frequencies (νCPMG) and intensities to effective relaxation rates (R2eff).

    The --time-t2 parameter is the constant time delay in the CPMG block (in seconds).
    Common values: 0.02-0.06s for backbone amides.

    Examples:
      Standard CPMG with T2 = 40ms:
        $ peakfit plot cpmg Fits/ --time-t2 0.04

      Save to custom file:
        $ peakfit plot cpmg Fits/ --time-t2 0.04 --output my_cpmg.pdf

      With interactive display (first 10):
        $ peakfit plot cpmg Fits/ --time-t2 0.04 --show

      Different T2 time (60ms):
        $ peakfit plot cpmg Fits/ --time-t2 0.06 --output cpmg_60ms.pdf
    """
    from peakfit.cli.plot_command import plot_cpmg_profiles

    plot_cpmg_profiles(results, output, show, time_t2)


@plot_app.command("spectra")
def plot_spectra(
    results: Annotated[
        Path,
        typer.Argument(
            help="Path to results directory",
            exists=True,
            resolve_path=True,
        ),
    ],
    spectrum: Annotated[
        Path,
        typer.Option(
            "--spectrum",
            "-s",
            help="Path to experimental spectrum for overlay (required)",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
) -> None:
    """Launch interactive spectra viewer (PyQt5).

    Opens a graphical interface showing experimental and simulated spectra
    side-by-side. Allows interactive plane selection, zooming, and comparison
    of fit quality across all planes.

    Requires PyQt5 to be installed. Install with: pip install PyQt5

    Examples:
      Basic usage:
        $ peakfit plot spectra Fits/ --spectrum data.ft2

      Using relative paths:
        $ peakfit plot spectra ./results --spectrum ../data/spectrum.ft2

      Full path specification:
        $ peakfit plot spectra /path/to/Fits --spectrum /path/to/spectrum.ft2
    """
    from peakfit.cli.plot_command import plot_spectra_viewer

    plot_spectra_viewer(results, spectrum)


@app.command()
def info(
    benchmark: Annotated[
        bool,
        typer.Option(
            "--benchmark",
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
    from peakfit.core.backend import get_available_backends, get_best_backend
    from peakfit.core.optimized import get_optimization_info

    console.print("[bold]PeakFit System Information[/bold]\n")

    # Version info
    console.print(f"[green]PeakFit version:[/green] {__version__}")
    console.print(f"[green]Python version:[/green] {sys.version}")
    console.print(f"[green]NumPy version:[/green] {np.__version__}")

    # Backend status
    available_backends = get_available_backends()
    best_backend = get_best_backend()

    console.print("\n[bold]Computation Backends:[/bold]")
    console.print(f"[green]Available:[/green] {', '.join(available_backends)}")
    console.print(f"[green]Recommended:[/green] {best_backend}")

    # Numba status
    opt_info = get_optimization_info()
    numba_available = opt_info["numba_available"]

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
    console.print("\n  [cyan]Gaussian:[/cyan]")
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
    console.print("\n  [cyan]Lorentzian:[/cyan]")
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
    console.print("\n  [cyan]Pseudo-Voigt:[/cyan]")
    console.print(f"    NumPy:     {numpy_time:.3f}s")
    console.print(f"    Optimized: {jit_time:.3f}s")
    if numba_available:
        console.print(f"    [green]Speedup: {speedup:.1f}x[/green]")


@app.command()
def analyze(
    method: Annotated[
        str,
        typer.Argument(
            help="Analysis method: mcmc, profile, correlation",
        ),
    ],
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
            help="Parameter name (required for 'profile' method)",
        ),
    ] = None,
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
            help="Number of MCMC walkers",
            min=4,
        ),
    ] = 32,
    n_steps: Annotated[
        int,
        typer.Option(
            "--steps",
            help="Number of MCMC steps",
            min=100,
        ),
    ] = 1000,
    burn_in: Annotated[
        int,
        typer.Option(
            "--burn-in",
            help="MCMC burn-in steps",
            min=0,
        ),
    ] = 200,
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
    """Analyze fitting uncertainties using advanced methods.

    MCMC sampling provides full posterior distributions:
        peakfit analyze mcmc Fits/
        peakfit analyze mcmc Fits/ --walkers 64 --steps 2000

    Profile likelihood gives accurate confidence intervals:
        peakfit analyze profile Fits/ --param peak1_x0
        peakfit analyze profile Fits/ --param peak1_x_fwhm --plot

    Parameter correlation analysis:
        peakfit analyze correlation Fits/
    """
    from peakfit.cli.analyze_command import run_correlation, run_mcmc, run_profile_likelihood

    valid_methods = ["mcmc", "profile", "correlation"]
    if method not in valid_methods:
        console.print(f"[red]Invalid method:[/red] {method}")
        console.print(f"[yellow]Valid methods:[/yellow] {', '.join(valid_methods)}")
        raise typer.Exit(1)

    if method == "mcmc":
        run_mcmc(
            results_dir=results,
            n_walkers=n_walkers,
            n_steps=n_steps,
            burn_in=burn_in,
            peaks=peaks,
            output_file=output,
        )
    elif method == "profile":
        if param is None:
            console.print("[red]Error:[/red] --param required for profile method")
            console.print("Example: peakfit analyze profile Fits/ --param peak1_x0")
            raise typer.Exit(1)
        run_profile_likelihood(
            results_dir=results,
            param_name=param,
            n_points=n_points,
            confidence_level=confidence,
            plot=plot,
            output_file=output,
        )
    elif method == "correlation":
        run_correlation(
            results_dir=results,
            output_file=output,
        )


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

    from peakfit.cli.fit_command import FitArguments
    from peakfit.clustering import create_clusters
    from peakfit.core.fast_fit import fit_clusters_fast
    from peakfit.core.parallel import fit_clusters_parallel_refined
    from peakfit.noise import prepare_noise_level
    from peakfit.peaklist import read_list
    from peakfit.spectra import get_shape_names, read_spectra

    console.print("[bold]PeakFit Performance Benchmark[/bold]\n")

    # Load data
    with console.status("[yellow]Loading spectrum..."):
        # Create args for loading
        clargs = FitArguments(
            path_spectra=spectrum,
            path_z_values=z_values,
            path_list=peaklist,
            exclude=[],
            noise=0.0,
            contour_level=None,
            fixed=False,
            jx=False,
            phx=False,
            phy=False,
            pvoigt=False,
            lorentzian=False,
            gaussian=False,
        )

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
    for _i in range(iterations):
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
        for _i in range(iterations):
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
            console.print(f"  Use [green]--parallel[/green] for {speedup:.1f}x speedup")
        else:
            console.print("  Sequential fitting is optimal (parallel overhead exceeds benefit)")
    else:
        console.print("\n[yellow]Note:[/yellow] Only 1 cluster, parallel comparison skipped")
        console.print("\n[bold]Recommendation:[/bold]")
        console.print("  Sequential fitting is optimal for single cluster")


if __name__ == "__main__":
    app()
