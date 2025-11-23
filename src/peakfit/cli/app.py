"""Main Typer application for PeakFit."""

from pathlib import Path
from typing import Annotated

import typer

from peakfit.cli.callbacks import version_callback
from peakfit.io.config import generate_default_config, load_config
from peakfit.models import PeakFitConfig
from peakfit.ui import PeakFitUI as ui, console

app = typer.Typer(
    name="peakfit",
    help="PeakFit - Lineshape fitting for pseudo-3D NMR spectra",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


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
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Show banner and verbose output",
        ),
    ] = False,
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
        optimizer=optimizer,
        save_state=save_state,
        verbose=verbose,
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
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Show banner and verbose output",
        ),
    ] = False,
) -> None:
    """Validate input files before fitting.

    Check that spectrum and peak list files are readable and compatible.
    """
    from peakfit.cli.validate_command import run_validate

    run_validate(spectrum, peaklist, verbose)


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
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Show banner and verbose output",
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
    # Show banner based on verbosity
    ui.show_banner(verbose)

    if path.exists() and not force:
        ui.error(f"File already exists: [path]{path}[/path]")
        ui.info("Use [code]--force[/code] to overwrite")
        raise typer.Exit(1)

    config_content = generate_default_config()
    path.write_text(config_content)

    # Enhanced success message with details
    ui.success(f"Created configuration file: [path]{path}[/path]")

    console.print("\n[bold cyan]📄 Configuration includes:[/]")
    console.print("  • [green]Fitting parameters[/] (optimizer, lineshape, tolerances)")
    console.print("  • [green]Clustering settings[/] (algorithm, thresholds)")
    console.print("  • [green]Output preferences[/] (formats, directories)")
    console.print("  • [green]Advanced options[/] (parallel processing, backends)")

    # Suggest next steps
    ui.print_next_steps(
        [
            f"Review and customize: [cyan]{path}[/]",
            f"Run fitting: [cyan]peakfit fit spectrum.ft2 peaks.list --config {path}[/]",
            "Documentation: [cyan]https://github.com/gbouvignies/PeakFit[/]",
        ]
    )


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
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Show banner and verbose output",
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

    plot_intensity_profiles(results, output, show, verbose)


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
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Show banner and verbose output",
        ),
    ] = False,
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

    plot_cest_profiles(results, output, show, ref or [-1], verbose)


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
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Show banner and verbose output",
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

    plot_cpmg_profiles(results, output, show, time_t2, verbose)


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
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Show banner and verbose output",
        ),
    ] = False,
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

    plot_spectra_viewer(results, spectrum, verbose)


@plot_app.command("diagnostics")
def plot_diagnostics(
    results: Annotated[
        Path,
        typer.Argument(
            help="Path to results directory from 'peakfit analyze mcmc'",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output PDF file (default: mcmc_diagnostics.pdf)",
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    peaks: Annotated[
        list[str] | None,
        typer.Option(
            "--peaks",
            help="Peak names to plot (default: all)",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Show banner and verbose output",
        ),
    ] = False,
) -> None:
    """Generate MCMC diagnostic plots (trace, corner, autocorrelation).

    Creates comprehensive diagnostic plots from MCMC sampling results to assess:
    - Chain convergence (trace plots)
    - Parameter correlations (corner plots)
    - Mixing efficiency (autocorrelation plots)

    This command requires MCMC results from 'peakfit analyze mcmc' with saved chain data.

    Examples:
      Generate diagnostics for all peaks:
        $ peakfit plot diagnostics Fits/ --output diagnostics.pdf

      Plot specific peaks only:
        $ peakfit plot diagnostics Fits/ --peaks 2N-H 5L-H

      Quick diagnostics with default output:
        $ peakfit plot diagnostics Fits/
    """
    from peakfit.cli.plot_command import plot_mcmc_diagnostics

    plot_mcmc_diagnostics(results, output, peaks, verbose)


@app.command()
def info() -> None:
    """Show system information.

    Display details about the PeakFit installation and system capabilities.
    """
    import sys

    import numpy as np

    from peakfit import __version__

    console.print("[bold]PeakFit System Information[/bold]\n")

    # Version info
    console.print(f"[green]PeakFit version:[/green] {__version__}")
    console.print(f"[green]Python version:[/green] {sys.version}")
    console.print(f"[green]NumPy version:[/green] {np.__version__}")


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
            help="Parameter to profile: exact name, peak name, or parameter type (default: all)",
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
            "--chains",
            help="Number of MCMC walkers/chains",
            min=4,
        ),
    ] = 32,
    n_steps: Annotated[
        int,
        typer.Option(
            "--steps",
            "--samples",
            help="Number of MCMC steps/samples per walker",
            min=100,
        ),
    ] = 1000,
    burn_in: Annotated[
        int | None,
        typer.Option(
            "--burn-in",
            help="MCMC burn-in steps (manual override; default: auto-determined using R-hat)",
            min=0,
        ),
    ] = None,
    auto_burnin: Annotated[
        bool,
        typer.Option(
            "--auto-burnin/--no-auto-burnin",
            help="Automatically determine burn-in using R-hat convergence monitoring",
        ),
    ] = True,
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
        peakfit analyze mcmc Fits/ --chains 64 --samples 2000
        peakfit analyze mcmc Fits/ --walkers 64 --steps 2000  # Alternative syntax

    Profile likelihood gives accurate confidence intervals:
        peakfit analyze profile Fits/                    # All parameters
        peakfit analyze profile Fits/ --param 2N-H       # All params for peak 2N-H
        peakfit analyze profile Fits/ --param x0         # All x0 parameters
        peakfit analyze profile Fits/ --param 2N-H_x0    # Specific parameter

    Parameter correlation analysis:
        peakfit analyze correlation Fits/

    Display existing uncertainties from fit:
        peakfit analyze uncertainty Fits/
    """
    from peakfit.cli.analyze_command import (
        run_correlation,
        run_mcmc,
        run_profile_likelihood,
        run_uncertainty,
    )

    valid_methods = ["mcmc", "profile", "correlation", "uncertainty"]
    if method not in valid_methods:
        ui.error(f"Invalid method: {method}")
        ui.info(f"Valid methods: {', '.join(valid_methods)}")
        raise typer.Exit(1)

    if method == "mcmc":
        # Handle manual override: if --burn-in is specified, disable auto-burnin
        if burn_in is not None and auto_burnin:
            ui.info("Manual burn-in specified; disabling auto-burnin")
            auto_burnin = False

        run_mcmc(
            results_dir=results,
            n_walkers=n_walkers,
            n_steps=n_steps,
            burn_in=burn_in,
            auto_burnin=auto_burnin,
            peaks=peaks,
            output_file=output,
            verbose=False,  # No banner for analyze commands
        )
    elif method == "profile":
        run_profile_likelihood(
            results_dir=results,
            param_name=param,
            n_points=n_points,
            confidence_level=confidence,
            plot=plot,
            output_file=output,
            verbose=False,  # No banner for analyze commands
        )
    elif method == "correlation":
        run_correlation(
            results_dir=results,
            output_file=output,
            verbose=False,  # No banner for analyze commands
        )
    elif method == "uncertainty":
        run_uncertainty(
            results_dir=results,
            output_file=output,
            verbose=False,  # No banner for analyze commands
        )


if __name__ == "__main__":
    app()
