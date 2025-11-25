"""Plot subcommands for PeakFit CLI.

This module contains all plot-related commands extracted from the main app.py.
It creates a Typer sub-application with commands for:
- Intensity profiles
- CEST profiles
- CPMG profiles
- Interactive spectra viewer
- MCMC diagnostics
"""

from pathlib import Path
from typing import Annotated

import typer

# Create plot sub-application
plot_app = typer.Typer(
    help="Visualization commands for PeakFit results",
    no_args_is_help=True,
)


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
