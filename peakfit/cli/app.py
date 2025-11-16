"""Main Typer application for PeakFit."""

from pathlib import Path
from typing import Annotated, Optional

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
        Optional[bool],
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
        Optional[Path],
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
        Optional[Path],
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
        Optional[float],
        typer.Option(
            "--contour",
            "-t",
            help="Contour level for segmentation (default: 5 * noise)",
        ),
    ] = None,
    noise: Annotated[
        Optional[float],
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
        Optional[list[int]],
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
        Optional[Path],
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
        Optional[Path],
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


if __name__ == "__main__":
    app()
