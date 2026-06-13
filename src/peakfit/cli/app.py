"""Main Typer application for PeakFit.

Thin orchestration layer that registers commands from the commands/ subpackage.
"""

from typing import Annotated

import typer

from peakfit.cli.callbacks import version_callback
from peakfit.cli.commands.fit import fit_command
from peakfit.cli.commands.init import init_command
from peakfit.cli.commands.mcmc import mcmc_command
from peakfit.cli.commands.plot import plot_app

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

    Workflow:
        1. peakfit fit spectrum.ft2 [peaks.list]   # Fit (auto-pick if omitted)
        2. peakfit mcmc results/                   # Uncertainty estimation
        3. peakfit plot intensity results/         # Visualize fitted amplitudes

    For help on any command:
        peakfit <command> --help
    """


# Top-level commands
app.command(name="fit")(fit_command)
app.command(name="mcmc")(mcmc_command)
app.command(name="init")(init_command)

# Sub-applications
app.add_typer(plot_app, name="plot")
