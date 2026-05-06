"""CLI command modules for PeakFit.

Each module exports a command function or Typer sub-app
that is registered with the main application.
"""

from peakfit.cli.commands.fit import fit_command
from peakfit.cli.commands.init import init_command
from peakfit.cli.commands.mcmc import mcmc_command
from peakfit.cli.commands.plot import plot_app

__all__ = [
    "fit_command",
    "init_command",
    "mcmc_command",
    "plot_app",
]
