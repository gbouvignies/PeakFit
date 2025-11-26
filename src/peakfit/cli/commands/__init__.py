"""CLI command modules for PeakFit.

This package contains individual command implementations that are
registered with the main Typer application.

Each module exports:
- A command function decorated with the necessary Typer annotations
- Any helper functions needed for the command

The main app.py imports and registers these commands.
"""

from peakfit.cli.commands.analyze import analyze_app
from peakfit.cli.commands.benchmark import benchmark_command
from peakfit.cli.commands.fit import fit_command
from peakfit.cli.commands.info import info_command
from peakfit.cli.commands.init import init_command
from peakfit.cli.commands.plot import plot_app
from peakfit.cli.commands.validate import validate_command

__all__ = [
    "analyze_app",
    "benchmark_command",
    "fit_command",
    "info_command",
    "init_command",
    "plot_app",
    "validate_command",
]
