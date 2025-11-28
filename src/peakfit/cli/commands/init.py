"""Init command implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from peakfit.io.config import generate_default_config
from peakfit.ui import console, error, info, print_next_steps, show_banner, success


def init_command(
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

    Examples
    --------
      Create default config:
        $ peakfit init

      Create config with custom name:
        $ peakfit init my_analysis.toml

      Overwrite existing config:
        $ peakfit init --force
    """
    # Show banner based on verbosity
    show_banner(verbose)

    if path.exists() and not force:
        error(f"File already exists: [path]{path}[/path]")
        info("Use [code]--force[/code] to overwrite")
        raise typer.Exit(1)

    config_content = generate_default_config()
    path.write_text(config_content)

    # Enhanced success message with details
    success(f"Created configuration file: [path]{path}[/path]")

    console.print("\n[bold cyan]📄 Configuration includes:[/]")
    console.print("  • [green]Fitting parameters[/] (optimizer, lineshape, tolerances)")
    console.print("  • [green]Clustering settings[/] (algorithm, thresholds)")
    console.print("  • [green]Output preferences[/] (formats, directories)")
    console.print("  • [green]Advanced options[/] (backends)")

    # Suggest next steps
    print_next_steps([
        f"Review and customize: [cyan]{path}[/]",
        f"Run fitting: [cyan]peakfit fit spectrum.ft2 peaks.list --config {path}[/]",
        "Documentation: [cyan]https://github.com/gbouvignies/PeakFit[/]",
    ])
