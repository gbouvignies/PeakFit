"""Init command implementation."""

from pathlib import Path
from typing import Annotated

import typer

from peakfit.io.config import generate_default_config
from peakfit.ui import (
    Verbosity,
    bullet,
    display_path,
    error,
    info,
    print_next_steps,
    set_verbosity,
    show_command_manifest,
    success,
)


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
            "-v",
            help="Show banner and verbose output",
        ),
    ] = False,
) -> None:
    """Generate a default configuration file.

    Creates a TOML configuration file with default settings that can be customized.
    All parameters are documented with inline comments explaining their purpose.

    Examples:
    --------
    Create default config:
        $ peakfit init

    Create config with custom name:
        $ peakfit init my_analysis.toml

    Overwrite existing config:
        $ peakfit init --force
    """
    # Set verbosity and show header
    set_verbosity(Verbosity.VERBOSE if verbose else Verbosity.NORMAL)
    show_command_manifest(
        "Configuration Initialization",
        sections=[
            (
                "Configuration",
                {
                    "Target file": display_path(path),
                    "Overwrite": "Yes" if force else "No",
                },
            )
        ],
    )

    if path.exists() and not force:
        error(f"File already exists: [path]{display_path(path)}[/path]")
        info("Use [code]--force[/code] to overwrite")
        raise typer.Exit(1)

    config_content = generate_default_config()
    path.write_text(config_content)

    # Enhanced success message with details
    success(f"Created configuration file: [path]{display_path(path)}[/path]")

    info("Configuration includes:")
    bullet("[value]Fitting parameters[/value] (optimizer, lineshape, tolerances)")
    bullet("[value]Clustering settings[/value] (algorithm, thresholds)")
    bullet("[value]Output preferences[/value] (formats, directories)")
    bullet("[value]Advanced options[/value] (backends)")

    # Suggest next steps
    print_next_steps(
        [
            f"Review and customize: [path]{display_path(path)}[/path]",
            (
                "Run fitting: [code]peakfit fit spectrum.ft2 [peaks.list] --config "
                f"{display_path(path)}[/code]"
            ),
            "Documentation: [url]https://github.com/gbouvignies/PeakFit[/url]",
        ]
    )
