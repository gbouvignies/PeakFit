"""Init command implementation."""

from pathlib import Path
from typing import Annotated

import typer

from peakfit.io.config import generate_default_config
from peakfit.ui.branding import show_command_summary
from peakfit.ui.console import (
    Verbosity,
    display_path,
    set_verbosity,
)
from peakfit.ui.messages import (
    error,
    info,
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
            help="Show verbose output",
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
    show_command_summary(
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
    info(f"Next: [code]peakfit fit spectrum.ft2 [peaks.list] --config {display_path(path)}[/code]")
