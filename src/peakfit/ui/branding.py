"""Branding and banner display for PeakFit UI."""

import contextlib
import os
import platform
import socket
import sys
from datetime import datetime
from pathlib import Path

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from peakfit.ui.console import (
    LOGO_ASCII,
    REPO_URL,
    VERSION,
    Verbosity,
    console,
    display_path,
    get_verbosity,
    hr,
    icon,
)
from peakfit.ui.logging import log

_PLATFORM_PARTS_SHOWN = 3
_SECTION_KEY_WIDTH = 20
_AUTHOR_NAME = "Guillaume Bouvignies"


def _normalize_manifest_value(value: str) -> str:
    """Normalize manifest values for consistent user-facing path display."""
    candidate = Path(value)
    if candidate.is_absolute():
        return display_path(candidate)
    return value


def show_standard_header(title: str | None = None) -> None:
    """Show standard header based on current verbosity level."""
    verbosity = get_verbosity()

    if verbosity == Verbosity.QUIET:
        return

    if verbosity == Verbosity.VERBOSE:
        # Verbose: Show full ASCII banner and run info
        _show_full_banner()
        _show_run_info_panel()
    else:
        # Normal: Show compact header
        _show_compact_header(title)


def _show_full_banner() -> None:
    """Show full ASCII banner."""
    logo_text = Text(LOGO_ASCII, style="header")
    description_text = Text(
        f"Modern NMR Peak Fitting for Pseudo-3D Spectra\n{REPO_URL}\n\n",
        style="dim",
    )
    version_text = Text("Version: ", style="dim")
    version_number_text = Text(f"{VERSION}", style="success")
    all_text = Text.assemble(logo_text, description_text, version_text, version_number_text)
    panel = Panel.fit(
        all_text,
        border_style="panel.border",
        title="[panel.title]PeakFit[/panel.title]",
    )
    console.print(panel)


def _show_run_info_panel() -> None:
    """Show detailed run information panel."""
    start_time = datetime.now()

    # Get command line arguments and clean them
    if sys.argv and ("peakfit" in sys.argv[0] or sys.argv[0].endswith(".py")):
        clean_argv = ["peakfit", *sys.argv[1:]]
    else:
        clean_argv = sys.argv

    command_args = " ".join(clean_argv)

    # Truncate long commands
    max_cmd_length = 80
    if len(command_args) > max_cmd_length:
        command_display = command_args[: max_cmd_length - 3] + "..."
    else:
        command_display = command_args

    # Simplify platform string
    platform_str = platform.platform()
    platform_parts = platform_str.split("-")
    platform_display = (
        "-".join(platform_parts[:_PLATFORM_PARTS_SHOWN])
        if len(platform_parts) > _PLATFORM_PARTS_SHOWN
        else platform_str
    )

    # Create run information panel
    info_text = (
        f"[key]Started:[/key] {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"[key]Command:[/key] {command_display}\n"
        f"[key]Working directory:[/key] {display_path(Path.cwd())}\n"
        f"[key]Python:[/key] {sys.version.split()[0]} | "
        f"[key]Platform:[/key] {platform_display}"
    )

    run_info_panel = Panel(
        info_text,
        title="[panel.title]Run Information[/panel.title]",
        border_style="panel.border",
        box=box.ROUNDED,
        padding=(0, 2),
        expand=False,
    )
    console.print(run_info_panel)
    console.print()

    # Log this information
    _log_run_info(start_time, command_args)


def _show_compact_header(title: str | None) -> None:
    """Show compact header with version and timestamp."""
    grid = Table.grid(expand=True)
    grid.add_column(justify="left", ratio=1)
    grid.add_column(justify="right", ratio=1)

    grid.add_row(
        f"[header]PeakFit v{VERSION}[/header]",
        f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
    )
    console.print(grid)
    if title:
        console.print(f"[header]{title}[/header]")

    console.print(hr(style="panel.border"))
    console.print()


def show_command_manifest(
    command_title: str,
    sections: list[tuple[str, dict[str, str]]],
) -> None:
    """Render a consistent boxed manifest for CLI commands.

    Args:
        command_title: Command name shown as the first metadata line.
        sections: Ordered section tuples of (section name, key/value pairs).
    """
    if get_verbosity() == Verbosity.QUIET:
        return

    content = Table.grid(padding=(0, 0))
    content.add_column()
    content.add_row(Text(f"Command: {command_title}", style="dim"))
    content.add_row(Text(f"Author: {_AUTHOR_NAME}", style="dim"))

    for section_title, items in sections:
        content.add_row(Text(""))
        content.add_row(Text(section_title, style="subheader"))

        section_table = Table.grid(padding=(0, 2))
        section_table.add_column(width=2)
        section_table.add_column(width=_SECTION_KEY_WIDTH, style="key")
        section_table.add_column()

        for key, value in items.items():
            section_table.add_row(icon("bullet"), f"{key}:", _normalize_manifest_value(value))

        content.add_row(section_table)

    panel = Panel(
        content,
        title=f"[header]PeakFit v{VERSION}[/header]",
        title_align="left",
        border_style="panel.border",
        box=box.HEAVY,
        padding=(1, 2),
    )
    console.print()
    console.print(panel)
    console.print()


def _log_run_info(start_time: datetime, command_args: str) -> None:
    """Log run information to file."""
    log("=" * 60)
    log(f"PeakFit v{VERSION} started")
    log("=" * 60)
    log(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Command: {command_args}")
    log(f"Working directory: {display_path(Path.cwd())}")
    log(f"Python: {sys.version.split()[0]}")
    log(f"Platform: {platform.platform()}")
    log(f"User: {os.getenv('USER', 'unknown')}")

    with contextlib.suppress(OSError, ImportError):
        log(f"Hostname: {socket.gethostname()}")
    log("=" * 60)


__all__ = [
    "show_command_manifest",
    "show_standard_header",
    "show_version",
]


def show_version() -> None:
    """Show version information (for --version flag)."""
    console.print(f"\n[header]PeakFit[/header] [dim]v{VERSION}[/dim]")
    console.print(f"[dim]{REPO_URL}[/dim]\n")
