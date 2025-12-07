"""Branding and banner display for PeakFit UI."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

from rich import box
from rich.panel import Panel
from rich.text import Text

from peakfit.ui.console import (
    LOGO_ASCII,
    LOGO_EMOJI,
    REPO_URL,
    VERSION,
    Verbosity,
    console,
    hr,
    icon,
    get_verbosity,
)
from peakfit.ui.logging import log


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
    panel = Panel.fit(all_text, border_style="panel.border", title=f"{LOGO_EMOJI} PeakFit")
    console.print(panel)


def _show_run_info_panel() -> None:
    """Show detailed run information panel."""
    from datetime import datetime

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
    platform_display = "-".join(platform_parts[:3]) if len(platform_parts) > 3 else platform_str

    # Create run information panel
    info_text = (
        f"[key]Started:[/key] {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"[key]Command:[/key] {command_display}\n"
        f"[key]Working directory:[/key] {Path.cwd()}\n"
        f"[key]Python:[/key] {sys.version.split()[0]} | "
        f"[key]Platform:[/key] {platform_display}"
    )

    run_info_panel = Panel(
        info_text,
        title="Run Information",
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
    from datetime import datetime

    from rich.table import Table

    grid = Table.grid(expand=True)
    grid.add_column(justify="left", ratio=1)
    grid.add_column(justify="right", ratio=1)

    grid.add_row(
        f"{LOGO_EMOJI} [header]PeakFit v{VERSION}[/header]",
        f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
    )

    if title:
        console.print(
            Panel(
                grid,
                title=f"[header]{title}[/header]",
                border_style="panel.border",
                subtitle="[dim]Modern NMR Analysis[/dim]",
            )
        )
    else:
        console.print(grid)
        console.print(hr())
    console.print()


def _log_run_info(start_time: datetime, command_args: str) -> None:
    """Log run information to file."""
    log("=" * 60)
    log(f"PeakFit v{VERSION} started")
    log("=" * 60)
    log(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Command: {command_args}")
    log(f"Working directory: {Path.cwd()}")
    log(f"Python: {sys.version.split()[0]}")
    log(f"Platform: {platform.platform()}")
    log(f"User: {os.getenv('USER', 'unknown')}")
    try:
        import socket

        log(f"Hostname: {socket.gethostname()}")
    except (OSError, ImportError):
        pass
    log("=" * 60)


# Deprecated functions kept for backward compatibility
def show_banner(verbose: bool = False) -> None:
    """Show PeakFit banner (Deprecated: use show_standard_header)."""
    if verbose:
        _show_full_banner()


def show_run_info(start_time: datetime) -> None:
    """Show run info (Deprecated: use show_standard_header)."""
    _show_run_info_panel()


def show_version() -> None:
    """Show version information (for --version flag)."""
    console.print(f"\n{LOGO_EMOJI} [header]PeakFit[/header] [dim]v{VERSION}[/dim]")
    console.print(f"[dim]{REPO_URL}[/dim]\n")


def show_footer(start_time: datetime, end_time: datetime) -> None:
    """Show run completion footer."""
    if get_verbosity() == Verbosity.QUIET:
        return

    elapsed = end_time - start_time
    minutes, seconds = divmod(elapsed.total_seconds(), 60)

    time_str = f"{int(minutes)}m {seconds:.1f}s" if minutes > 0 else f"{seconds:.2f}s"

    console.print()
    console.print(hr())
    console.print(f"{LOGO_EMOJI} [success]{icon('check')} Complete![/success] [dim]Elapsed: {time_str}[/dim]")
    console.print(hr())
    console.print()


__all__ = [
    "show_banner",
    "show_footer",
    "show_run_info",
    "show_standard_header",
    "show_version",
]
