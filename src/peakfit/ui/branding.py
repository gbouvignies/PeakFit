"""Branding and banner display for PeakFit UI.

This module provides banner, version, and run info display functions.
"""

from __future__ import annotations

import os
import platform
import sys
from datetime import datetime
from pathlib import Path

from rich import box
from rich.panel import Panel
from rich.text import Text

from peakfit.ui.console import LOGO_ASCII, LOGO_EMOJI, REPO_URL, VERSION, console
from peakfit.ui.logging import log


def show_banner(verbose: bool = False) -> None:
    """Show PeakFit banner based on verbosity level.

    Args:
        verbose: If True, show full banner with logo
    """
    if not verbose:
        return

    logo_text = Text(LOGO_ASCII, style="bold cyan")
    description_text = Text(
        f"Modern NMR Peak Fitting for Pseudo-3D Spectra\n{REPO_URL}\n\n",
        style="dim",
    )
    version_text = Text("Version: ", style="dim")
    version_number_text = Text(f"{VERSION}", style="bold green")
    all_text = Text.assemble(logo_text, description_text, version_text, version_number_text)
    panel = Panel.fit(all_text, border_style="cyan", title=f"{LOGO_EMOJI} PeakFit")
    console.print(panel)


def show_version() -> None:
    """Show version information (for --version flag)."""
    console.print(f"\n{LOGO_EMOJI} [header]PeakFit[/header] [dim]v{VERSION}[/dim]")
    console.print(f"[dim]{REPO_URL}[/dim]\n")


def show_run_info(start_time: datetime) -> None:
    """Show run information header with context.

    Args:
        start_time: When the program started
    """
    # Logo and version
    console.print(f"\n{LOGO_EMOJI} [bold cyan]PeakFit[/bold cyan] [dim]v{VERSION}[/dim]")
    console.print("━" * 70 + "\n")

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
    if len(platform_parts) > 3:
        platform_display = "-".join(platform_parts[:3])
    else:
        platform_display = platform_str

    # Create run information panel
    info_text = (
        f"[cyan]Started:[/cyan] {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"[cyan]Command:[/cyan] {command_display}\n"
        f"[cyan]Working directory:[/cyan] {Path.cwd()}\n"
        f"[cyan]Python:[/cyan] {sys.version.split()[0]} | "
        f"[cyan]Platform:[/cyan] {platform_display}"
    )

    run_info_panel = Panel(
        info_text,
        title="Run Information",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(0, 2),
        expand=False,
    )
    console.print(run_info_panel)
    console.print()

    # Log this information
    original_command = " ".join(sys.argv)
    log("=" * 60)
    log(f"PeakFit v{VERSION} started")
    log("=" * 60)
    log(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Command: {original_command}")
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


def show_footer(start_time: datetime, end_time: datetime) -> None:
    """Show run completion footer.

    Args:
        start_time: When the program started
        end_time: When the program ended
    """
    elapsed = end_time - start_time
    minutes, seconds = divmod(elapsed.total_seconds(), 60)

    if minutes > 0:
        time_str = f"{int(minutes)}m {seconds:.1f}s"
    else:
        time_str = f"{seconds:.2f}s"

    console.print()
    console.print("━" * 70)
    console.print(
        f"{LOGO_EMOJI} [bold green]Complete![/bold green] "
        f"[dim]Elapsed: {time_str}[/dim]"
    )
    console.print("━" * 70)
    console.print()


__all__ = [
    "show_banner",
    "show_footer",
    "show_run_info",
    "show_version",
]
