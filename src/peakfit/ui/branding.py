"""Branding helpers for PeakFit terminal output."""

from pathlib import Path

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from peakfit.ui.console import (
    REPO_URL,
    VERSION,
    Verbosity,
    console,
    display_path,
    get_verbosity,
    icon,
)

_SECTION_KEY_WIDTH = 20


def _normalize_summary_value(value: str) -> str:
    """Normalize summary values for consistent user-facing path display."""
    candidate = Path(value)
    if candidate.is_absolute():
        return display_path(candidate)
    return value


def show_command_summary(
    command_title: str,
    sections: list[tuple[str, dict[str, str]]],
) -> None:
    """Render a compact command summary for CLI commands.

    Args:
        command_title: Command name shown as the first metadata line.
        sections: Ordered section tuples of (section name, key/value pairs).
    """
    if get_verbosity() == Verbosity.QUIET:
        return

    content = Table.grid(padding=(0, 0))
    content.add_column()
    content.add_row(Text(f"Command: {command_title}", style="dim"))

    for section_title, items in sections:
        content.add_row(Text(""))
        content.add_row(Text(section_title, style="subheader"))

        section_table = Table.grid(padding=(0, 2))
        section_table.add_column(width=2)
        section_table.add_column(width=_SECTION_KEY_WIDTH, style="key")
        section_table.add_column()

        for key, value in items.items():
            section_table.add_row(icon("bullet"), f"{key}:", _normalize_summary_value(value))

        content.add_row(section_table)

    panel = Panel(
        content,
        title=f"[header]PeakFit v{VERSION}[/header]",
        title_align="left",
        border_style="panel.border",
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print()
    console.print(panel)
    console.print()


__all__ = [
    "show_command_summary",
    "show_version",
]


def show_version() -> None:
    """Show version information (for --version flag)."""
    console.print(f"\n[header]PeakFit[/header] [dim]v{VERSION}[/dim]")
    console.print(f"[dim]{REPO_URL}[/dim]\n")
