"""Console configuration and theme for PeakFit UI.

This module provides the central console instance and theme used throughout
the application for consistent styling.
"""

import os
from pathlib import Path

from rich.console import Console
from rich.text import Text
from rich.theme import Theme

try:
    # Import version from package root when available
    from peakfit import __version__ as version
except Exception:
    version = "dev"

# Define custom theme for consistent colors
# "Clarity" Theme - A modern, clean, and readable theme for PeakFit
PEAKFIT_THEME = Theme(
    {
        # --- Semantic Status ---
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "critical": "bold white on red",
        "info": "cyan",
        "neutral": "dim",
        # --- UI Structure ---
        "header": "bold cyan",
        "subheader": "bold",
        "panel.border": "blue",
        "panel.title": "bold",
        "box.border": "dim blue",
        # --- Data & Values ---
        "key": "cyan",
        "value": "bold",
        "metric": "bold",
        "metric.good": "green",
        "metric.warn": "yellow",
        "metric.bad": "red",
        "number": "green",
        "string": "yellow",
        "path": "blue underline",
        "url": "blue underline",
        # --- Code & Technical ---
        "code": "magenta",
        "class": "bold yellow",
        "method": "bold blue",
        "param": "cyan",
        # --- Progress ---
        "progress.description": "white",
        "progress.percentage": "green",
        "progress.remaining": "cyan",
        "progress.elapsed": "dim",
        # --- Modifiers ---
        "dim": "dim",
        "emphasis": "bold",
    }
)

# Single console instance for entire application
console = Console(theme=PEAKFIT_THEME, record=True)

# Version and branding
VERSION = version
REPO_URL = "https://github.com/gbouvignies/PeakFit"

# ASCII Logo
LOGO_ASCII = r"""
   ___           _     _____ _ _
  / _ \___  __ _| | __|  ___(_) |_
 / /_)/ _ \/ _` | |/ /| |_  | | __|
/ ___/  __/ (_| |   < |  _| | | |_
\/    \___|\__,_|_|\_\|_|   |_|\__|
"""

__all__ = [
    "LOGO_ASCII",
    "PEAKFIT_THEME",
    "REPO_URL",
    "VERSION",
    "Verbosity",
    "console",
    "display_path",
    "export_html",
    "get_verbosity",
    "subsection_header",
]


def export_html(filepath: Path) -> None:
    """Export the console output to an HTML file."""
    filepath.write_text(console.export_html())


def display_path(path: Path | str) -> str:
    """Format paths for user-facing output using a CWD-relative representation."""
    p = Path(path)
    if not p.is_absolute():
        return str(p)

    try:
        return os.path.relpath(p.resolve(), Path.cwd().resolve())
    except Exception:
        return str(p)


class Verbosity:
    """Verbosity levels for UI output."""

    QUIET = 0  # Errors only
    NORMAL = 1  # Standard output (headers, progress, results)
    VERBOSE = 2  # Detailed output (banners, debug info)


# Global verbosity state
_verbosity = Verbosity.NORMAL


def set_verbosity(level: int) -> None:
    """Set the global verbosity level.

    Args:
        level: Verbosity level (0=QUIET, 1=NORMAL, 2=VERBOSE)
    """
    global _verbosity  # noqa: PLW0603
    _verbosity = level
    # Update console quiet mode
    console.quiet = level == Verbosity.QUIET


def get_verbosity() -> int:
    """Get the current verbosity level."""
    return _verbosity


# --- UI helpers ---
_EMOJI_DISABLED = os.getenv("PEAKFIT_NO_EMOJI", "").lower() in {"1", "true", "yes"}


def _supports_emoji() -> bool:
    """Best-effort detection if the terminal supports emoji/Unicode symbols."""
    return not _EMOJI_DISABLED


def icon(name: str) -> str:
    """Return a UI icon string based on terminal capabilities.

    Names: check, warn, error, info, bullet, play, stop, dot
    """
    use_emoji = _supports_emoji()
    mapping = {
        "check": "✓" if use_emoji else "+",
        "warn": "⚠" if use_emoji else "!",
        "error": "✗" if use_emoji else "x",
        "info": "▸" if use_emoji else ">",
        "bullet": "‣" if use_emoji else "-",
        "play": "▶" if use_emoji else ">",
        "stop": "■" if use_emoji else "#",
        "dot": "•" if use_emoji else ".",
        "separator": "━" if use_emoji else "-",
    }
    return mapping.get(name, mapping["bullet"])


def hr(width: int | None = None, style: str = "dim", char: str | None = None) -> str:
    """Return a horizontal rule string sized to the console width.

    Args:
        width: explicit width; defaults to min(console.width, 100)
        style: Rich style to wrap
        char: override character; defaults to icon('separator')
    """
    total_width = console.width or 80
    # Cap default width at 100 for readability on wide screens
    default_width = min(total_width, 100)

    w = width or max(20, default_width)
    ch = char or icon("separator")
    return f"[{style}]{ch * w}[/{style}]"


def log(
    message: str | Text,
    level: str = "info",
    verbosity: int = Verbosity.NORMAL,
    highlight: bool = False,
) -> None:
    """Log a message to the console if verbosity allows.

    Args:
        message: The string or Text object to print.
        level: Style name (e.g. 'info', 'warning', 'error', 'debug').
        verbosity: Minimum verbosity required to show this message.
        highlight: Whether to apply Rich syntax highlighting.
    """
    if _verbosity < verbosity:
        return

    # Map 'debug' level to VERBOSE verbosity implicitly if not specified
    if level == "debug" and verbosity == Verbosity.NORMAL and _verbosity < Verbosity.VERBOSE:
        return

    style = level if level in PEAKFIT_THEME.styles else "neutral"

    # If it's a Text object, apply style if not present, otherwise print as is
    if isinstance(message, Text):
        console.print(message, style=style, highlight=highlight)
    else:
        console.print(message, style=style, highlight=highlight)


def log_success(message: str) -> None:
    """Log a success message with icon."""
    log(f"[{icon('check')}] {message}", level="success")


def log_warning(message: str) -> None:
    """Log a warning message with icon."""
    log(f"[{icon('warn')}] {message}", level="warning")


def log_error(message: str) -> None:
    """Log an error message with icon."""
    log(f"[{icon('error')}] {message}", level="error", verbosity=Verbosity.QUIET)


def log_step(step: int, total: int, message: str) -> None:
    """Log a progress step."""
    log(f"[subheader]Step {step}/{total}:[/subheader] {message}")


def log_section(title: str) -> None:
    """Log a main section header."""
    console.print()
    console.print(f"[header]{title}[/header]")
    console.print(hr(style="panel.border"))
    console.print()


def subsection_header(title: str) -> None:
    """Log a subsection header."""
    console.print()
    console.print(f"[subheader]{icon('info')} {title}[/subheader]")
