"""Console configuration and theme for PeakFit UI.

This module provides the central console instance and theme used throughout
the application for consistent styling.
"""

import os

from rich.console import Console
from rich.theme import Theme

from peakfit.shared.paths import format_path as display_path

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

__all__ = [
    "PEAKFIT_THEME",
    "REPO_URL",
    "VERSION",
    "Verbosity",
    "console",
    "display_path",
    "get_verbosity",
]


class Verbosity:
    """Verbosity levels for UI output."""

    QUIET = 0  # Errors only
    NORMAL = 1  # Standard output (headers, progress, results)
    VERBOSE = 2  # Detailed output


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

    Names: check, warn, error, info, bullet
    """
    use_emoji = _supports_emoji()
    mapping = {
        "check": "✓" if use_emoji else "+",
        "warn": "⚠" if use_emoji else "!",
        "error": "✗" if use_emoji else "x",
        "info": "▸" if use_emoji else ">",
        "bullet": "‣" if use_emoji else "-",
    }
    return mapping.get(name, mapping["bullet"])
