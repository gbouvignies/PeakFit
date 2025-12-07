"""Console configuration and theme for PeakFit UI.

This module provides the central console instance and theme used throughout
the application for consistent styling.
"""

from rich.console import Console
from rich.theme import Theme
import os
import sys

try:
    # Import version from package root when available
    from peakfit import __version__ as _PKG_VERSION  # type: ignore
except Exception:
    _PKG_VERSION = "dev"

# Define custom theme for consistent colors
# Palette chosen for good contrast in light/dark terminals and colorblind awareness
PEAKFIT_THEME = Theme(
    {
        # --- Semantic Status ---
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "critical": "bold white on red",
        "info": "cyan",
        "neutral": "dim white",
        # --- UI Structure ---
        "header": "bold cyan",
        "subheader": "bold white",
        "panel.border": "blue",
        "panel.title": "bold white",
        "box.border": "dim blue",
        # --- Data & Values ---
        "key": "cyan",
        "value": "green",
        "metric": "bold green",
        "number": "green",
        "string": "yellow",
        "path": "blue underline",
        "url": "blue underline",
        # --- Code & Technical ---
        "code": "bold magenta",
        "class": "bold yellow",
        "method": "bold blue",
        "param": "cyan",
        # --- Progress ---
        "progress.description": "bold white",
        "progress.percentage": "green",
        "progress.remaining": "cyan",
        "progress.elapsed": "dim white",
        # --- Modifiers ---
        "dim": "dim",
        "emphasis": "bold",
    }
)

# Single console instance for entire application
console = Console(theme=PEAKFIT_THEME, record=True)

# Version and branding
VERSION = _PKG_VERSION
REPO_URL = "https://github.com/gbouvignies/PeakFit"
LOGO_EMOJI = "🎯"

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
    "LOGO_EMOJI",
    "PEAKFIT_THEME",
    "REPO_URL",
    "VERSION",
    "Verbosity",
    "console",
    "icon",
    "hr",
    "set_verbosity",
]


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
    global _verbosity
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
    if _EMOJI_DISABLED:
        return False
    enc = getattr(console, "encoding", None) or sys.getdefaultencoding()
    if enc is not None and "utf" not in enc.lower():
        return False
    # Windows terminals are fine on modern versions; allow by default
    return True


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
        width: explicit width; defaults to console.width
        style: Rich style to wrap
        char: override character; defaults to icon('separator')
    """
    w = width or max(20, (console.width or 80) - 2)
    ch = char or icon("separator")
    return f"[{style}]{ch * w}[/{style}]"
