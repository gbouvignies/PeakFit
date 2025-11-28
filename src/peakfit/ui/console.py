"""Console configuration and theme for PeakFit UI.

This module provides the central console instance and theme used throughout
the application for consistent styling.
"""

from rich.console import Console
from rich.theme import Theme

from peakfit import __version__

# Define custom theme for consistent colors
PEAKFIT_THEME = Theme(
    {
        # Status colors
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "info": "cyan",
        # UI elements
        "header": "bold cyan",
        "subheader": "bold white",
        "emphasis": "bold",
        "dim": "dim",
        "code": "bold magenta",
        # Values
        "value": "green",
        "metric": "cyan",
        "path": "blue underline",
    }
)

# Single console instance for entire application
console = Console(theme=PEAKFIT_THEME, record=True)

# Version and branding
VERSION = __version__
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
    "console",
]
