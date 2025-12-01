"""UI components for displaying fit results.

This module provides specialized display functions for exporting HTML logs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from .console import console

__all__ = [
    "export_html",
]


def export_html(filepath: Path) -> None:
    """Export console output to an HTML file.

    Args:
        filepath: Path to save HTML file
    """
    filepath.write_text(console.export_html())
