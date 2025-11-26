"""Console-based reporter implementation using Rich.

This module provides a Reporter implementation that uses the PeakFitUI
styling system for rich console output. It adapts the Reporter protocol
to the existing UI infrastructure.
"""

from __future__ import annotations

from peakfit.core.shared.reporter import Reporter
from peakfit.ui.style import PeakFitUI


class ConsoleReporter:
    """Reporter implementation using Rich console output.

    Adapts the Reporter protocol to use PeakFitUI styling for consistent,
    styled terminal output.

    Example:
        >>> from peakfit.ui.reporter import ConsoleReporter
        >>> reporter = ConsoleReporter()
        >>> reporter.action("Loading spectrum...")
        >>> reporter.success("Loaded 1024 data points")
    """

    def action(self, message: str) -> None:
        """Display an action message with visual separation.

        Args:
            message: Action being performed
        """
        PeakFitUI.action(message)

    def info(self, message: str) -> None:
        """Display an informational message.

        Args:
            message: Informational message
        """
        PeakFitUI.info(message)

    def warning(self, message: str) -> None:
        """Display a warning message.

        Args:
            message: Warning message
        """
        PeakFitUI.warning(message)

    def error(self, message: str) -> None:
        """Display an error message.

        Args:
            message: Error message
        """
        PeakFitUI.error(message)

    def success(self, message: str) -> None:
        """Display a success message.

        Args:
            message: Success message
        """
        PeakFitUI.success(message)


# Verify protocol compliance at import time
if not isinstance(ConsoleReporter(), Reporter):
    raise TypeError("ConsoleReporter must satisfy Reporter protocol")

# Default reporter instance for CLI usage
default_reporter: Reporter = ConsoleReporter()
