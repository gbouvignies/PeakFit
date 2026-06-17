"""Console progress reporter backed by the Rich UI helpers."""

from peakfit.ui.console import Verbosity, get_verbosity
from peakfit.ui.messages import action, error, info, success, warning


class ConsoleReporter:
    """Progress reporter that routes messages through the current console verbosity."""

    def action(self, message: str) -> None:
        """Display an action message with visual separation.

        Args:
            message: Action being performed
        """
        if get_verbosity() >= Verbosity.NORMAL:
            action(message)

    def info(self, message: str) -> None:
        """Display an informational message.

        Args:
            message: Informational message
        """
        if get_verbosity() >= Verbosity.NORMAL:
            info(message)

    def warning(self, message: str) -> None:
        """Display a warning message.

        Args:
            message: Warning message
        """
        if get_verbosity() >= Verbosity.QUIET:
            warning(message)

    def error(self, message: str) -> None:
        """Display an error message.

        Args:
            message: Error message
        """
        if get_verbosity() >= Verbosity.QUIET:
            error(message)

    def success(self, message: str) -> None:
        """Display a success message.

        Args:
            message: Success message
        """
        if get_verbosity() >= Verbosity.NORMAL:
            success(message)
