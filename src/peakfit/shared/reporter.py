"""Progress and status reporting abstraction.

Core and infrastructure layers can accept a Reporter without depending on the
Rich terminal UI. The concrete console implementation lives in `peakfit.ui`.
"""

from typing import Protocol


class Reporter(Protocol):
    """Protocol for progress and status reporting."""

    def action(self, message: str) -> None:
        """Report an action being performed."""
        ...

    def info(self, message: str) -> None:
        """Report informational message."""
        ...

    def warning(self, message: str) -> None:
        """Report a warning."""
        ...

    def error(self, message: str) -> None:
        """Report an error."""
        ...

    def success(self, message: str) -> None:
        """Report successful completion."""
        ...


class NullReporter:
    """Silent reporter that discards all messages."""

    def action(self, message: str) -> None:
        """Discard action message."""

    def info(self, message: str) -> None:
        """Discard info message."""

    def warning(self, message: str) -> None:
        """Discard warning message."""

    def error(self, message: str) -> None:
        """Discard error message."""

    def success(self, message: str) -> None:
        """Discard success message."""
