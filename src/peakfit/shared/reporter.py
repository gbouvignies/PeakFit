"""Progress and status reporting abstraction.

This module provides a protocol-based abstraction for progress and status
reporting, allowing core and infrastructure layers to report progress without
depending on specific UI implementations.

Design Pattern: Protocol-based dependency injection
    - Reporter protocol defines the contract
    - NullReporter provides silent operation for testing/batch
    - LoggingReporter uses Python's logging module
    - ConsoleReporter (in ui/) uses Rich console (not imported here)
"""

import logging
from typing import Protocol, runtime_checkable


@runtime_checkable
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


class LoggingReporter:
    """Reporter that writes to Python logging."""

    def __init__(self, logger_name: str = "peakfit") -> None:
        self._logger = logging.getLogger(logger_name)

    def action(self, message: str) -> None:
        """Log action at INFO level with prefix."""
        self._logger.info("[ACTION] %s", message)

    def info(self, message: str) -> None:
        """Log info at INFO level."""
        self._logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning at WARNING level."""
        self._logger.warning(message)

    def error(self, message: str) -> None:
        """Log error at ERROR level."""
        self._logger.error(message)

    def success(self, message: str) -> None:
        """Log success at INFO level with prefix."""
        self._logger.info("[SUCCESS] %s", message)


class CompositeReporter:
    """Reporter that delegates to multiple reporters."""

    def __init__(self, reporters: list[Reporter]) -> None:
        self._reporters = reporters

    def action(self, message: str) -> None:
        """Delegate action to all reporters."""
        for reporter in self._reporters:
            reporter.action(message)

    def info(self, message: str) -> None:
        """Delegate info to all reporters."""
        for reporter in self._reporters:
            reporter.info(message)

    def warning(self, message: str) -> None:
        """Delegate warning to all reporters."""
        for reporter in self._reporters:
            reporter.warning(message)

    def error(self, message: str) -> None:
        """Delegate error to all reporters."""
        for reporter in self._reporters:
            reporter.error(message)

    def success(self, message: str) -> None:
        """Delegate success to all reporters."""
        for reporter in self._reporters:
            reporter.success(message)
