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

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable


@runtime_checkable
class Reporter(Protocol):
    """Protocol for progress and status reporting.

    This abstraction allows core and infrastructure layers to report
    progress without depending on specific UI implementations.

    All methods are intentionally simple strings to avoid coupling
    to any specific output format or styling system.
    """

    def action(self, message: str) -> None:
        """Report an action being performed.

        Use for ongoing operations like 'Fitting peaks...', 'Loading data...'

        Args:
            message: Description of the action being performed
        """
        ...

    def info(self, message: str) -> None:
        """Report informational message.

        Use for general status updates that don't indicate success/failure.

        Args:
            message: Informational message
        """
        ...

    def warning(self, message: str) -> None:
        """Report a warning.

        Use for non-fatal issues that the user should be aware of.

        Args:
            message: Warning message
        """
        ...

    def error(self, message: str) -> None:
        """Report an error.

        Use for errors that may affect results but don't stop execution.

        Args:
            message: Error message
        """
        ...

    def success(self, message: str) -> None:
        """Report successful completion.

        Use when an operation completes successfully.

        Args:
            message: Success message
        """
        ...


class NullReporter:
    """Silent reporter that discards all messages.

    Useful for testing or batch processing where output is not needed.

    Example:
        >>> reporter = NullReporter()
        >>> reporter.action("Processing...")  # No output
        >>> reporter.success("Done!")  # No output
    """

    def action(self, message: str) -> None:
        """Discard action message."""
        pass

    def info(self, message: str) -> None:
        """Discard info message."""
        pass

    def warning(self, message: str) -> None:
        """Discard warning message."""
        pass

    def error(self, message: str) -> None:
        """Discard error message."""
        pass

    def success(self, message: str) -> None:
        """Discard success message."""
        pass


class LoggingReporter:
    """Reporter that writes to Python logging.

    Useful for background processing or when console output is not available.
    Maps reporter methods to appropriate logging levels.

    Example:
        >>> reporter = LoggingReporter("peakfit.fitting")
        >>> reporter.action("Fitting cluster 1...")  # INFO level
        >>> reporter.warning("Low signal")  # WARNING level
        >>> reporter.error("Convergence failed")  # ERROR level
    """

    def __init__(self, logger_name: str = "peakfit") -> None:
        """Initialize with a logger name.

        Args:
            logger_name: Name for the logger (default: 'peakfit')
        """
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
    """Reporter that delegates to multiple reporters.

    Useful when you want both console output and logging simultaneously.

    Example:
        >>> console = ConsoleReporter()
        >>> logging = LoggingReporter()
        >>> reporter = CompositeReporter([console, logging])
        >>> reporter.success("Done!")  # Goes to both reporters
    """

    def __init__(self, reporters: list[Reporter]) -> None:
        """Initialize with list of reporters.

        Args:
            reporters: List of reporters to delegate to
        """
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
