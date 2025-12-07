"""Logging configuration for PeakFit UI."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from peakfit.ui.console import VERSION, console

# Module-level logger (configured by setup_logging)
_logger: logging.Logger | None = None


class JSONFormatter(logging.Formatter):
    """JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        import json
        from datetime import datetime

        log_record = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record)


def setup_logging(
    log_file: Path | None = None,
    verbose: bool = False,
    level: int = logging.INFO,
) -> None:
    """Configure logging for PeakFit."""
    global _logger

    if log_file is None:
        _logger = None
        return

    # Create log directory if needed
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure logger
    _logger = logging.getLogger("peakfit")
    _logger.setLevel(level)
    _logger.handlers.clear()

    # File handler with structured format
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(level)

    if log_file.suffix == ".json":
        file_formatter = JSONFormatter()
    else:
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-5s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    file_handler.setFormatter(file_formatter)
    _logger.addHandler(file_handler)

    # Console handler (only if verbose)
    if verbose:
        console_handler = RichHandler(
            console=console,
            show_time=False,
            show_path=False,
            markup=True,
        )
        console_handler.setLevel(level)
        _logger.addHandler(console_handler)

    # Log session start
    _logger.info("━" * 60)
    _logger.info(f"PeakFit v{VERSION} - Session Started")
    _logger.info("━" * 60)
    _logger.info(f"Command: {' '.join(sys.argv)}")
    _logger.info(f"Working directory: {Path.cwd()}")
    _logger.info(f"Python: {sys.version.split()[0]} | Platform: {sys.platform}")
    _logger.info("")


def log(message: str, level: str = "info") -> None:
    """Log a message to file (if logging is enabled)."""
    if _logger is None:
        return

    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    log_level = level_map.get(level.lower(), logging.INFO)
    _logger.log(log_level, message)


def log_section(title: str) -> None:
    """Log a section header."""
    if _logger is None:
        return

    _logger.info("")
    _logger.info(f"=== {title.upper()} ===")


def log_dict(data: dict[str, object], indent: str = "  ") -> None:
    """Log a dictionary as key-value pairs."""
    if _logger is None:
        return

    for key, value in data.items():
        _logger.info(f"{indent}- {key}: {value}")


def close_logging() -> None:
    """Close logging and finalize log file."""
    if _logger is None:
        return

    _logger.info("")
    _logger.info("━" * 60)
    _logger.info("PeakFit Session Completed Successfully")
    _logger.info("━" * 60)

    # Close all handlers
    for handler in _logger.handlers[:]:
        handler.close()
        _logger.removeHandler(handler)


__all__ = [
    "close_logging",
    "log",
    "log_dict",
    "log_section",
    "setup_logging",
]
