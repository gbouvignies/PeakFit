"""Logging configuration for PeakFit UI."""

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.logging import RichHandler

from peakfit.ui.console import VERSION, console, display_path


@dataclass
class _LoggingState:
    logger: logging.Logger | None = None


_logging_state = _LoggingState()


class JSONFormatter(logging.Formatter):
    """JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
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
    if log_file is None:
        _logging_state.logger = None
        return

    # Create log directory if needed
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure logger
    logger = logging.getLogger("peakfit")
    logger.setLevel(level)
    logger.handlers.clear()

    # File handler with structured format
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(level)

    if log_file.suffix == ".json":
        file_formatter: logging.Formatter = JSONFormatter()
    else:
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-5s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler (only if verbose)
    if verbose:
        console_handler = RichHandler(
            console=console,
            show_time=False,
            show_path=False,
            markup=True,
        )
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    _logging_state.logger = logger

    # Log session start
    logger.info("━" * 60)
    logger.info(f"PeakFit v{VERSION} - Session Started")
    logger.info("━" * 60)
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info(f"Working directory: {display_path(Path.cwd())}")
    logger.info(f"Python: {sys.version.split()[0]} | Platform: {sys.platform}")
    logger.info("")


def log(message: str, level: str = "info") -> None:
    """Log a message to file (if logging is enabled)."""
    logger = _logging_state.logger
    if logger is None:
        return

    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    log_level = level_map.get(level.lower(), logging.INFO)
    logger.log(log_level, message)


def log_section(title: str) -> None:
    """Log a section header."""
    logger = _logging_state.logger
    if logger is None:
        return

    logger.info("")
    logger.info(f"=== {title.upper()} ===")


def log_dict(data: dict[str, object], indent: str = "  ") -> None:
    """Log a dictionary as key-value pairs."""
    logger = _logging_state.logger
    if logger is None:
        return

    for key, value in data.items():
        logger.info(f"{indent}- {key}: {value}")


def close_logging() -> None:
    """Close logging and finalize log file."""
    logger = _logging_state.logger
    if logger is None:
        return

    logger.info("")
    logger.info("━" * 60)
    logger.info("PeakFit Session Completed Successfully")
    logger.info("━" * 60)

    # Close all handlers
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    _logging_state.logger = None


__all__ = [
    "close_logging",
    "log",
    "log_dict",
    "log_section",
    "setup_logging",
]
