"""Job failures: a message the scientist can act on, and a traceback in the log.

Expected failures (``ValueError``, ``OSError``) already say what went wrong and what
to do; they are shown as raised. Anything else is a PeakFit defect: its message
asks for a report, and the full traceback is logged, including tracebacks raised
inside worker processes. The log goes to the terminal that started PeakFit and,
once the application installs it, to a per-user log file whose path the GUI shows.
"""

import logging
import os
import platform
import sys
import traceback
from concurrent.futures import CancelledError
from logging.handlers import RotatingFileHandler
from pathlib import Path

import PySide6

from peakfit import __version__

logger = logging.getLogger("peakfit.gui")
_handlers: list[logging.Handler] = []
_log_file: Path | None = None


def default_log_path() -> Path:
    """The per-user PeakFit log file in the platform's usual place for logs."""
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Logs" / "PeakFit" / "peakfit.log"
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        return Path(local or home / "AppData" / "Local") / "PeakFit" / "Logs" / "peakfit.log"
    state = os.environ.get("XDG_STATE_HOME")
    return Path(state or home / ".local" / "state") / "peakfit" / "peakfit.log"


def install_log(path: Path | None = None) -> Path:
    """Log GUI failures to the terminal and to a per-user file; return the file.

    The file keeps about 2 MB: one rotated predecessor of at most 1 MB.
    """
    global _log_file  # noqa: PLW0603 - one log per process
    close_log()
    path = path or default_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    file = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=1, encoding="utf-8")
    file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    # A handler of our own replaces Python's last-resort output to the terminal.
    terminal = logging.StreamHandler()
    for handler in (file, terminal):
        handler.setLevel(logging.ERROR)
        logger.addHandler(handler)
        _handlers.append(handler)
    _log_file = path
    return path


def close_log() -> None:
    """Stop writing the log file; failures reach the terminal only."""
    global _log_file  # noqa: PLW0603 - one log per process
    while _handlers:
        handler = _handlers.pop()
        logger.removeHandler(handler)
        handler.close()
    _log_file = None


def log_path() -> Path | None:
    """The log file failures are written to, if one is installed."""
    return _log_file


def failure_details(status: str, stage: str, message: str, context: list[tuple[str, str]]) -> str:
    """Plain text for a report: what failed and where, with diagnostic context."""
    path = log_path()
    lines = [
        ("Status", status),
        ("Stage", stage),
        ("Message", message),
        *context,
        ("PeakFit", __version__),
        ("Python", platform.python_version()),
        ("Qt", f"PySide6 {PySide6.__version__}"),
        ("Platform", platform.platform()),
        ("Log", str(path) if path else "terminal only"),
    ]
    return "\n".join(f"{label}: {value}" for label, value in lines) + "\n"


class WorkerError(Exception):
    """An operation failed in a worker process; its traceback is already logged."""


def sentence(text: str) -> str:
    """End ``text`` with one full stop unless it already ends a sentence."""
    return text if text.endswith((".", "!", "?")) else text + "."


def failure_message(error: BaseException) -> str:
    """What to show for a failed job."""
    if isinstance(error, (ValueError, OSError, WorkerError)):
        return str(error)
    text = str(error).rstrip()
    return (
        f"Unexpected {type(error).__name__}: {sentence(text)} This is a PeakFit error; its "
        "traceback is in the PeakFit log. Please report it."
    )


def log_failure(operation: str, error: BaseException) -> None:
    """Log a job failure caught in this process, with its traceback."""
    logger.error("%s failed", operation, exc_info=error)


def child_failure(error: BaseException) -> tuple[str, str, str, str]:
    """The record a worker process sends back instead of a result."""
    return ("error", type(error).__name__, failure_message(error), traceback.format_exc())


def raise_child_failure(record: tuple[str, str, str, str], operation: str):
    """Log a worker's traceback here, then raise its failure for the GUI."""
    _, name, message, trace = record
    if name == "CancelledError":
        raise CancelledError(message)
    logger.error("%s failed in its worker process:\n%s", operation, trace)
    raise WorkerError(message)
