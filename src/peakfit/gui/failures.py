"""Job failures: a message the scientist can act on, and a traceback in the log.

Expected failures (``ValueError``, ``OSError``) already say what went wrong and what
to do; they are shown as raised. Anything else is a PeakFit defect: its message
asks for a report, and the full traceback is logged to the terminal that started
PeakFit, including tracebacks raised inside worker processes.
"""

import logging
import traceback
from concurrent.futures import CancelledError

logger = logging.getLogger("peakfit.gui")


class WorkerError(Exception):
    """An operation failed in a worker process; its traceback is already logged."""


def failure_message(error: BaseException) -> str:
    """What to show for a failed job."""
    if isinstance(error, (ValueError, OSError, WorkerError)):
        return str(error)
    return (
        f"Unexpected {type(error).__name__}: {error}. This is a PeakFit error; its "
        "traceback was logged to the terminal that started PeakFit. Please report it."
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
