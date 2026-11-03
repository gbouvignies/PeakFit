"""One disposable spawned fit process; contours keep their existing worker."""

import pickle
from concurrent.futures import CancelledError
from contextlib import suppress
from threading import Event, Thread
from time import perf_counter
from typing import TYPE_CHECKING

from PySide6 import QtCore

from peakfit.gui.child_process import encode, run_child
from peakfit.gui.dataset_fit import finish_dataset_fit, fit_in_child
from peakfit.gui.failures import WorkerError, failure_message, log_failure

if TYPE_CHECKING:
    from peakfit.gui.fitting import PreparedDatasetFit


class FitProcess(QtCore.QObject):
    """Own one fit, never queue fits or share mutable scientific objects with Qt."""

    changed = QtCore.Signal()
    finished = QtCore.Signal(int, object)
    _delivery = QtCore.Signal(int, str, object)

    def __init__(self, parent: QtCore.QObject) -> None:
        super().__init__(parent)
        self.state = "idle"
        self.generation = 0
        self.started_at = 0.0
        self.progress = ""
        self._cancel = Event()
        self._closed = False
        self._thread: Thread | None = None
        self._delivery.connect(self._receive)
        parent.destroyed.connect(self.close)
        application = QtCore.QCoreApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(self.close)

    @property
    def busy(self) -> bool:
        """Remain busy until cancellation has reaped the actual child."""
        return self.state in {"starting", "running", "canceling"}

    def start(self, prepared: PreparedDatasetFit) -> int:
        """Start exactly one operation; serialize and launch outside the Qt thread."""
        if self.busy or self._closed:
            raise RuntimeError("A fit is already active or this viewer is closed.")
        self.generation += 1
        self.started_at = perf_counter()
        self.state = "starting"
        self.progress = "Starting fit process"
        self._cancel = Event()
        self._thread = Thread(
            target=self._work,
            args=(self.generation, prepared, self._cancel),
            name="peakfit-fit",
            # Python must not exit between spawning and reaping the disposable child.
            # Window/app shutdown signals cancellation; all waiting remains off Qt.
            daemon=False,
        )
        self._thread.start()
        self.changed.emit()
        return self.generation

    def cancel(self) -> None:
        """Request actual process termination, not just suppression of its result."""
        if self.busy:
            self.state = "canceling"
            self._cancel.set()
            self.changed.emit()

    @QtCore.Slot()
    def close(self) -> None:
        """Invalidate deliveries and let the bridge terminate/reap off the GUI thread."""
        self._closed = True
        self._cancel.set()

    def _emit(self, generation: int, state: str, value: object = None) -> None:
        if not self._closed:
            with suppress(RuntimeError):
                self._delivery.emit(generation, state, value)

    def _work(self, generation: int, prepared: PreparedDatasetFit, cancel: Event) -> None:
        outcome: object = None
        state = "failed"
        try:
            payload = encode(prepared)
            # This owned copy retains the exact inputs that cross the process boundary.
            owned = pickle.loads(payload)
            attempt = run_child(
                fit_in_child,
                payload,
                operation="Dataset fit",
                progress=lambda text: self._emit(generation, "progress", text),
                canceled=cancel.is_set,
                terminate_on_cancel=True,
                started=lambda: self._emit(generation, "running"),
            )
            outcome = finish_dataset_fit(owned, attempt)
            state = "completed"
        except CancelledError:
            pass
        except Exception as error:
            if not isinstance(error, WorkerError):
                log_failure("Dataset fit", error)
            outcome = failure_message(error)
        finally:
            self._emit(generation, "canceled" if cancel.is_set() else state, outcome)

    @QtCore.Slot(int, str, object)
    def _receive(self, generation: int, state: str, outcome: object) -> None:
        if self._closed or generation != self.generation:
            return
        if state == "progress":
            if not self._cancel.is_set():
                self.progress = str(outcome)
                self.changed.emit()
            return
        if self._cancel.is_set() and state == "running":
            return
        self.state = "canceled" if self._cancel.is_set() else state
        self.changed.emit()
        if state != "running":
            self.finished.emit(generation, outcome if self.state != "canceled" else None)
