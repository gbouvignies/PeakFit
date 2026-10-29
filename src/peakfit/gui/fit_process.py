"""One disposable spawned fit process; contours keep their existing worker."""

import multiprocessing
import pickle
from contextlib import suppress
from dataclasses import dataclass
from threading import Event, Thread
from time import perf_counter
from typing import TYPE_CHECKING

from PySide6 import QtCore

from peakfit.gui.dataset_fit import DatasetFitAttempt, finish_dataset_fit, run_dataset_child
from peakfit.gui.fitting import PreparedDatasetFit
from peakfit.gui.local_fit import LocalFitAttempt, finish_local_fit, run_fit_child

if TYPE_CHECKING:
    from peakfit.gui.dataset_fit import DatasetFitResult
    from peakfit.gui.fitting import PreparedLocalFit
    from peakfit.gui.local_fit import LocalFitResult


@dataclass(frozen=True, slots=True)
class FitProcessResult:
    """Completed result and observed transfer/startup costs for diagnostics."""

    scientific: LocalFitResult | DatasetFitResult
    startup_seconds: float
    serialization_seconds: float
    input_bytes: int
    wall_seconds: float


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

    def start(self, prepared: PreparedLocalFit | PreparedDatasetFit) -> int:
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

    def _work(  # noqa: PLR0912, PLR0915
        self, generation: int, prepared: PreparedLocalFit | PreparedDatasetFit, cancel: Event
    ) -> None:
        started = perf_counter()
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe()
        dataset = isinstance(prepared, PreparedDatasetFit)
        process = context.Process(
            target=run_dataset_child if dataset else run_fit_child,
            args=(sender,),
            daemon=True,
        )
        outcome: object = None
        state = "failed"
        try:
            before_serialization = perf_counter()
            payload = pickle.dumps(prepared, protocol=pickle.HIGHEST_PROTOCOL)
            # This owned copy retains the exact inputs that cross the process boundary.
            owned = pickle.loads(payload)
            serialization_seconds = perf_counter() - before_serialization
            if cancel.is_set():
                return
            process.start()
            sender.close()
            while not receiver.poll(0.02):
                if cancel.is_set():
                    return
                if not process.is_alive():
                    raise RuntimeError(f"Fit process exited with code {process.exitcode}.")
            if receiver.recv() != "ready":
                raise RuntimeError("Fit process did not initialize correctly.")
            startup_seconds = perf_counter() - started - serialization_seconds
            if cancel.is_set():
                return
            receiver.send_bytes(payload)
            self._emit(generation, "running")
            while True:
                while not receiver.poll(0.02):
                    if cancel.is_set():
                        return
                    if not process.is_alive():
                        raise RuntimeError(f"Fit process exited with code {process.exitcode}.")
                if cancel.is_set():
                    return
                attempt = receiver.recv()
                if isinstance(attempt, tuple) and attempt[0] == "progress":
                    self._emit(generation, "progress", attempt[1])
                    continue
                break
            if dataset and isinstance(attempt, bytes):
                attempt = pickle.loads(attempt)  # Only our own child's trusted in-memory record.
            if isinstance(owned, PreparedDatasetFit) and isinstance(attempt, DatasetFitAttempt):
                scientific = finish_dataset_fit(owned, attempt)
            elif not isinstance(owned, PreparedDatasetFit) and isinstance(attempt, LocalFitAttempt):
                scientific = finish_local_fit(owned, attempt)
            else:
                raise RuntimeError(str(attempt))
            outcome = FitProcessResult(
                scientific,
                startup_seconds,
                serialization_seconds,
                len(payload),
                perf_counter() - started,
            )
            state = "completed"
        except Exception as error:
            outcome = f"{type(error).__name__}: {error}"
        finally:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=1)
                if process.is_alive():
                    process.kill()
                    process.join()
                process.close()
            receiver.close()
            sender.close()
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
