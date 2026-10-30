"""One serial computation lane with only the latest pending viewer requests."""

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from typing import TYPE_CHECKING

from PySide6 import QtCore

if TYPE_CHECKING:
    from collections.abc import Callable


class ViewerComputation(QtCore.QObject):
    """Keep numerical work off Qt, without a queue of obsolete slider requests.

    Calls and result delivery happen on the GUI thread. Worker callbacks only
    emit a queued signal; they never touch widgets or create Qt graphics.
    """

    completed = QtCore.Signal(str, int, object)
    _ready = QtCore.Signal(str, int, object)

    def __init__(self, parent: QtCore.QObject) -> None:
        super().__init__(parent)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="peakfit-preview")
        self._pending: dict[str, tuple[int, Callable[[], object]]] = {}
        self._busy = False
        self._closed = False
        self._ready.connect(self._finished)
        parent.destroyed.connect(lambda: self.close())

    def submit(self, kind: str, revision: int, operation: Callable[[], object]) -> None:
        """Replace a pending request; never start a second expensive computation."""
        if self._closed:
            return
        self._pending[kind] = (revision, operation)
        self._start_next()

    def cancel(self, kind: str) -> None:
        """Drop pending work; callers invalidate running results by revision."""
        self._pending.pop(kind, None)

    def _start_next(self) -> None:
        if self._busy or self._closed or not self._pending:
            return
        priority = {"contours": 0, "fit": 1, "slices": 2, "model": 3}
        kind = min(self._pending, key=lambda key: priority.get(key, 4))
        revision, operation = self._pending.pop(kind)
        self._busy = True
        future = self._executor.submit(operation)
        future.add_done_callback(lambda result: self._deliver(kind, revision, result))

    def _deliver(self, kind: str, revision: int, future: Future[object]) -> None:
        try:
            outcome = future.result()
        except Exception as error:  # Return failures to the GUI, including numerical errors.
            outcome = error.with_traceback(None)
        if not getattr(self, "_closed", True):
            # The owning window can be destroyed between the check and emit.
            with suppress(RuntimeError):
                self._ready.emit(kind, revision, outcome)

    @QtCore.Slot(str, int, object)
    def _finished(self, kind: str, revision: int, outcome: object) -> None:
        self._busy = False
        if not self._closed:
            self.completed.emit(kind, revision, outcome)
            self._start_next()

    def cleanup_after_work(self, operation):
        """Queue owned-file cleanup in FIFO order; it must never be coalesced away."""
        self._executor.submit(operation)

    @QtCore.Slot()
    def close(self, finalize=None) -> None:
        """Do not wait for numerical work while closing a Qt window."""
        if self._closed:
            return
        self._closed = True
        getattr(self, "_pending", {}).clear()
        executor = getattr(self, "_executor", None)
        if executor is not None:
            if finalize is not None:
                try:
                    executor.submit(finalize)
                except RuntimeError:  # Interpreter shutdown has already joined executor work.
                    finalize()
            # A finalizer queued behind an in-flight writer must not be canceled.
            executor.shutdown(wait=False, cancel_futures=finalize is None)
