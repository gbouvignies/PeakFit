"""The background lane never emits into a closed, and so possibly deleted, owner."""

from threading import Event, Thread, current_thread, main_thread

from PySide6 import QtCore

from peakfit.gui.background import ViewerComputation


def test_close_waits_for_a_delivery_already_emitting(qapp):
    owner = QtCore.QObject()
    worker = ViewerComputation(owner)
    emitting, release, closed = Event(), Event(), Event()
    emits = []

    def hold(*args):
        # Runs in the worker thread, inside the emit, like Qt's own signal dispatch.
        emits.append((current_thread() is main_thread(), *args))
        emitting.set()
        release.wait(10)

    worker._ready.connect(hold, QtCore.Qt.ConnectionType.DirectConnection)
    go = Event()
    # The job ends after submit() returns, so delivery runs on the worker thread.
    worker.submit("model", 0, lambda: go.wait(10) and "first")
    go.set()
    assert emitting.wait(10)
    closer = Thread(target=lambda: (worker.close(), closed.set()))
    closer.start()
    # Deleting the owner after close() returns would race this emit; close() waits.
    assert not closed.wait(0.3)
    release.set()
    assert closed.wait(10)
    closer.join()
    assert emits == [(False, "model", 0, "first")]  # delivered off the main thread
    owner.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    qapp.processEvents()


def test_a_job_finishing_after_close_never_emits(qapp):
    owner = QtCore.QObject()
    worker = ViewerComputation(owner)
    started, finish = Event(), Event()
    emits = []
    worker._ready.connect(
        lambda *args: emits.append(args), QtCore.Qt.ConnectionType.DirectConnection
    )

    def slow():
        started.set()
        finish.wait(10)
        return "late"

    worker.submit("model", 0, slow)
    assert started.wait(10)
    worker.close()
    finish.set()
    worker._executor.shutdown(wait=True)
    assert emits == []
    owner.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    qapp.processEvents()
