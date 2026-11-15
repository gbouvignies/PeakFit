"""Qt event-loop helpers for GUI tests."""

from time import perf_counter

from PySide6 import QtCore

# Deadline for real fits, saves and cleanup: their completion is under test, not their
# speed, and parallel suites on one machine slow them several-fold.
COMPLETION = 120


def wait_until(condition, timeout=10):
    """Run the Qt event loop until ``condition()`` holds, failing after ``timeout`` seconds."""
    started = perf_counter()
    while not condition() and perf_counter() - started < timeout:
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(10, loop.quit)
        loop.exec()
    assert condition(), f"Timed out after {timeout} s waiting for Qt delivery"
