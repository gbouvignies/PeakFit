"""The one child-process runner: results, progress, failures, cancellation and reaping."""

import os
import time
from concurrent.futures import CancelledError
from threading import Event

import pytest

from peakfit.gui.child_process import encode, run_child
from peakfit.gui.failures import WorkerError


def _double(value, *, progress, canceled):
    del canceled
    progress("halfway")
    return {"twice": value * 2}


def _fail(value, *, progress, canceled):
    del progress, canceled
    raise ValueError(f"Cannot use {value}; choose another.")


def _until_canceled(value, *, progress, canceled):
    del value
    progress("waiting")
    while not canceled():
        time.sleep(0.01)
    raise CancelledError("stopped by request")


def _sleep_forever(value, *, progress, canceled):
    del value, canceled
    progress("working")
    time.sleep(60)


def _die(value, *, progress, canceled):
    del value, progress, canceled
    os._exit(3)


def _run(target, value, **options):
    messages = []
    result = run_child(
        target,
        encode(value),
        operation="Test",
        progress=messages.append,
        canceled=options.pop("canceled", lambda: False),
        **options,
    )
    return result, messages


def test_result_and_progress_cross_the_private_pipe():
    started = Event()
    result, messages = _run(_double, 21, started=started.set)
    assert result == {"twice": 42}
    assert messages == ["halfway"]
    assert started.is_set()


def test_a_child_failure_arrives_as_its_own_message():
    with pytest.raises(WorkerError, match=r"Cannot use 5; choose another\."):
        _run(_fail, 5)


def test_cooperative_cancellation_is_forwarded_to_the_child():
    requested = Event()

    def canceled():
        return requested.is_set()

    def progress(message):
        if message == "waiting":
            requested.set()

    with pytest.raises(CancelledError, match="stopped by request"):
        run_child(
            _until_canceled,
            encode(None),
            operation="Test",
            progress=progress,
            canceled=canceled,
        )


def test_terminating_cancellation_ends_a_busy_child_promptly():
    requested = Event()
    began = time.perf_counter()
    with pytest.raises(CancelledError):
        run_child(
            _sleep_forever,
            encode(None),
            operation="Test",
            progress=lambda _: requested.set(),
            canceled=requested.is_set,
            terminate_on_cancel=True,
        )
    assert time.perf_counter() - began < 30


def test_cancellation_before_the_payload_is_sent_ends_the_child():
    with pytest.raises(CancelledError):
        _run(_double, 1, canceled=lambda: True)


def test_a_child_that_exits_without_a_result_is_reported():
    with pytest.raises(RuntimeError, match="Test worker exited with code 3"):
        _run(_die, None)
