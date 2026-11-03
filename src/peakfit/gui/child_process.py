"""One disposable spawned process per operation, for fits and analysis archives.

The parent and child exchange only records our own code creates over a private
pipe: a ``ready`` handshake, the encoded payload, ``progress`` messages, then one
``result`` or ``error`` record. Project files never cross it and are never unpickled.
"""

import io
import multiprocessing
import pickle
from concurrent.futures import CancelledError
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from peakfit.gui.failures import child_failure, raise_child_failure

if TYPE_CHECKING:
    from collections.abc import Callable
    from multiprocessing.connection import Connection

# How often the parent checks cancellation and the child's liveness, in seconds.
_POLL_SECONDS = 0.02


def _mapping(values):
    return MappingProxyType(values)


def _reduce_mapping(value):
    return _mapping, (dict(value),)


def encode(value: object) -> bytes:
    """Preserve immutable mappings and read-only NumPy buffers across the private pipe.

    This is not a file format or an archive reader. Protocol 5 keeps array write
    protection; the dispatch table applies only to this writer.
    """
    stream = io.BytesIO()
    writer = pickle.Pickler(stream, protocol=5)
    writer.dispatch_table = {MappingProxyType: _reduce_mapping}
    writer.dump(value)
    return stream.getvalue()


def _child_main(connection: Connection, target: Callable[..., Any]) -> None:
    try:
        connection.send("ready")
        payload = pickle.loads(connection.recv_bytes())  # Only the parent's own record.
        canceled = False

        def is_canceled():
            nonlocal canceled
            if not canceled and connection.poll():
                canceled = connection.recv() == "cancel"
            return canceled

        def progress(message):
            connection.send(("progress", message))

        result = target(payload, progress=progress, canceled=is_canceled)
        connection.send(("result", encode(result)))
    except Exception as error:
        connection.send(child_failure(error))
    finally:
        connection.close()


def _reap(process, *, terminate: bool) -> None:
    """Join the child, ending it first when its work must not finish."""
    if process.pid is None:
        return
    if terminate and process.is_alive():
        process.terminate()
    process.join(timeout=1)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1)
    if process.is_alive():
        process.kill()
        process.join()
    process.close()


def run_child(
    target: Callable[..., Any],
    payload: bytes,
    *,
    operation: str,
    progress: Callable[[str], None],
    canceled: Callable[[], bool],
    terminate_on_cancel: bool = False,
    started: Callable[[], None] = lambda: None,
) -> Any:
    """Run ``target(decoded payload, progress=…, canceled=…)`` in a spawned process.

    ``payload`` is already ``encode``d, so the caller can keep exactly what crossed.
    Cancellation is forwarded for ``target`` to poll, or, with ``terminate_on_cancel``,
    ends the child at once. ``started`` runs once the child holds the payload.

    Returns the child's result. Raises ``CancelledError`` when canceled and
    ``WorkerError`` (its traceback logged here) when the child fails. The child is
    always reaped before this returns.
    """
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe()
    process = context.Process(target=_child_main, args=(sender, target), daemon=True)
    hard_cancel = cancel_sent = False

    def wait_for_record(*, cooperative: bool):
        # Before the payload is sent the child cannot read a cancel request.
        nonlocal hard_cancel, cancel_sent
        while True:
            if canceled():
                if terminate_on_cancel or not cooperative:
                    hard_cancel = True
                    raise CancelledError(f"{operation} canceled")
                if not cancel_sent:
                    receiver.send("cancel")
                    cancel_sent = True
            if receiver.poll(_POLL_SECONDS):
                try:
                    return receiver.recv()
                except EOFError:
                    # The child closed the pipe by exiting; report how it ended.
                    process.join(timeout=1)
            if not process.is_alive():
                raise RuntimeError(f"{operation} worker exited with code {process.exitcode}.")

    try:
        process.start()
        sender.close()
        if wait_for_record(cooperative=False) != "ready":
            raise RuntimeError(f"{operation} worker did not initialize correctly.")
        receiver.send_bytes(payload)
        started()
        while True:
            record = wait_for_record(cooperative=True)
            if record[0] == "progress":
                progress(record[1])
            elif record[0] == "result":
                return pickle.loads(record[1])  # Only our own child's in-memory record.
            else:
                raise_child_failure(record, operation)
    finally:
        _reap(process, terminate=hard_cancel)
        receiver.close()
        sender.close()


__all__ = ["encode", "run_child"]
