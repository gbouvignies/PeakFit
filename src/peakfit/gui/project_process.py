"""Disposable archive process for scientific evidence and source verification.

Only records created by our own child cross this trusted pipe. Project files
remain strictly metadata/typed arrays and are never unpickled.
"""

import json
import multiprocessing
import pickle
from concurrent.futures import CancelledError
from dataclasses import replace

from peakfit.gui.dataset_fit import _completed_bytes
from peakfit.gui.project import (
    open_analysis,
    save_analysis,
    spectrum_path,
    stored_analysis,
    verify_sources,
)


def _archive_child(connection):
    try:
        kind, payload = pickle.loads(connection.recv_bytes())
        canceled = False

        def is_canceled():
            nonlocal canceled
            if connection.poll():
                canceled = connection.recv() == "cancel"
            return canceled

        def progress(message):
            connection.send(("progress", message))

        if kind in {"save", "preserve", "portable"}:
            analysis, destination = payload
            result = save_analysis(
                analysis,
                destination,
                progress=progress,
                canceled=is_canceled,
                preserve_input_runs=kind == "preserve",
                portable=kind == "portable",
            )
        elif kind == "open":
            result = open_analysis(payload, progress=progress, canceled=is_canceled)
        elif kind == "history":
            result = payload.load()
        else:
            analysis, relink = payload
            result = verify_sources(
                analysis, relink=relink, progress=progress, canceled=is_canceled
            )
        connection.send(("result", _completed_bytes(result)))
    except Exception as error:
        connection.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


def project_operation(kind, payload, *, progress, canceled):
    """Bridge one operation off Qt, with cooperative cancellation and child reaping."""
    if kind in {"save", "preserve", "portable", "relink"}:
        analysis, path = payload
        # Linked saves/checkpoints never transfer the cube. Portable export alone
        # needs it, unless an already published embedded payload can be reused.
        manifest = analysis.path / "manifest.json" if analysis.path else None
        previous = json.loads(manifest.read_text()) if manifest and manifest.is_file() else {}
        embedded = previous.get("spectrum", {}).get("path") == spectrum_path(analysis.header)
        payload = (
            replace(
                analysis,
                dataset=analysis.dataset if kind == "portable" and not embedded else None,
                active=None if kind in {"save", "preserve", "portable"} else analysis.completion,
            ),
            path,
        )
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe()
    process = context.Process(target=_archive_child, args=(sender,), daemon=True)
    try:
        encoded = _completed_bytes((kind, payload))
        process.start()
        sender.close()
        receiver.send_bytes(encoded)
        cancel_sent = False
        while True:
            if canceled() and not cancel_sent:
                receiver.send("cancel")
                cancel_sent = True
            if not receiver.poll(0.02):
                if not process.is_alive():
                    raise RuntimeError(f"Analysis worker exited with code {process.exitcode}")
                continue
            state, value = receiver.recv()
            if state == "progress":
                progress(value)
            elif state == "result":
                return pickle.loads(value)  # Trusted child output only, never a project file.
            else:
                if value.startswith("CancelledError:"):
                    raise CancelledError("Analysis save canceled; previous generation retained")
                raise ValueError(value)
    finally:
        if process.pid is not None:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join()
            process.close()
        receiver.close()
        sender.close()


def save_project(analysis, destination, *, reusable, progress, canceled, portable=False):
    """Reuse fast metadata-only saves; isolate new evidence validation from Qt."""
    if portable and destination.exists():
        raise FileExistsError("Portable export requires a new analysis directory.")
    if reusable:
        path = save_analysis(
            analysis, destination, progress=progress, canceled=canceled, portable=portable
        )
    else:
        path = project_operation(
            "portable" if portable else "save",
            (analysis, destination),
            progress=progress,
            canceled=canceled,
        )
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    saved = stored_analysis(analysis, path)
    return path, saved, manifest["storage_bytes"]
