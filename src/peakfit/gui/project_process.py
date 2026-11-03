"""Analysis save, open and source verification in a disposable child process.

Only records created by our own child cross the private pipe
(``gui.child_process``). Project files remain strictly metadata/typed arrays and
are never unpickled.
"""

import json
from dataclasses import replace

from peakfit.gui.child_process import encode, run_child
from peakfit.gui.project import (
    open_analysis,
    save_analysis,
    stored_analysis,
    verify_sources,
)


def _archive_operation(job, *, progress, canceled):
    """Run one archive operation in the child; ``canceled`` polls the parent's request."""
    kind, payload = job
    if kind in {"save", "preserve"}:
        analysis, destination = payload
        return save_analysis(
            analysis,
            destination,
            progress=progress,
            canceled=canceled,
            preserve_input_runs=kind == "preserve",
        )
    if kind == "open":
        return open_analysis(payload, progress=progress, canceled=canceled)
    analysis, relink = payload
    return verify_sources(analysis, relink=relink, progress=progress, canceled=canceled)


def project_operation(kind, payload, *, progress, canceled):
    """Run one archive operation off Qt, with cooperative cancellation and child reaping."""
    if kind in {"save", "preserve", "relink"}:
        analysis, path = payload
        payload = (
            replace(
                analysis,
                dataset=None,
                active=None if kind in {"save", "preserve"} else analysis.completion,
            ),
            path,
        )
    return run_child(
        _archive_operation,
        encode((kind, payload)),
        operation=f"Analysis {kind}",
        progress=progress,
        canceled=canceled,
    )


def save_project(analysis, destination, *, reusable, progress, canceled):
    """Reuse fast metadata-only saves; isolate new evidence validation from Qt."""
    if reusable:
        path = save_analysis(analysis, destination, progress=progress, canceled=canceled)
    else:
        path = project_operation(
            "save",
            (analysis, destination),
            progress=progress,
            canceled=canceled,
        )
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    saved = stored_analysis(analysis, path)
    return path, saved, manifest["storage_bytes"]
