"""Workspace-owned, readable temporary evidence, with explicit normal-close cleanup."""

import os
import shutil
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from tempfile import gettempdir, mkdtemp

from peakfit.gui.project import stored_analysis
from peakfit.gui.project_process import project_operation


class SessionEvidence:
    """Spool accepted runs before release; preserve a project-readable crash snapshot.

    Only this session's directory is removed on normal close or successful Save.
    An interrupted process leaves Recovery.peakfit for explicit recovery. These are
    ordinary projects, not a second recovery format or an autosave of every edit.
    Calls that write/clean run serially on the fitting worker.
    """

    def __init__(self):
        self.root = None

    @property
    def path(self):
        """Return the last recoverable project location, if one exists."""
        return self.root / "Recovery.peakfit" if self.root else None

    def preserve(self, analysis, *, canceled=lambda: False):
        """Durably publish acceptance before dropping a resident result."""
        if self.root is None:
            self.root = Path(mkdtemp(prefix="peakfit-session-")).resolve()
            (self.root / "pid").write_text(str(os.getpid()))
        path = self.path
        project_operation(
            "preserve",
            (replace(analysis, path=path if path.exists() else None), path),
            progress=lambda _: None,
            canceled=canceled,
        )
        return stored_analysis(analysis, path)

    def prune(self, keep_ids):
        """Release previous-generation payload leases only after Qt adoption."""
        path = self.path
        if path is None:
            return
        with suppress(FileNotFoundError):  # The owner may already be closing after a view transfer.
            for directory in (path / "runs").iterdir():
                if directory.is_dir() and directory.name not in keep_ids:
                    shutil.rmtree(directory)

    def adopt(self, path):
        """Own a user-selected interrupted session after successful recovery."""
        path = path.resolve()
        if (
            path.name != "Recovery.peakfit"
            or not path.parent.name.startswith("peakfit-session-")
            or path.parent.parent != Path(gettempdir()).resolve()
        ):
            raise ValueError("This is not a PeakFit session recovery directory.")
        if path not in interrupted_sessions():
            raise ValueError(
                "This recovery is owned by a live session or cannot be claimed safely."
            )
        self.root = path.parent
        (self.root / "pid").write_text(str(os.getpid()))

    def close(self):
        """Remove owned scratch evidence after preservation/discard; never touch a saved project."""
        if self.root is not None:
            shutil.rmtree(self.root)
            self.root = None


def interrupted_sessions():
    """Find only PeakFit-owned temporary projects, excluding known live owners."""
    paths = []
    for path in Path(gettempdir()).glob("peakfit-session-*/Recovery.peakfit"):
        if not (path / "manifest.json").is_file():
            continue
        try:
            pid = int((path.parent / "pid").read_text())
            if pid == os.getpid():
                continue
            if os.name != "posix":
                # No safe portable PID probe: never claim another potentially live owner.
                continue
            os.kill(pid, 0)
            continue
        except ProcessLookupError:
            pass
        except (OSError, ValueError):
            continue
        paths.append(path.resolve())
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
