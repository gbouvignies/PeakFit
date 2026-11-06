"""Analysis actions and asynchronous delivery in the native editor shell."""

from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.gui.background import ViewerComputation
from peakfit.gui.editor.numerical import NumericalReview
from peakfit.gui.failures import failure_message
from peakfit.gui.project import (
    run_id,
    source_link_unchanged,
)
from peakfit.gui.project_process import project_operation, save_project
from peakfit.gui.session_evidence import SessionEvidence, interrupted_sessions

if TYPE_CHECKING:
    from peakfit.gui.editor.window import EditorWindow


class AnalysisActions:
    """One analysis lifecycle; fitting remains owned by the existing workspace."""

    def build_analysis_actions(self: EditorWindow, menu):
        """Expose familiar file commands without changing the compact workspace."""
        self.analysis_path = None
        self._analysis = None
        self._recovered_session = SessionEvidence()
        self._recovering_path = None
        self._saved_state = None
        self._saving = False
        self._after_save = None
        self._analysis_cancel = Event()
        self._analysis_worker = ViewerComputation(self)
        self._analysis_worker.completed.connect(self._analysis_finished)
        self._analysis_progress = _Progress(self)
        self._analysis_progress.message.connect(self.loading_status.setText)
        self.analysis_open = self._action(
            menu, "Open analysis…", self.open_analysis_dialog, QtGui.QKeySequence.StandardKey.Open
        )
        self.analysis_save = self._action(
            menu, "Save analysis", self.save_analysis_dialog, QtGui.QKeySequence.StandardKey.Save
        )
        keeps = "Keeps working state and the latest ten runs; spectra stay external."
        self.analysis_save.setStatusTip(keeps)
        self.analysis_save.setToolTip(keeps)
        self.analysis_save_as = self._action(
            menu,
            "Save analysis as…",
            lambda: self.save_analysis_dialog(save_as=True),
            QtGui.QKeySequence.StandardKey.SaveAs,
        )
        self.relink_action = self._action(menu, "Relink source…", self.relink_source)
        self.history_action = self._action(menu, "Analysis history…", self.show_history)
        self.recover_action = self._action(
            menu, "Recover interrupted session…", self.recover_session
        )
        self._analysis_actions()

    def _analysis_state(self: EditorWindow):
        if self.workspace:
            return self.workspace.analysis_state()
        if self._analysis:
            return {"kept runs": self._analysis.protected}
        return None

    def unsaved(self: EditorWindow) -> tuple[str, ...]:
        """What differs from the last saved or opened analysis: the one Modified state."""
        state = self._analysis_state()
        if state is None:
            return ()
        saved = self._saved_state
        if saved is None:
            parts = ["not saved as an analysis"]
        else:
            parts = [name for name, value in state.items() if saved.get(name) != value]
            if len(parts) > 1 and "scientific edits" in parts:
                parts.remove("scientific edits")  # named by what was edited
        if self.workspace and self.workspace.pending:
            parts.append("uncommitted draft")
        return tuple(parts)

    def _show_modified(self: EditorWindow) -> None:
        unsaved = self.unsaved()
        self.setWindowModified(bool(unsaved))
        if self.workspace:
            self.workspace.show_modified(unsaved)

    def _analysis_actions(self: EditorWindow):
        available = self.workspace is not None or self._analysis is not None
        busy = self._busy or self._saving or bool(self.workspace and self.workspace.fitting.busy)
        self.analysis_open.setEnabled(not busy)
        self.recover_action.setEnabled(not busy)
        if self.workspace:
            self.workspace.set_inputs_loading(self._busy or self._saving)
        for name in (
            "open_action",
            "import_action",
            "values_action",
            "fit_action",
            "settings_action",
            "run_action",
        ):
            action = getattr(self, name, None)
            if action is not None and busy:
                action.setEnabled(False)
        for action in (self.analysis_save, self.analysis_save_as):
            action.setEnabled(available and not busy)
        self.history_action.setEnabled(available and not busy)
        self.relink_action.setEnabled(available and not busy and self.analysis_path is not None)

    def capture_analysis(self: EditorWindow):
        """Capture immutable references and modest navigation state on Qt."""
        if self.workspace is None:
            return self._analysis
        analysis = self.workspace.capture()
        header = analysis.header
        if self._analysis:
            header = dict(header, source=self._analysis.header["source"])
        return replace(analysis, header=header, path=self.analysis_path)

    def save_analysis_dialog(self: EditorWindow, *, save_as=False, after=None):
        """Save off Qt; only successful publication marks this exact state saved."""
        if self._saving or self._busy or (self.workspace and self.workspace.fitting.busy):
            return False
        if self.workspace and not self.workspace.commit_fields():
            return False
        analysis = self.capture_analysis()
        if analysis is None:
            return False
        path = self.analysis_path
        if save_as or path is None:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save analysis",
                str(path or "Analysis.peakfit"),
                "PeakFit analysis (*.peakfit)",
            )
            if not filename:
                return False
            path = Path(filename)
            if path.suffix != ".peakfit":
                path = path.with_suffix(".peakfit")
        self._save_state = self._analysis_state()
        self._saving = True
        self._after_save = after
        self._analysis_cancel.clear()
        self.loading_status.setText("Saving linked analysis")
        self.cancel_load.setText("Cancel saving")
        self.cancel_load.show()
        self._analysis_actions()
        known = {run_id(c) for c in self._analysis.runs} if self._analysis else set()
        reusable = analysis.path is not None and all(run_id(c) in known for c in analysis.runs)
        self._analysis_worker.submit(
            "save",
            0,
            lambda: save_project(
                analysis,
                path,
                reusable=reusable,
                progress=self._analysis_progress.message.emit,
                canceled=self._analysis_cancel.is_set,
            ),
        )
        return True

    def open_analysis_dialog(self: EditorWindow):
        """Choose the .peakfit directory explicitly; no disk search or substitution."""
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open analysis (.peakfit)")
        if path:
            self.load_analysis(Path(path))

    def load_analysis(self: EditorWindow, path, *, recovery=False):
        """Validate in the background, retaining the old workspace until delivery."""
        if not self.protect_edits(after=lambda: self.load_analysis(path, recovery=recovery)):
            return
        self._recovering_path = path if recovery else None
        self._busy = True
        self._analysis_cancel.clear()
        self.loading_status.setText("Opening analysis · Validating retained evidence")
        self.cancel_load.setText("Cancel verification")
        self.cancel_load.show()
        self._analysis_actions()
        self._analysis_worker.submit(
            "analysis",
            0,
            lambda: project_operation(
                "open",
                path,
                progress=self._analysis_progress.message.emit,
                canceled=self._analysis_cancel.is_set,
            ),
        )

    def relink_source(self: EditorWindow):
        """Verify only the explicitly selected spectrum against immutable identity."""
        analysis = self.capture_analysis()
        if analysis is None:
            return
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Relink original spectrum")
        if not filename:
            return
        self._relink_was_modified = bool(self.unsaved())
        self._busy = True
        self._analysis_cancel.clear()
        self.cancel_load.show()
        self.cancel_load.setText("Cancel verification")
        self._analysis_actions()
        self._analysis_worker.submit(
            "relink",
            0,
            lambda: project_operation(
                "relink",
                (analysis, Path(filename)),
                progress=self._analysis_progress.message.emit,
                canceled=self._analysis_cancel.is_set,
            ),
        )

    def _analysis_finished(self: EditorWindow, kind, _revision, outcome):
        self._busy = False
        self._saving = False
        self.cancel_load.hide()
        self.cancel_load.setText("Cancel loading")
        if isinstance(outcome, CancelledError):
            self._after_save = None
            self.loading_status.setText(
                {
                    "save": "Save canceled · Previous saved analysis retained",
                    "analysis": "Opening canceled · Current workspace retained",
                    "relink": "Relink canceled · Source link unchanged",
                }[kind]
            )
        elif isinstance(outcome, Exception):
            self._after_save = None
            message = failure_message(outcome)
            self.loading_status.setText(f"Analysis operation failed: {message}")
            QtWidgets.QMessageBox.warning(self, "Cannot complete analysis operation", message)
        elif kind == "save":
            path, analysis, storage = outcome
            self.analysis_path = path
            self._analysis = replace(
                analysis, path=path, active=None if self.workspace else analysis.completion
            )
            if analysis.dataset is not None and not source_link_unchanged(analysis.header):
                self.install_analysis(
                    replace(
                        analysis,
                        path=path,
                        dataset=None,
                        reason="Source unavailable or changed. Relink to verify it.",
                    )
                )
            self._saved_state = self._save_state
            size = sum(storage.values()) / 1024**2
            self.loading_status.setText(
                f"Saved {path.name} · {size:.1f} MiB project data · Spectrum linked"
            )
            if self.workspace:
                self.workspace.fitting.adopt_saved(analysis.runs, analysis.journal)
            self._analysis_worker.cleanup_after_work(self._recovered_session.close)
            after, self._after_save = self._after_save, None
            if after:
                QtCore.QTimer.singleShot(0, after)
        else:
            self.install_analysis(outcome)
            if kind == "relink" and self._relink_was_modified:
                self._saved_state = None
        if self.workspace:
            self.workspace.set_inputs_loading(False)
            self.workspace.render_fitting()
        self._work_changed()
        self.open_action.setEnabled(True)

    def install_analysis(self: EditorWindow, analysis):
        """Publish verified spectra or an explicitly numerical-only workspace."""
        # A capability change keeps the same evidence. Transfer its scratch owner
        # before stopping the old workspace, rather than deleting still-linked history.
        owner = self.workspace.fitting.hand_over_evidence(analysis.runs) if self.workspace else None
        if owner is not None:
            self._analysis_worker.cleanup_after_work(self._recovered_session.close)
            self._recovered_session = owner
        self._analysis = analysis
        self.analysis_path = analysis.path
        if analysis.dataset is not None:
            self.install_dataset(analysis.dataset, analysis=analysis)
            self._analysis = analysis
            self.analysis_path = analysis.path
            w = self.workspace
            assert w is not None
            w.restore(analysis)
            self._analysis = replace(analysis, active=None)
        else:
            self.release_unlinked_recovery(analysis.runs)
            if self.workspace:
                self.workspace.stop()
            self.workspace = None
            self.adopt_status(None)
            review = NumericalReview(analysis)
            review.relink.clicked.connect(self.relink_source)
            review.keep_changed.connect(self._keep_evidence)
            self.setCentralWidget(review)
            self.edit_menu.clear()
            for action in (
                self.import_action,
                self.values_action,
                self.export_action,
                self.fit_action,
                self.settings_action,
                self.run_action,
                self.fit_export_action,
            ):
                action.setEnabled(False)
        recovering = self._recovering_path == analysis.path
        if recovering:
            self._recovered_session.adopt(analysis.path)
            self.analysis_path = None
            self._recovering_path = None
        self._saved_state = None if recovering else self._analysis_state()
        self.setWindowTitle(f"{analysis.path.name}[*] — PeakFit")
        self._show_modified()
        self.loading_status.setText(
            "Numerical review only · Relink source to restore spectral views"
            if analysis.dataset is None
            else "Analysis opened · Experimental spectrum and model verified"
        )
        self._analysis_actions()

    def show_history(self: EditorWindow):
        """Inspect retained full/summary evidence and protect named full runs."""
        analysis = self.capture_analysis()
        if analysis is None:
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Analysis history")
        dialog.resize(1000, 700)
        dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        layout = QtWidgets.QVBoxLayout(dialog)
        review = NumericalReview(analysis)
        review.relink.hide()
        review.keep_changed.connect(self._keep_evidence)
        layout.addWidget(review)
        dialog.finished.connect(review.release)
        dialog.exec()

    def release_unlinked_recovery(self: EditorWindow, runs=()):
        """End scratch ownership when replacing the analysis, not just the window."""
        owner = self._recovered_session
        if owner.path and not any(getattr(r, "root", None) == owner.path for r in runs):
            self._recovered_session = SessionEvidence()
            self._analysis_worker.cleanup_after_work(owner.close)

    def recover_session(self: EditorWindow):
        """Read an interrupted session through the same strict project reader."""
        candidates = interrupted_sessions()
        if not candidates:
            QtWidgets.QMessageBox.information(
                self, "Session recovery", "No interrupted PeakFit session was found."
            )
            return
        selected, accepted = QtWidgets.QInputDialog.getItem(
            self,
            "Recover interrupted session",
            "Choose retained accepted evidence. Save as a new analysis after recovery.",
            [str(p) for p in candidates],
            0,
            False,
        )
        if accepted:
            self.load_analysis(Path(selected), recovery=True)

    def _keep_evidence(self: EditorWindow, run_id, keep):
        if self.workspace:
            self.workspace.fitting.keep_run(run_id, keep)
        elif self._analysis:
            protected = set(self._analysis.protected)
            protected.add(run_id) if keep else protected.discard(run_id)
            self._analysis = replace(self._analysis, protected=frozenset(protected))
            self._show_modified()


class _Progress(QtCore.QObject):
    message = QtCore.Signal(str)
