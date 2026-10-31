"""Whole-dataset execution and atomic completion in the existing editor workspace."""

from dataclasses import replace
from datetime import UTC, datetime
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import uuid4

from PySide6 import QtCore, QtWidgets

from peakfit.engine.domain.config import FitConfig, PeakFitConfig
from peakfit.gui.background import ViewerComputation
from peakfit.gui.dataset_dialog import DatasetFitDialog
from peakfit.gui.dataset_fit import DatasetFitResult, dataset_summary
from peakfit.gui.export_dialog import choose_and_export
from peakfit.gui.fit_positions import dataset_completion
from peakfit.gui.fit_process import FitProcess, FitProcessResult
from peakfit.gui.fitting import PreparedDatasetFit, prepare_dataset_fit
from peakfit.gui.project import Analysis, dataset_header, run_id
from peakfit.gui.project_results import journal_entry
from peakfit.gui.session_evidence import SessionEvidence

if TYPE_CHECKING:
    from peakfit.gui.editor.workspace import EditorWorkspace
    from peakfit.gui.fitting import FitInputSnapshot


class DatasetFitting(QtCore.QObject):
    """One active immutable result, one fit and fresh whole-dataset preparation."""

    state_changed = QtCore.Signal()

    def __init__(self, workspace: EditorWorkspace) -> None:
        super().__init__(workspace)
        self.workspace = workspace
        self.configuration = PeakFitConfig(fitting=FitConfig(refine_iterations=2))
        self.reviewed = False
        self.completion = None
        self.history = []
        self.journal = []
        self.protected = set()
        self._session = SessionEvidence()
        self._attempt_id = None
        self._attempt_parent = None
        self._attempt_initial = ()
        self.exported = set()
        self.attempt = ""
        self.failure = ""
        self._preparing = False
        self._generation = 0
        self._submitted_revision = 0
        self._started = 0.0
        self._worker = ViewerComputation(self)
        self._worker.completed.connect(self._prepared)
        self.process = FitProcess(self)
        self.process.changed.connect(self.update)
        self.process.finished.connect(self._finished)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.update)
        self.timer.start()
        workspace.changed.connect(self.update)

    @property
    def _recovery_path(self):
        return self._session.path

    def release_recovery(self):
        """Clean scratch files only after Save has rebound history to published evidence."""
        self._worker.cleanup_after_work(self._session.close)

    @property
    def busy(self) -> bool:
        """Protect scientific edits through preparation, execution and publication."""
        return self._preparing or self.process.busy

    @property
    def result(self) -> DatasetFitResult | None:
        """Retain the previous accepted result across canceled and failed attempts."""
        return self.completion.result if self.completion else None

    def snapshot(self, template: FitInputSnapshot) -> FitInputSnapshot:
        """Read scientific identity without preparing or rounding any model inputs."""
        w = self.workspace
        spectra = w.dataset.spectra
        return replace(
            template,
            spectrum_id=id(spectra),
            data_shape=tuple(spectra.data.shape),
            spectral_parameters=tuple(axis.model_dump_json() for axis in spectra.params),
            plane_values=tuple(float(z) for z in spectra.z_values),
            peaks=w.peaks,
            configuration=self.configuration.model_dump_json(),
            source_files=w.dataset.sources,
        )

    @property
    def current(self) -> bool:
        """Only the recorded completion transition can certify optimized starts."""
        return self.completion is not None and self.completion.is_current(
            self.snapshot(self.completion.result.inputs.snapshot), self.workspace.edits.revision
        )

    def fit(self, *, settings: bool = False) -> None:
        """Review once; subsequent runs prepare all working peaks afresh off Qt."""
        w = self.workspace
        if self.busy or getattr(w, "inputs_loading", False) or not w.peaks or not w.commit_fields():
            return
        w.cancel_gesture()
        if settings or not self.reviewed:
            dialog = DatasetFitDialog(w, self.configuration)
            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted or dialog.prepared is None:
                return
            self.configuration = dialog.configuration
            self.reviewed = True
            self.start(dialog.prepared)
            return
        self._generation += 1
        self._submitted_revision = w.edits.revision
        self._preparing = True
        self._started = perf_counter()
        self.attempt = "Preparing complete dataset"
        self.failure = ""
        spectra, peaks = w.dataset.spectra, w.peaks
        settings_copy = self.configuration.model_copy(deep=True)
        sources = w.dataset.sources

        def prepare():
            value = prepare_dataset_fit(spectra, peaks, settings_copy)
            return replace(value, snapshot=replace(value.snapshot, source_files=sources))

        self._worker.submit("fit", self._generation, prepare)
        self.update()

    def _prepared(self, kind, generation, outcome) -> None:
        if generation != self._generation:
            return
        if kind == "publication":
            self._publish(outcome)
            return
        self._preparing = False
        if isinstance(outcome, Exception):
            self.attempt = "Failed during preparation"
            self.failure = str(outcome)
            self.update()
            return
        self.start(outcome)

    def start(self, prepared: PreparedDatasetFit) -> None:
        """Submit exactly the reviewed snapshot, independently of view/selection."""
        if prepared.snapshot != self.snapshot(prepared.snapshot):
            self.attempt = "Failed: obsolete preparation; review settings again"
            self.update()
            return
        self._submitted_revision = self.workspace.edits.revision
        self._started = perf_counter()
        self.failure = ""
        self.attempt = "Fitting complete dataset"
        self._attempt_id = uuid4().hex
        self._attempt_parent = self.result.run_id if self.result else None
        self._attempt_initial = prepared.snapshot.peaks
        self.process.start(prepared)
        self.update()

    def cancel(self) -> None:
        """Acknowledge immediately; no canceled work can publish positions or results."""
        self._generation += 1
        self._worker.cancel("fit")
        self._worker.cancel("publication")
        self._preparing = False
        if self.process.busy:
            self._record_attempt("canceled", "Canceled by the scientist")
        self.process.cancel()
        self.attempt = "Canceled · previous result retained"
        self.update()

    def _finished(self, _generation: int, outcome: object) -> None:
        if self.process.state == "canceled":
            self.attempt = "Canceled · previous result retained"
        elif isinstance(outcome, FitProcessResult) and isinstance(
            outcome.scientific, DatasetFitResult
        ):
            self._preparing = True
            self.attempt = "Validating completion"
            result = outcome.scientific
            current = self.snapshot(result.inputs.snapshot)
            revision = self.workspace.edits.revision
            submitted = self._submitted_revision

            history = tuple(self.history)
            journal = tuple(self.journal)
            header = (
                dataset_header(self.workspace.dataset)
                if "spectrum" in dict(self.workspace.dataset.sources)
                else None
            )
            protected = frozenset(self.protected)

            dataset = self.workspace.dataset
            generation = self._generation
            parent_id = self._attempt_parent

            def complete():
                completion = dataset_completion(
                    result, current, submitted_revision=submitted, manual_revision=revision
                )
                entry = dict(journal_entry(completion), parent_run_id=parent_id)
                if header is None:
                    return completion, (completion,), (*journal, entry)
                self.attempt = "Preserving accepted evidence"
                analysis = Analysis(
                    dataset,
                    header,
                    completion.working_peaks,
                    current.configuration,
                    revision,
                    (*history, completion),
                    journal=(*journal, entry),
                    protected=protected,
                )
                saved = self._session.preserve(
                    analysis, canceled=lambda: generation != self._generation
                )
                return completion, saved.runs, saved.journal

            self._worker.submit("publication", self._generation, complete)
        else:
            self.attempt = "Failed · previous result retained"
            self.failure = str(outcome)
            self._record_attempt("failed", self.failure)
        self.update()

    def _publish(self, outcome) -> None:
        self._preparing = False
        if isinstance(outcome, Exception):
            self.attempt = "Failed during completion · previous result retained"
            self.failure = str(outcome)
            self.update()
            return
        completion, history, journal = outcome
        w = self.workspace
        if (
            self._submitted_revision != w.edits.revision
            or self.snapshot(completion.result.inputs.snapshot) != completion.result.inputs.snapshot
        ):
            self.attempt = "Failed: scientific inputs changed before publication"
            self.update()
            return
        following = w.canvas.pin_peak_id
        # No signal or event-loop yield occurs between installing W and its run.
        self.completion = completion
        self.journal = list(journal)
        self._attempt_id = None
        self.history = list(history)
        if self._session.path:
            owner = self._session
            keep_ids = {run_id(run) for run in history}
            self._worker.cleanup_after_work(lambda: owner.prune(keep_ids))
        w.edits.accept_completion(completion.working_peaks)
        self.attempt = ""
        w._refresh()
        if following:
            peak = next((p for p in w.peaks if p.id == following), None)
            if peak:
                w.canvas.sample(*w.canvas.display(peak))
                w.canvas.pin_peak_id = following
        usable = any(c.usable for c in completion.result.outcome.clusters)
        w.set_mode(1 if usable else 0)
        self.update()

    def update(self) -> None:
        """One compact status line; inspection and navigation stay available."""
        w = self.workspace
        busy = self.busy
        w.canvas.editing_enabled = not busy
        w.inspector.setEnabled(not busy)
        w.add_button.setEnabled(not busy)
        if not busy:
            w._update_actions()
        for action in (w.undo_action, w.redo_action, w.add_action, w.delete_action):
            if busy:
                action.setEnabled(False)
        w.fit_button.setEnabled(bool(w.peaks) and not busy and not w.inputs_loading)
        w.settings_button.setEnabled(bool(w.peaks) and not busy and not w.inputs_loading)
        # While a fit runs its settings cannot change; the run line needs the room.
        w.settings_button.setVisible(not busy)
        w.cancel_fit.setVisible(busy)
        usable = (
            self.current
            and self.result is not None
            and any(c.usable for c in self.result.outcome.clusters)
        )
        for button in w.mode_buttons[1:]:
            button.setEnabled(usable)
            button.setToolTip("" if usable else "Requires a current result with usable clusters")
        if not self.current and w.canvas.comparison_result is not None:
            w.set_mode(0)
        if busy:
            stage = self.process.progress if self.process.busy else self.attempt
            if self.process.state == "canceling":
                stage = "Canceling and reaping fit process"
            text = f"{stage} · {perf_counter() - self._started:.1f} s · Navigation available"
        elif self.result is not None:
            # Currentness first: a narrow toolbar elides only the tail.
            text = (
                f"Run {self.result.run_id[:8]} · {'Current' if self.current else 'Outdated'} · "
                + (self.attempt + " · " if self.attempt else "")
                + dataset_summary(self.result, warnings=False)
            )
        else:
            text = (
                self.attempt
                or f"Not fitted · {len(w.peaks)} peaks · {w.dataset.spectra.data.shape[0]} planes"
            )
        w.summary.setText(text)
        flagged = (
            sum(
                c.analytical_evaluation is not None
                and c.analytical_evaluation.uncertainty_warning is not None
                for c in self.result.outcome.clusters
            )
            if self.result is not None and not busy
            else 0
        )
        w.warnings.setProperty("count", flagged)
        w.warnings.setToolTip(
            f"{flagged} clusters with an engine uncertainty warning; "
            "Review lists them cluster by cluster"
        )
        w.warnings.setVisible(bool(flagged))
        w.update_settings_summary()
        w.summary.setToolTip(text + ("\n" + self.failure if self.failure else ""))
        w._update_inspector()
        self.state_changed.emit()

    def next_warning(self, _url) -> None:
        """Explicitly navigate questionable clusters without isolating their models."""
        result = self.result
        if result is None:
            return
        warnings = [
            c
            for c in result.outcome.clusters
            if not c.usable
            or not c.optimizer_provenance.converged
            or (c.analytical_evaluation and c.analytical_evaluation.uncertainty_warning)
        ]
        if not warnings:
            return
        identities = {i.engine_name: i.working_id for i in result.inputs.identities}
        index = next(
            (
                index
                for index, cluster in enumerate(warnings)
                if self.workspace.selected_id in [identities[n] for n in cluster.peak_names]
            ),
            -1,
        )
        cluster = warnings[(index + 1) % len(warnings)]
        self.workspace.select_peak(identities[cluster.peak_names[0]])

    def export(self) -> bool:
        """Export retained evidence with its currentness explicitly labeled."""
        if self.busy or self.result is None or not self.workspace.commit_fields():
            return False
        if (
            not self.current
            and QtWidgets.QMessageBox.question(
                self.workspace,
                "Export outdated result?",
                "This run is Outdated. Export its original scientific evidence "
                "and starting inputs?",
            )
            != QtWidgets.QMessageBox.StandardButton.Yes
        ):
            return False
        path = choose_and_export(
            self.workspace,
            self.result,
            completion=self.completion,
            current=self.current,
            dataset=self.workspace.dataset,
        )
        if path:
            self.exported.add(self.result.run_id)
            if self.completion and self.workspace.peaks == self.completion.working_peaks:
                # The shared coordinate table contains this exact stable-ID coordinate
                # tuple. Its successful export also preserves the working peaks.
                self.workspace.edits.mark_saved()
                self.workspace.changed.emit()
            self.workspace.status.setText(f"Exported retained run to {path}")
        return path is not None

    def _record_attempt(self, status, diagnostic):
        if self._attempt_id is not None:
            self.journal.append(
                {
                    "run_id": self._attempt_id,
                    "parent_run_id": self._attempt_parent,
                    "completed_at": datetime.now(UTC).isoformat(),
                    "status": status,
                    "configuration": self.configuration.model_dump_json(),
                    "initial_peaks": [
                        {"id": p.id, "name": p.name, "positions": p.positions}
                        for p in self._attempt_initial
                    ],
                    "diagnostics": diagnostic,
                    "parameters": [],
                }
            )
            self._attempt_id = None

    def stop(self) -> None:
        """Stop workers and invalidate queued deliveries without writing session files."""
        self.timer.stop()
        self._generation += 1
        self.process.close()
        self._worker.close(finalize=self._session.close)
