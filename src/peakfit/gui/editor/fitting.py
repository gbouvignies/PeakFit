"""Whole-dataset execution and atomic completion in the existing editor workspace."""

from concurrent.futures import CancelledError
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import uuid4

from PySide6 import QtCore, QtWidgets

from peakfit.engine.domain.config import FitConfig, PeakFitConfig
from peakfit.gui.background import ViewerComputation
from peakfit.gui.dataset_fit import DatasetFitResult
from peakfit.gui.export_dialog import choose_and_export
from peakfit.gui.failures import WorkerError, failure_message, log_failure
from peakfit.gui.fit_positions import DatasetCompletion, continuation_source, dataset_completion
from peakfit.gui.fit_process import FitProcess
from peakfit.gui.fitting import PreparedDatasetFit, prepare_dataset_fit
from peakfit.gui.project import Analysis, dataset_header, run_id
from peakfit.gui.project_results import journal_entry
from peakfit.gui.session_evidence import SessionEvidence

if TYPE_CHECKING:
    from peakfit.gui.editor.workspace import EditorWorkspace
    from peakfit.gui.fitting import FitInputSnapshot
    from peakfit.gui.peaks import WorkingPeak


@dataclass(frozen=True)
class Comparison:
    """The run whose model and residual the spectral views compare with, if any.

    A current run compares directly. After working inputs change, the Outdated run
    stays comparable as *historical* evidence while it was recorded on the same
    spectrum, axes and planes; ``reason`` says why it was withdrawn otherwise.
    ``added`` holds working IDs that run did not fit; ``departed`` the run's
    adopted peaks that were moved or deleted since, at their recorded positions.
    """

    result: DatasetFitResult | None = None
    historical: bool = False
    reason: str = ""
    added: frozenset[str] = frozenset()
    departed: tuple[WorkingPeak, ...] = field(default=())

    @property
    def label(self) -> str:
        """The recorded run identity every historical display carries; empty if current."""
        if self.result is None or not self.historical:
            return ""
        return f"Run {self.result.run_id[:8]} · historical"


def comparison_mismatch(recorded: FitInputSnapshot, working: FitInputSnapshot) -> str:
    """Why a run's model cannot be compared with the working data, or an empty string.

    The model of plane i is compared with measured plane i on the same axes, so the
    spectrum, its axes and its plane mapping must be the ones the run recorded.
    """
    if recorded.spectral_parameters != working.spectral_parameters:
        return "recorded with different spectral axes"
    if recorded.source_plane_indices != working.source_plane_indices:
        return "recorded on a different selection of planes"
    if recorded.plane_values != working.plane_values:
        return "recorded with different plane values"
    if (recorded.spectrum_id, recorded.data_shape, recorded.source_files) != (
        working.spectrum_id,
        working.data_shape,
        working.source_files,
    ):
        return "recorded on a different spectrum"
    return ""


class DatasetFitting(QtCore.QObject):
    """One active immutable result, one fit and fresh whole-dataset preparation.

    ``state_changed`` is emitted on every transition of the job or result state and
    at no other time; the workspace renders from that state.
    """

    state_changed = QtCore.Signal()

    def __init__(self, workspace: EditorWorkspace) -> None:
        super().__init__(workspace)
        self.workspace = workspace
        self.configuration = PeakFitConfig(fitting=FitConfig(refine_iterations=2))
        # The first Fit of a session reveals the setup unless it was already displayed.
        self.setup_seen = False
        # Preparation of the working inputs, refreshed after every committed change.
        self.preview: PreparedDatasetFit | None = None
        self.preview_error = ""
        self._preview_inputs = None
        # Parent runs read for run-to-run changes: result, reason, or "loading".
        self._parents: dict[str, DatasetFitResult | str] = {}
        self._preview_generation = 0
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(150)
        self._preview_timer.timeout.connect(self._prepare_preview)
        self.completion = None
        self.history = []
        self.journal = []
        self.protected = set()
        self._session = SessionEvidence()
        self._attempt_id = None
        self._attempt_parent = None
        self._attempt_initial = ()
        self.exported = set()
        # Why the accepted run is not in the recovery copy; cleared once it is, or saved.
        self.recovery_error = ""
        self.attempt = ""
        self.failure = ""
        self._preparing = False
        self._generation = 0
        self._submitted_revision = 0
        self._started = 0.0
        self._worker = ViewerComputation(self)
        self._worker.completed.connect(self._prepared)
        self.process = FitProcess(self)
        self.process.changed.connect(self.state_changed)
        self.process.finished.connect(self._finished)

    @property
    def _recovery_path(self):
        return self._session.path

    def restore(self, configuration, completion, runs, journal, protected) -> None:
        """Adopt an opened analysis's settings, active run and retained evidence."""
        self.configuration = configuration
        self.completion = completion
        self.history = list(runs)
        self.journal = list(journal)
        self.protected = set(protected)

    def adopt_saved(self, runs, journal) -> None:
        """Rebind history to the published evidence, then clean the scratch files."""
        self.history = list(runs)
        self.journal = list(journal)
        self.recovery_error = ""
        self._worker.cleanup_after_work(self._session.close)

    def hand_over_evidence(self, runs) -> SessionEvidence | None:
        """Hand scratch evidence that ``runs`` still link to the caller; keep a fresh owner."""
        owner = self._session
        if not owner.path or not any(getattr(r, "root", None) == owner.path for r in runs):
            return None
        self._session = SessionEvidence()
        return owner

    def keep_run(self, run_id: str, keep: bool) -> None:
        """Protect a run from history eviction, or release it."""
        if keep:
            self.protected.add(run_id)
        else:
            self.protected.discard(run_id)
        self.state_changed.emit()

    @property
    def busy(self) -> bool:
        """Protect scientific edits through preparation, execution and publication."""
        return self._preparing or self.process.busy

    @property
    def stage(self) -> str:
        """What the active job is doing, while busy."""
        if self.process.state == "canceling":
            return "Canceling and reaping fit process"
        return self.process.progress if self.process.busy else self.attempt

    @property
    def elapsed(self) -> float:
        """Seconds since the active job started."""
        return perf_counter() - self._started

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

    @property
    def comparison(self) -> Comparison:
        """The current run, else the Outdated run as historical evidence on the same data.

        Currentness is unchanged: a historical comparison never certifies the run.
        """
        completion = self.completion
        if completion is None:
            return Comparison()
        result = completion.result
        if self.current:
            return Comparison(result)
        recorded = result.inputs.snapshot
        working = replace(
            self.snapshot(recorded),
            source_plane_indices=tuple(self.workspace.dataset.spectra.source_plane_indices),
        )
        reason = comparison_mismatch(recorded, working)
        if reason:
            return Comparison(reason=f"Run {result.run_id[:8]} was {reason}")
        fitted = {p.id: p for p in completion.working_peaks}
        working_peaks = {p.id: p for p in self.workspace.peaks}
        return Comparison(
            result,
            historical=True,
            added=frozenset(working_peaks.keys() - fitted.keys()),
            departed=tuple(
                p
                for p in completion.working_peaks
                if p.id not in working_peaks or working_peaks[p.id].positions != p.positions
            ),
        )

    def _inputs(self):
        # The continuation source is an input: a new or reopened run re-prepares.
        return self.workspace.peaks, self.configuration.model_dump_json(), self._source_id

    @property
    def _source_id(self) -> str | None:
        return self.completion.result.run_id if self.completion else None

    def _source(self):
        """The active run's fitted state, which unchanged peaks continue from."""
        return continuation_source(self.completion) if self.completion else None

    def configure(self, configuration: PeakFitConfig) -> None:
        """Adopt edited working settings; an existing run becomes Outdated."""
        if self.busy or configuration == self.configuration:
            return
        self.configuration = configuration
        self.request_preview()
        self.state_changed.emit()

    def request_preview(self) -> None:
        """Hide an obsolete preparation and prepare the committed inputs afresh."""
        inputs = self._inputs()
        if inputs == self._preview_inputs:
            return
        self._preview_inputs = inputs
        self._preview_generation += 1
        self.preview = None
        self.preview_error = ""
        self._preview_timer.start()

    def _prepare_preview(self) -> None:
        w = self.workspace
        if not w.peaks:
            self.preview_error = "Load or add peaks before fitting the dataset."
            self.state_changed.emit()
            return
        spectra, peaks, sources = w.dataset.spectra, w.peaks, w.dataset.sources
        settings = self.configuration.model_copy(deep=True)
        source = self._source()

        def prepare():
            value = prepare_dataset_fit(spectra, peaks, settings, source)
            return replace(value, snapshot=replace(value.snapshot, source_files=sources))

        self._worker.submit("preview", self._preview_generation, prepare)

    @property
    def prepared(self) -> PreparedDatasetFit | None:
        """The preparation of exactly the current working inputs, if ready."""
        preview = self.preview
        if (
            preview is None
            or preview.snapshot != self.snapshot(preview.snapshot)
            or preview.continuation is None
            or preview.continuation.run_id != self._source_id
        ):
            return None
        return preview

    @property
    def blocked(self) -> str:
        """Why the current working inputs cannot be fitted, from their preparation."""
        return self.preview_error if self._preview_inputs == self._inputs() else ""

    def fit(self) -> None:
        """Submit the prepared working inputs; the first Fit may reveal the setup."""
        w = self.workspace
        if self.busy or w.inputs_loading or not w.peaks or not w.commit_fields():
            return
        if not self.setup_seen or self.blocked:
            w.show_fit_setup()
            return
        w.cancel_gesture()
        prepared = self.prepared
        if prepared is not None:
            self.start(prepared)
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
        source = self._source()

        def prepare():
            value = prepare_dataset_fit(spectra, peaks, settings_copy, source)
            return replace(value, snapshot=replace(value.snapshot, source_files=sources))

        self._worker.submit("fit", self._generation, prepare)
        self.state_changed.emit()

    def parent_result(self) -> tuple[str | None, DatasetFitResult | str | None]:
        """The active run's parent: (its ID, its result, a reason, or None while loading).

        Stored history is loaded once off Qt and cached; nothing is refitted.
        """
        result = self.result
        if result is None:
            return None, None
        entry = next((e for e in reversed(self.journal) if e.get("run_id") == result.run_id), None)
        parent_id = entry.get("parent_run_id") if entry else None
        if not parent_id:
            return None, None
        known = self._parents.get(parent_id)
        if known is None:
            run = next((r for r in self.history if run_id(r) == parent_id), None)
            if run is None:
                known = "its recorded evidence is no longer retained"
            elif isinstance(run, DatasetCompletion):
                known = run.result
            else:
                self._parents[parent_id] = "loading"
                self._worker.submit("parent", 0, lambda: (parent_id, run.load()))
                return parent_id, None
            self._parents[parent_id] = known
        if known == "loading":
            return parent_id, None
        return parent_id, known

    def _prepared(self, kind, generation, outcome) -> None:
        if kind == "parent":
            if isinstance(outcome, Exception):
                # The failing run is unknown here; mark every pending load as failed.
                for key, value in self._parents.items():
                    if value == "loading":
                        self._parents[key] = f"its recorded evidence could not be read: {outcome}"
            else:
                parent_id, completion = outcome
                self._parents[parent_id] = completion.result
            self.state_changed.emit()
            return
        if kind == "preview":
            if generation == self._preview_generation:
                if isinstance(outcome, Exception):
                    self.preview_error = str(outcome)
                else:
                    self.preview = outcome
                self.state_changed.emit()
            return
        if generation != self._generation:
            return
        if kind == "publication":
            self._publish(outcome)
            return
        self._preparing = False
        if isinstance(outcome, Exception):
            self.attempt = "Failed during preparation"
            self.failure = str(outcome)
            self.state_changed.emit()
            return
        self.start(outcome)

    def start(self, prepared: PreparedDatasetFit) -> None:
        """Submit exactly the reviewed snapshot, independently of view/selection."""
        if prepared.snapshot != self.snapshot(prepared.snapshot) or (
            prepared.continuation is None or prepared.continuation.run_id != self._source_id
        ):
            self.attempt = "Failed: inputs changed during preparation; fit again"
            self.state_changed.emit()
            return
        self._submitted_revision = self.workspace.edits.revision
        self._started = perf_counter()
        self.failure = ""
        self.attempt = "Fitting complete dataset"
        self._attempt_id = uuid4().hex
        self._attempt_parent = self.result.run_id if self.result else None
        self._attempt_initial = prepared.snapshot.peaks
        self.process.start(prepared)
        self.state_changed.emit()

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
        self.state_changed.emit()

    def _finished(self, _generation: int, outcome: object) -> None:
        if self.process.state == "canceled":
            self.attempt = "Canceled · previous result retained"
        elif isinstance(outcome, DatasetFitResult):
            self._preparing = True
            result = outcome
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
            self.attempt = (
                "Validating completion" if header is None else "Preserving accepted evidence"
            )

            dataset = self.workspace.dataset
            generation = self._generation
            parent_id = self._attempt_parent

            def complete():
                completion = dataset_completion(
                    result, current, submitted_revision=submitted, manual_revision=revision
                )
                entry = dict(journal_entry(completion), parent_run_id=parent_id)
                if header is None:
                    return completion, (completion,), (*journal, entry), ""
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
                try:
                    saved = self._session.preserve(
                        analysis, canceled=lambda: generation != self._generation
                    )
                except CancelledError:
                    raise
                except Exception as error:
                    # The run is validated; losing its recovery copy must not lose the
                    # run. It stays resident until a later copy or Save holds it.
                    if not isinstance(error, WorkerError):
                        log_failure("Recovery copy", error)
                    return completion, analysis.runs, analysis.journal, failure_message(error)
                return completion, saved.runs, saved.journal, ""

            self._worker.submit("publication", self._generation, complete)
        else:
            self.attempt = "Failed · previous result retained"
            self.failure = str(outcome)
            self._record_attempt("failed", self.failure)
        self.state_changed.emit()

    def _publish(self, outcome) -> None:
        self._preparing = False
        if isinstance(outcome, Exception):
            self.attempt = "Failed during completion · previous result retained"
            self.failure = str(outcome)
            self.state_changed.emit()
            return
        completion, history, journal, recovery_error = outcome
        w = self.workspace
        if (
            self._submitted_revision != w.edits.revision
            or self.snapshot(completion.result.inputs.snapshot) != completion.result.inputs.snapshot
        ):
            self.attempt = "Failed: scientific inputs changed before publication"
            self.state_changed.emit()
            return
        following = w.canvas.pin_peak_id
        # No signal or event-loop yield occurs between installing W and its run.
        self.completion = completion
        self.journal = list(journal)
        self._attempt_id = None
        self.history = list(history)
        self.recovery_error = recovery_error
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
        self.state_changed.emit()

    def next_warning(self) -> None:
        """Move to the next warned cluster the navigator shows, never a hidden one."""
        w = self.workspace
        target = w.navigator.next_warned(w.selected_id)
        if target is not None:
            w.select_peak(target, cluster=True)
        elif any(r.warnings for r in w.reviews()):
            w.status.setText("No cluster with a numerical warning is shown by the current filter")

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
            # Exporting results never marks the analysis saved.
            self.exported.add(self.result.run_id)
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
        self._generation += 1
        self.process.close()
        self._worker.close(finalize=self._session.close)
