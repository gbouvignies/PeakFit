"""Whole-dataset execution and atomic completion in the existing editor workspace."""

from concurrent.futures import CancelledError
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
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
from peakfit.gui.project_results import completed, journal_entry, run_name
from peakfit.gui.session_evidence import SessionEvidence

if TYPE_CHECKING:
    from peakfit.gui.editor.workspace import EditorWorkspace
    from peakfit.gui.fitting import FitInputSnapshot
    from peakfit.gui.peaks import WorkingPeak


class FailureStage(StrEnum):
    """The stage of a fit attempt that failed, as the failure section names it."""

    PREPARATION = "Preparation"
    FITTING = "Fitting"
    COMPLETION = "Completion"


@dataclass(frozen=True)
class Comparison:
    """The run whose model and residual the spectral views compare with, if any.

    A current run compares directly. After working inputs change, the Outdated run
    stays comparable as *historical* evidence while it was recorded on the same
    spectrum, axes and planes; ``reason`` says why it was withdrawn otherwise.
    ``run`` is how captions name the run (its number); ``added`` holds working IDs
    that run did not fit; ``departed`` the run's adopted peaks that were moved or
    deleted since, at their recorded positions.
    """

    result: DatasetFitResult | None = None
    run: str = ""
    historical: bool = False
    reason: str = ""
    added: frozenset[str] = frozenset()
    departed: tuple[WorkingPeak, ...] = field(default=())

    @property
    def label(self) -> str:
        """The recorded run identity every historical display carries; empty if current."""
        if self.result is None or not self.historical:
            return ""
        return f"Run {self.run} · historical"


def comparison_mismatch(recorded: FitInputSnapshot, working: FitInputSnapshot) -> str:
    """Why a run's model cannot be compared with the working data, or an empty string.

    The model of plane i is compared with measured plane i on the same axes, so the
    spectrum, its axes and its plane mapping must be the ones the run recorded. A
    plane the run excluded is measured but has no model (``fit_display.not_fitted``).
    """
    if recorded.spectral_parameters != working.spectral_parameters:
        return "recorded with different spectral axes"
    if recorded.plane_values != working.plane_values:
        return "recorded with different plane values"
    if (recorded.spectrum_id, recorded.data_shape, recorded.source_files) != (
        working.spectrum_id,
        working.data_shape,
        working.source_files,
    ):
        return "recorded on a different spectrum"
    return ""


def default_settings() -> PeakFitConfig:
    """The GUI's fit settings for a new analysis: two optimizer passes (spec §7)."""
    return PeakFitConfig(fitting=FitConfig(refine_iterations=2))


def _warmed(completion: DatasetCompletion) -> DatasetCompletion:
    """A loaded run with what every view shares already computed, off the GUI thread."""
    completion.result.warm()
    return completion


class DatasetFitting(QtCore.QObject):
    """One active immutable result, one fit and fresh whole-dataset preparation.

    ``state_changed`` is emitted on every transition of the job or result state and
    at no other time; the workspace renders from that state.
    """

    state_changed = QtCore.Signal()

    def __init__(self, workspace: EditorWorkspace) -> None:
        super().__init__(workspace)
        self.workspace = workspace
        # The first Fit of a session reveals the Model page unless it was already shown.
        self.model_seen = False
        # Preparation of the working inputs, refreshed after every committed change.
        self.preview: PreparedDatasetFit | None = None
        self.preview_error = ""
        self._preview_inputs = None
        # Retained runs read for run-to-run changes: result, reason, or "loading".
        # One historical result at most stays resident (specification §4).
        self._runs_read: dict[str, DatasetFitResult | str] = {}
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
        # The last attempt's failure: its reason and the stage that failed.
        self.failure = ""
        self.failure_stage: FailureStage | None = None
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

    @property
    def configuration(self) -> PeakFitConfig:
        """The committed working settings: the edit history's, so Undo restores them."""
        return self.workspace.edits.settings

    def restore(self, completion, runs, journal, protected) -> None:
        """Adopt an opened analysis's active run and retained evidence."""
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
    def failed(self) -> bool:
        """Whether the last attempt failed and no new attempt is under way."""
        return bool(self.failure) and not self.busy

    def _fail(self, stage: FailureStage, reason: str) -> None:
        """Record a failed attempt; the working peaks and the active run stay as they are."""
        self.attempt = f"Failed · {self._retained}"
        self.failure_stage = stage
        self.failure = reason

    @property
    def _retained(self) -> str:
        """What a canceled or failed attempt leaves: the previous result, or none."""
        return "previous result retained" if self.result is not None else "no result yet"

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
        run = run_name(result.run_id, self.journal)
        if self.current:
            return Comparison(result, run)
        recorded = result.inputs.snapshot
        reason = comparison_mismatch(recorded, self.snapshot(recorded))
        if reason:
            return Comparison(reason=f"Run {run} was {reason}")
        fitted = {p.id: p for p in completion.working_peaks}
        working_peaks = {p.id: p for p in self.workspace.peaks}
        return Comparison(
            result,
            run,
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
        """Submit the prepared working inputs; the first Fit may reveal the Model page."""
        w = self.workspace
        if self.busy or w.inputs_loading or not w.peaks or not w.commit_fields():
            return
        if not self.model_seen or self.blocked or w.panel.model_page.invalid:
            w.show_model_page()
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
        self.failure, self.failure_stage = "", None
        spectra, peaks = w.dataset.spectra, w.peaks
        settings_copy = self.configuration.model_copy(deep=True)
        sources = w.dataset.sources
        source = self._source()

        def prepare():
            value = prepare_dataset_fit(spectra, peaks, settings_copy, source)
            return replace(value, snapshot=replace(value.snapshot, source_files=sources))

        self._worker.submit("fit", self._generation, prepare)
        self.state_changed.emit()

    def parent_id(self) -> str | None:
        """The run the active run was refitted from, as its journal entry records it."""
        result = self.result
        if result is None:
            return None
        entry = next((e for e in reversed(self.journal) if e.get("run_id") == result.run_id), None)
        return (entry.get("parent_run_id") or None) if entry else None

    def completed_runs(self) -> list[dict]:
        """The journal's completed runs, oldest first; failed or canceled attempts are not."""
        return [e for e in self.journal if completed(e)]

    def full_evidence(self, identity: str) -> bool:
        """Whether a run's full evidence is retained, rather than its summary only."""
        return any(run_id(r) == identity for r in self.history)

    def run_result(self, identity: str) -> DatasetFitResult | str | None:
        """A retained run's recorded result, why it cannot be read, or None while loading.

        Stored history is loaded off Qt; nothing is refitted. Only the last result read
        is kept, so choosing baselines one after another never accumulates results.
        """
        known = self._runs_read.get(identity)
        if known is None:
            run = next((r for r in self.history if run_id(r) == identity), None)
            if run is None:
                known = "its full evidence is no longer retained"
            elif isinstance(run, DatasetCompletion):
                known = run.result
            else:
                # A new request replaces a pending one, which is asked for again if needed.
                self._runs_read = {k: v for k, v in self._runs_read.items() if v != "loading"}
                self._runs_read[identity] = "loading"
                self._worker.submit("run", 0, lambda: (identity, _warmed(run.load())))
                return None
            self._keep_read(identity, known)
        return None if known == "loading" else known

    def _keep_read(self, identity: str, known: DatasetFitResult | str) -> None:
        """Remember a run read, dropping any other historical result already held."""
        self._runs_read = {
            key: value for key, value in self._runs_read.items() if isinstance(value, str)
        }
        self._runs_read[identity] = known

    def _prepared(self, kind, generation, outcome) -> None:
        if kind == "run":
            if isinstance(outcome, Exception):
                # The failing run is unknown here; mark every pending load as failed.
                for key, value in self._runs_read.items():
                    if value == "loading":
                        self._runs_read[key] = f"its recorded evidence could not be read: {outcome}"
            else:
                identity, completion = outcome
                self._keep_read(identity, completion.result)
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
            self._fail(FailureStage.PREPARATION, str(outcome))
            self.state_changed.emit()
            return
        self.start(outcome)

    def start(self, prepared: PreparedDatasetFit) -> None:
        """Submit exactly the reviewed snapshot, independently of view/selection."""
        if prepared.snapshot != self.snapshot(prepared.snapshot) or (
            prepared.continuation is None or prepared.continuation.run_id != self._source_id
        ):
            self._fail(FailureStage.PREPARATION, "Inputs changed during preparation; fit again.")
            self.state_changed.emit()
            return
        self._submitted_revision = self.workspace.edits.revision
        self._started = perf_counter()
        self.failure, self.failure_stage = "", None
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
        self.attempt = f"Canceled · {self._retained}"
        self.state_changed.emit()

    def _finished(self, _generation: int, outcome: object) -> None:
        if self.process.state == "canceled":
            self.attempt = f"Canceled · {self._retained}"
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
                result.warm()  # what every view shares, computed here rather than on Qt
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
            self._fail(FailureStage.FITTING, str(outcome))
            self._record_attempt("failed", self.failure)
        self.state_changed.emit()

    def _publish(self, outcome) -> None:
        self._preparing = False
        if isinstance(outcome, Exception):
            self._fail(FailureStage.COMPLETION, str(outcome))
            self.state_changed.emit()
            return
        completion, history, journal, recovery_error = outcome
        w = self.workspace
        if (
            self._submitted_revision != w.edits.revision
            or self.snapshot(completion.result.inputs.snapshot) != completion.result.inputs.snapshot
        ):
            self._fail(
                FailureStage.COMPLETION, "Scientific inputs changed before publication; fit again."
            )
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
        self.attempt = self.failure = ""
        self.failure_stage = None
        w._refresh()
        if following:
            peak = next((p for p in w.peaks if p.id == following), None)
            if peak:
                w.canvas.sample(*w.canvas.display(peak))
                w.canvas.pin_peak_id = following
        usable = any(c.usable for c in completion.result.outcome.clusters)
        w.set_mode(1 if usable else 0)
        self.state_changed.emit()

    def next_warning(self, *, backward: bool = False) -> None:
        """Move to the next (or previous) warned cluster the navigator shows, never a hidden one."""
        w = self.workspace
        target = w.navigator.next_warned(w.selected_id, backward=backward)
        if target is not None:
            w.select_peak(target, cluster=True, center=True)
        elif any(r.warnings for r in w.reviews()):
            w.status.setText("No cluster with a numerical warning is shown by the current filter")

    def next_marked(self, *, backward: bool = False) -> None:
        """N in the cluster list: the next cluster the Show choice lists, or a warned one."""
        w = self.workspace
        target = w.navigator.next_marked(w.selected_id, backward=backward)
        if target is not None:
            w.select_peak(target, cluster=True, center=True)
        elif not w.navigator.showing_all:
            w.status.setText("The Show choice lists no cluster")
        elif w.reviews():
            w.status.setText(
                "No cluster has a numerical warning. For N to step through other marks, "
                "choose one in Show"
            )

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
