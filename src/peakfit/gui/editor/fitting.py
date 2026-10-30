"""Whole-dataset execution and atomic completion in the existing editor workspace."""

import io
import pickle
from dataclasses import replace
from html import escape
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter
from types import MappingProxyType
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtWidgets

from peakfit.engine.domain.config import FitConfig, PeakFitConfig
from peakfit.gui.background import ViewerComputation
from peakfit.gui.dataset_dialog import DatasetFitDialog
from peakfit.gui.dataset_fit import DatasetFitResult, _reduce_mapping, dataset_summary
from peakfit.gui.export_dialog import choose_and_export
from peakfit.gui.fit_positions import dataset_completion
from peakfit.gui.fit_process import FitProcess, FitProcessResult
from peakfit.gui.fitting import PreparedDatasetFit, prepare_dataset_fit

if TYPE_CHECKING:
    from peakfit.gui.editor.workspace import EditorWorkspace
    from peakfit.gui.fitting import FitInputSnapshot


_RETAINED_RUNS = 10


def retain_result(result: DatasetFitResult, directory: Path) -> Path:
    """Spool trusted session evidence before replacing the resident result.

    This private pickle is never accepted as an imported analysis. It preserves
    exact inputs and reconstruction payloads for recovery during this session.
    """
    stream = io.BytesIO()
    writer = pickle.Pickler(stream, protocol=5)
    writer.dispatch_table = {MappingProxyType: _reduce_mapping}
    writer.dump(result)
    path = directory / f"{result.run_id}.pickle"
    temporary = path.with_suffix(".pending")
    temporary.write_bytes(stream.getvalue())
    temporary.replace(path)
    return path


class DatasetFitting(QtCore.QObject):
    """One active result, one fit, fresh preparation and a bounded session spool."""

    state_changed = QtCore.Signal()

    def __init__(self, workspace: EditorWorkspace) -> None:
        super().__init__(workspace)
        self.workspace = workspace
        self.configuration = PeakFitConfig(fitting=FitConfig(refine_iterations=2))
        self.reviewed = False
        self.completion = None
        self.exported = set()
        self.attempt = ""
        self.failure = ""
        self._preparing = False
        self._generation = 0
        self._submitted_revision = 0
        self._started = 0.0
        self._diagnostic_text = ""
        self._review_timer = QtCore.QTimer(self)
        self._review_timer.setSingleShot(True)
        self._review_timer.timeout.connect(self._reveal_review)
        self._session = Path(mkdtemp(prefix="peakfit-session-"))
        self.retained: list[tuple[str, Path]] = []
        self._worker = ViewerComputation(self)
        self._worker.completed.connect(self._prepared)
        self.process = FitProcess(self)
        self.process.changed.connect(self.update)
        self.process.finished.connect(self._finished)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.update)
        self.timer.start()
        self.review = QtWidgets.QTextBrowser()
        self.review.setAccessibleName("Completed fit diagnostics")
        self.review.setFixedHeight(180)
        self.review.setOpenLinks(False)
        self.review.anchorClicked.connect(self.next_warning)
        self.review.hide()
        inspector_layout = workspace.inspector.layout()
        assert isinstance(inspector_layout, QtWidgets.QGridLayout)
        inspector_layout.addWidget(self.review, 6, 0, 1, 2)
        workspace.changed.connect(self.update)

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
            if kind == "publication" and isinstance(outcome, tuple):
                outcome[1].unlink(missing_ok=True)
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
        self.process.start(prepared)
        self.update()

    def cancel(self) -> None:
        """Acknowledge immediately; no canceled work can publish positions or results."""
        self._generation += 1
        self._worker.cancel("fit")
        self._worker.cancel("publication")
        self._preparing = False
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
            self.attempt = "Validating completion and retaining evidence"
            result = outcome.scientific
            current = self.snapshot(result.inputs.snapshot)
            revision = self.workspace.edits.revision
            submitted = self._submitted_revision
            directory = Path(self._session)

            def complete():
                completion = dataset_completion(
                    result, current, submitted_revision=submitted, manual_revision=revision
                )
                path = retain_result(result, directory)
                return completion, path

            self._worker.submit("publication", self._generation, complete)
        else:
            self.attempt = "Failed · previous result retained"
            self.failure = str(outcome)
        self.update()

    def _publish(self, outcome) -> None:
        self._preparing = False
        if isinstance(outcome, Exception):
            self.attempt = "Failed during completion · previous result retained"
            self.failure = str(outcome)
            self.update()
            return
        completion, path = outcome
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
        w.edits.accept_completion(completion.working_peaks)
        self.retained.append((completion.result.run_id, path))
        try:
            self._prune()
        except OSError as error:
            self.failure = f"Could not prune temporary evidence: {error}"
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

    def _prune(self) -> None:
        # The active payload always survives, even if it exceeds the disk budget.
        while len(self.retained) > 1 and (
            len(self.retained) > _RETAINED_RUNS
            or sum(path.stat().st_size for _, path in self.retained) > 1024**3
        ):
            _, path = self.retained.pop(0)
            path.unlink()

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
        w.cancel_fit.setVisible(busy)
        w.export_fit.setEnabled(self.result is not None and not busy)
        w.review_fit.setEnabled(self.result is not None or bool(self.failure))
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
            text = (self.attempt + " · " if self.attempt else "") + (
                f"Run {self.result.run_id[:8]} · {'Current' if self.current else 'Outdated'} · "
                + dataset_summary(self.result)
            )
        else:
            text = (
                self.attempt
                or f"Not fitted · {len(w.peaks)} peaks · {w.dataset.spectra.data.shape[0]} planes"
            )
        w.summary.setText(text)
        w.summary.setToolTip(text + ("\n" + self.failure if self.failure else ""))
        if self.review.isVisible():
            self.show_diagnostics()
        self.state_changed.emit()

    def toggle_review(self) -> None:
        """Reveal diagnostics in the rail without changing the spectral viewport."""
        self.review.setVisible(not self.review.isVisible())
        if self.review.isVisible():
            self.workspace.peaks_button.setChecked(True)
            self.workspace.toggle_rail()
            self.show_diagnostics()
            self._review_timer.start(0)

    def _reveal_review(self) -> None:
        # Layout computes the newly visible review height on the next Qt turn.
        bar = self.workspace.rail.inspector_scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

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
        self.show_diagnostics()
        self._review_timer.start(0)

    def show_diagnostics(self) -> None:
        """Keep convergence, usability and uncertainty separate and inspectable."""
        lines = [self.failure] if self.failure else []
        result = self.result
        if result:
            lines += [
                f"Run {result.run_id[:8]} · {'Current' if self.current else 'Outdated'}",
            ]
            names = {i.engine_name: i.assignment for i in result.inputs.identities}
            selected = next(
                (
                    i.engine_name
                    for i in result.inputs.identities
                    if i.working_id == self.workspace.selected_id
                ),
                None,
            )
            clusters = [c for c in result.outcome.clusters if selected in c.peak_names]
            if not clusters:
                clusters = [
                    c
                    for c in result.outcome.clusters
                    if not c.usable
                    or not c.optimizer_provenance.converged
                    or (c.analytical_evaluation and c.analytical_evaluation.uncertainty_warning)
                ][:1]
            if not clusters:
                lines.append("Select a peak to inspect its recorded cluster diagnostics.")
            for cluster in clusters:
                evaluation = cluster.analytical_evaluation
                lines += [
                    f"\nCluster {cluster.cluster_id}: "
                    + ", ".join(names[n] for n in cluster.peak_names),
                    f"{'Usable' if cluster.usable else 'Unusable'} · "
                    f"{'Converged' if cluster.optimizer_provenance.converged else 'Not converged'}",
                    cluster.optimizer_provenance.termination_message or "No termination message",
                ]
                if cluster.unusable_reason:
                    lines.append(cluster.unusable_reason)
                if evaluation:
                    samples = next(
                        s
                        for s in result.completed_fit.samples
                        if s.cluster_id == cluster.cluster_id
                    )
                    varying = sum(p.vary for p in cluster.final_nonlinear_parameters)
                    lines.append(
                        f"{len(samples.grid_indices[0])} samples/plane · "
                        f"{varying} varying nonlinear parameters"
                    )
                    lines.append(f"Statistical noise: {result.outcome.noise:.9g} a.u.")
                    lines.append(f"Reduced χ²: {evaluation.statistics.reduced_chi_squared:.6g}")
                    if evaluation.uncertainty_warning:
                        lines.append(evaluation.uncertainty_warning)
            lines.append(
                "Recorded optimizer statistics use corrected fitting samples; "
                "the displayed residual uses native data minus the total model."
            )
            lines.extend(result.warnings)
            assert self.completion is not None
            lines.extend(self.completion.warnings)
            lines.append(f"Session evidence: {len(self.retained)} full runs in {self._session}")
            lines.append(
                "Temporary recovery only; export results before closing. "
                "Analysis reopening is a later milestone."
            )
        text = "\n".join(lines)
        if self._diagnostic_text != text:
            self._diagnostic_text = text
            self.review.setHtml(
                '<a href="next">Next cluster needing review</a><br>'
                + escape(text).replace("\n", "<br>")
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
            self.workspace, self.result, completion=self.completion, current=self.current
        )
        if path:
            self.exported.add(self.result.run_id)
            self.workspace.status.setText(f"Exported retained run to {path}")
        return path is not None

    def stop(self) -> None:
        """Stop work; temporary evidence persists for the lifetime of this workspace."""
        self.timer.stop()
        self._review_timer.stop()
        self._generation += 1
        self.process.close()
        self._worker.close()
