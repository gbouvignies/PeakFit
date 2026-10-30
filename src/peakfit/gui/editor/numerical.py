"""A useful evidence workspace when external spectra cannot be verified."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from peakfit.gui.background import ViewerComputation
from peakfit.gui.export_dialog import choose_and_export
from peakfit.gui.project import StoredRun, run_id
from peakfit.gui.project_process import project_operation


class NumericalReview(QtWidgets.QWidget):
    """Inspect retained fitted quantities without fabricating spectral observations."""

    keep_changed = QtCore.Signal(str, bool)

    def __init__(self, analysis):  # noqa: PLR0915
        super().__init__()
        self.analysis = analysis
        self._loaded_completion = None
        self._requested_id = None
        self._history_generation = 0
        self._worker = ViewerComputation(self)
        self._worker.completed.connect(self._samples_exported)
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel(
            "Numerical review only" if analysis.dataset is None else "Analysis history"
        )
        title.setObjectName("title")
        layout.addWidget(title)
        explanation = QtWidgets.QLabel(
            "The original spectrum could not be verified. Relink it to enable spectral views "
            "and new fits.\nRetained parameters, amplitudes, diagnostics and exact fitting "
            "samples remain available."
            if analysis.dataset is None
            else "Latest ten completed runs retain full evidence. Older runs are Summary only. "
            "Keep full evidence protects a run from automatic history pruning."
        )
        explanation.setWordWrap(True)
        explanation.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(explanation)
        actions = QtWidgets.QHBoxLayout()
        self.relink = QtWidgets.QPushButton("Relink source…")
        actions.addWidget(self.relink)
        details = QtWidgets.QToolButton()
        details.setText("Verification details")
        details.setCheckable(True)
        details.setVisible(bool(analysis.reason))
        actions.addWidget(details)
        actions.addStretch()
        if analysis.completion:
            status = QtWidgets.QLabel(
                ("Current" if analysis.current else "Outdated")
                + " numerical evidence · "
                + ("Source unverified" if analysis.dataset is None else "Source verified")
            )
            actions.addWidget(status)
        layout.addLayout(actions)
        failure = QtWidgets.QTextBrowser()
        failure.setPlainText(analysis.reason)
        failure.setFixedHeight(100)
        failure.hide()
        details.toggled.connect(failure.setVisible)
        layout.addWidget(failure)
        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs, 1)
        peaks = self.table(
            ["Assignment", "Stable ID", "Indirect / ppm", "Direct / ppm"],
            [(p.name, p.id, *p.positions) for p in analysis.peaks],
        )
        tabs.addTab(peaks, "Working peaks")
        self.run = QtWidgets.QComboBox()
        journal = {e["run_id"]: e for e in analysis.journal}
        for c in reversed(analysis.runs):
            identity = run_id(c)
            date = journal.get(identity, {}).get("completed_at", "")
            self.run.addItem(f"{date} · {identity[:8]} · Full", c)
        full_ids = {run_id(c) for c in analysis.runs}
        for entry in reversed(analysis.journal):
            if entry["run_id"] not in full_ids:
                self.run.addItem(
                    f"{entry.get('completed_at', '')} · {entry['run_id'][:8]} · Summary only", entry
                )
        layout.insertWidget(3, self.run)
        self.keep = QtWidgets.QCheckBox("Keep full evidence")
        self.keep.toggled.connect(self._keep)
        layout.insertWidget(4, self.keep)
        self.details = QtWidgets.QTextBrowser()
        tabs.addTab(self.details, "Run diagnostics and settings")
        self.parameters = self.table(
            ["Parameter", "Final value", "Unit", "Error", "Min", "Max"], []
        )
        tabs.addTab(self.parameters, "Fitted parameters")
        self.initial = self.table(
            ["Assignment", "Stable ID", "Initial indirect", "Initial direct"], []
        )
        tabs.addTab(self.initial, "Original starting peaks")
        self.starting_parameters = self.table(
            ["Parameter", "Start value", "Unit", "Min", "Max", "Vary"], []
        )
        tabs.addTab(self.starting_parameters, "Starting parameters")
        profiles = QtWidgets.QWidget()
        profile_layout = QtWidgets.QVBoxLayout(profiles)
        self.peak = QtWidgets.QComboBox()
        self.profile = pg.PlotWidget()
        self.profile.setLabel("bottom", "Original source plane index")
        self.profile.setLabel("left", "Signed fitted amplitude", units="a.u.")
        profile_layout.addWidget(self.peak)
        profile_layout.addWidget(self.profile)
        tabs.addTab(profiles, "Amplitude profiles")
        samples = QtWidgets.QWidget()
        sample_layout = QtWidgets.QVBoxLayout(samples)
        row = QtWidgets.QHBoxLayout()
        self.cluster = QtWidgets.QComboBox()
        self.plane = QtWidgets.QSpinBox()
        self.plane.setPrefix("Source plane ")
        self.sample_note = QtWidgets.QLabel()
        row.addWidget(self.cluster)
        row.addWidget(self.plane)
        row.addWidget(self.sample_note, 1)
        sample_layout.addLayout(row)
        self.samples = self.table(
            [
                "Indirect index",
                "Direct index",
                "Observation",
                "Terminal correction",
                "Cluster model",
                "Recorded residual",
            ],
            [],
        )
        sample_layout.addWidget(self.samples)
        self.sample_export = QtWidgets.QPushButton("Export exact sample evidence…")
        self.sample_export.clicked.connect(self.export_samples)
        sample_layout.addWidget(self.sample_export)
        self.cluster.currentIndexChanged.connect(self.show_samples)
        self.plane.valueChanged.connect(self.show_samples)
        tabs.addTab(samples, "Fitting samples")
        self.export = QtWidgets.QPushButton("Export retained run…")
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.export)
        bottom.addStretch()
        layout.addLayout(bottom)
        self.export.clicked.connect(self.export_run)
        self.run.currentIndexChanged.connect(self.show_run)
        self.peak.currentIndexChanged.connect(self.show_profile)
        self.show_run()

    @staticmethod
    def table(headers, rows):
        """Use native read-only tables with full precision and copyable cells."""
        table = QtWidgets.QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        NumericalReview.populate(table, rows)
        return table

    @staticmethod
    def populate(table, rows):
        """Replace a compact numerical table."""
        table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                cell = QtWidgets.QTableWidgetItem(str(value))
                cell.setToolTip(str(value))
                table.setItem(row, column, cell)
        table.resizeColumnsToContents()
        for column in range(table.columnCount()):
            table.setColumnWidth(column, min(table.columnWidth(column), 280))

    def show_run(self):
        """Show selected retained evidence; never infer unavailable quantities."""
        raw = self.run.currentData()
        identity = run_id(raw) if raw is not None and not isinstance(raw, dict) else None
        if identity != self._requested_id:
            self._history_generation += 1
            self._requested_id = identity
            self._loaded_completion = None
        completion = self.selected_run()
        if isinstance(raw, StoredRun) and completion is None:
            self.details.setPlainText("Loading and validating requested historical evidence…")
            for table in (self.parameters, self.initial, self.starting_parameters, self.samples):
                table.setRowCount(0)
            self.profile.clear()
            self.peak.clear()
            self.export.setEnabled(False)
            self.sample_export.setEnabled(False)
            self.keep.setEnabled(False)
            self._worker.submit(
                "history",
                self._history_generation,
                lambda: project_operation(
                    "history", raw, progress=lambda _: None, canceled=lambda: False
                ),
            )
            return
        self.export.setEnabled(completion is not None)
        self.cluster.clear()
        self.samples.setRowCount(0)
        self.sample_export.setEnabled(completion is not None and not isinstance(completion, dict))
        self.keep.setEnabled(completion is not None and not isinstance(completion, dict))
        with QtCore.QSignalBlocker(self.keep):
            self.keep.setChecked(
                completion is not None
                and not isinstance(completion, dict)
                and completion.result.run_id in self.analysis.protected
            )
        if isinstance(completion, dict):
            self.details.setPlainText(
                "Summary only · No sample evidence or amplitude profiles.\n"
                + json.dumps(completion, indent=2)
            )
            self.populate(
                self.parameters,
                [
                    (p["name"], p["value"], p["unit"], p["standard_error"], p["min"], p["max"])
                    for p in (completion.get("parameters") or [])
                ],
            )
            self.populate(
                self.initial,
                [
                    (p["name"], p["id"], *p["positions"])
                    for p in completion.get("initial_peaks", [])
                ],
            )
            self.populate(
                self.starting_parameters,
                [
                    (p["name"], p["value"], p["unit"], p["min"], p["max"], p["vary"])
                    for p in completion.get("initial_parameters", [])
                ],
            )
            self.peak.clear()
            self.profile.clear()
            return
        if completion is None:
            self.details.setPlainText(
                "No completed fit has been retained. Working peaks and settings are saved."
            )
            return
        r = completion.result
        self.populate(
            self.starting_parameters,
            [(p.name, p.value, p.unit, p.min, p.max, p.vary) for p in r.inputs.parameters.values()],
        )
        self.plane.setRange(1, len(r.inputs.snapshot.plane_values))
        for c in r.outcome.clusters:
            self.cluster.addItem(f"Cluster {c.cluster_id} · {c.classification.value}", c.cluster_id)
        self.show_samples()
        self.populate(
            self.parameters,
            [
                (p.name, p.value, p.unit, p.standard_error, p.min, p.max)
                for p in r.outcome.final_nonlinear_parameters
            ],
        )
        self.populate(self.initial, [(p.name, p.id, *p.positions) for p in r.inputs.snapshot.peaks])
        self.details.setPlainText(
            "\n".join(
                [
                    f"Run {r.run_id} · "
                    + (
                        ("Current" if self.analysis.current else "Outdated")
                        if completion is self.analysis.completion
                        else "Historical run"
                    )
                    + " · Source verification is independent of currentness.",
                    "Effective settings: " + r.inputs.snapshot.configuration,
                    *[
                        f"Cluster {c.cluster_id}: {c.classification.value}; "
                        f"{c.optimizer_provenance.termination_message}; "
                        f"{c.unusable_reason or ''}; "
                        + str(
                            c.analytical_evaluation.statistics
                            if c.analytical_evaluation
                            else "No analytical evaluation"
                        )
                        for c in r.outcome.clusters
                    ],
                    "Exact sampled observations, indices and terminal corrections "
                    "are retained and exportable.",
                ]
            )
        )
        self.peak.clear()
        names = {i.engine_name: i.assignment for i in r.inputs.identities}
        for c in r.outcome.clusters:
            if c.analytical_evaluation:
                for index, name in enumerate(c.peak_names):
                    self.peak.addItem(names[name], (c.cluster_id, index))
        self.show_profile()

    def show_profile(self):
        """Plot each retained plane as a distinct observation, with signed amplitudes."""
        self.profile.clear()
        item = self.peak.currentData()
        if item is None:
            return
        c = self.selected_run().result.outcome.cluster(item[0])
        self.profile.plot(c.analytical_evaluation.amplitudes[item[1]], symbol="o", pen=None)

    def export_run(self):
        """Reuse the canonical export workflow for retained numerical evidence."""
        completion = self.selected_run()
        if isinstance(completion, dict):
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export summary-only evidence", "run-summary.json", "JSON (*.json)"
            )
            if filename:
                Path(filename).write_text(json.dumps(completion, indent=2), encoding="utf-8")
        elif completion:
            choose_and_export(
                self,
                completion.result,
                completion=completion,
                current=self.analysis.current and completion is self.analysis.completion,
            )

    def _keep(self, keep):
        completion = self.selected_run()
        if completion is not None and not isinstance(completion, dict):
            protected = set(self.analysis.protected)
            protected.add(completion.result.run_id) if keep else protected.discard(
                completion.result.run_id
            )
            self.analysis = replace(self.analysis, protected=frozenset(protected))
            self.keep_changed.emit(completion.result.run_id, keep)

    def show_samples(self):
        """Inspect retained optimizer evidence; this is never a full-spectrum display."""
        completion = self.selected_run()
        cid = self.cluster.currentData()
        if cid is None or completion is None or isinstance(completion, dict):
            return
        result = completion.result
        s = next(s for s in result.completed_fit.samples if s.cluster_id == cid)
        e = result.outcome.cluster(cid).analytical_evaluation
        plane = self.plane.value() - 1
        count = min(2000, s.data.shape[0])
        self.sample_note.setText(
            f"{count} of {s.data.shape[0]} exact fitting samples · no interpolation"
        )
        self.populate(
            self.samples,
            [
                (
                    s.grid_indices[0][i],
                    s.grid_indices[1][i],
                    s.data[i, plane],
                    s.neighbor_corrections[i, plane],
                    e.model_values[i, plane] if e else "Unavailable",
                    e.raw_residuals[i, plane] if e else "Unavailable",
                )
                for i in range(count)
            ],
        )

    def export_samples(self):
        """Export all retained observations/corrections and analytical sample arrays losslessly."""
        completion = self.selected_run()
        if completion is None or isinstance(completion, dict):
            return
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export exact sample evidence", "sample-evidence.npz", "NPZ (*.npz)"
        )
        if not filename:
            return
        arrays = {}
        for s in completion.result.completed_fit.samples:
            prefix = f"c{s.cluster_id}_"
            arrays[prefix + "data"] = s.data
            arrays[prefix + "corrections"] = s.neighbor_corrections
            arrays.update(
                {prefix + f"index{dim}": values for dim, values in enumerate(s.grid_indices)}
            )
            e = completion.result.outcome.cluster(s.cluster_id).analytical_evaluation
            if e:
                arrays[prefix + "model"] = e.model_values
                arrays[prefix + "residual"] = e.raw_residuals
        self.sample_export.setEnabled(False)
        self.sample_note.setText("Exporting exact sample evidence…")
        self._worker.submit("export", 0, lambda: np.savez_compressed(filename, **arrays))

    def selected_run(self):
        """Return only the active or explicitly loaded historical record."""
        selected = self.run.currentData()
        if isinstance(selected, StoredRun):
            active = self.analysis.completion
            if active and selected.run_id == active.result.run_id:
                return active
            return self._loaded_completion
        return selected

    def _samples_exported(self, _kind, _revision, outcome):
        if _kind == "history":
            if _revision != self._history_generation:
                return
            if isinstance(outcome, Exception):
                self.details.setPlainText(f"Historical evidence unavailable: {outcome}")
            else:
                self._loaded_completion = outcome
                self.show_run()
            return
        self.sample_export.setEnabled(True)
        self.sample_note.setText(
            f"Export failed: {outcome}"
            if isinstance(outcome, Exception)
            else "Exact sample evidence exported"
        )

    def release(self):
        """Release the selected historical record and invalidate late worker deliveries."""
        self._loaded_completion = None
        self._worker.close()

    def closeEvent(self, event):  # noqa: N802
        """Closing a historical inspector must not retain a decoded run."""
        self.release()
        super().closeEvent(event)
