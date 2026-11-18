"""A useful evidence workspace when external spectra cannot be verified."""

import json
import math
from dataclasses import dataclass, replace

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from peakfit.engine.domain.spectrum import get_dimension_label
from peakfit.engine.scientific.completed import FinalClusterSamples
from peakfit.gui import formatting
from peakfit.gui import theme as themes
from peakfit.gui.background import ViewerComputation
from peakfit.gui.export_dialog import choose_and_export
from peakfit.gui.project import StoredRun, run_id
from peakfit.gui.project_results import completed, reconstruct_result, run_numbers


def axis_names(analysis) -> tuple[str, str]:
    """(vertical, horizontal) axis names in peak-position order; F-numbers without a source."""
    if analysis.dataset is not None:
        axes = analysis.dataset.spectra.spectral_params
        return formatting.axis_name(axes[0]), formatting.axis_name(axes[1])
    return get_dimension_label(0), get_dimension_label(1)


def cell_text(value) -> str:
    """A table cell: text as it is, numbers through the shared formatter."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return formatting.grouped(value)
    if isinstance(value, float):
        return formatting.scaled(value)
    return str(value)


def _float(value) -> float:
    """A recorded number, or NaN where a summary recorded none."""
    return math.nan if value is None else float(value)


def peak_rows(peaks) -> list[tuple]:
    """Name, ID and the two positions in ppm, at ppm's fixed decimals.

    ``peaks`` yields (name, ID, positions).
    """
    return [
        (name, identity, *(Exact(formatting.ppm(_float(x)), _float(x)) for x in positions))
        for name, identity, positions in peaks
    ]


def parameter_rows(parameters) -> list[tuple]:
    """Name, value ± σ (one power of ten; "not assessed" without σ), unit and bounds.

    ``parameters`` yields (name, value, unit, σ, lower, upper).
    """
    rows = []
    for record in parameters:
        name, unit = record[0], record[2]
        value, sigma, low, high = (_float(record[i]) for i in (1, 3, 4, 5))
        shown = formatting.fitted(value, sigma, unit)
        rows.append(
            (
                name,
                Exact(shown.value_text, value),
                unit,
                Exact(shown.sigma_text, sigma),
                Exact(formatting.quantity(low, unit), low),
                Exact(formatting.quantity(high, unit), high),
            )
        )
    return rows


def start_rows(parameters) -> list[tuple]:
    """Name, start, unit, bounds and varied, at each unit's fixed decimals."""
    return [
        (
            name,
            *(Exact(formatting.quantity(_float(v), unit), _float(v)) for v in (value,)),
            unit,
            *(Exact(formatting.quantity(_float(b), unit), _float(b)) for b in (low, high)),
            vary,
        )
        for name, value, unit, low, high, vary in parameters
    ]


def statistics_text(statistics) -> str:
    """A cluster's recorded statistics as label = value, χ²ᵣ at its fixed decimals."""
    return (
        f"χ²ᵣ {formatting.chi2(statistics.reduced_chi_squared)} "
        f"(χ² {formatting.scaled(statistics.chi_squared)}, "
        f"{formatting.count(statistics.n_observations, 'observation')}, "
        f"{formatting.count(statistics.n_fitted_parameters, 'fitted parameter')}, "
        f"{formatting.grouped(statistics.degrees_of_freedom)} degrees of freedom)"
    )


@dataclass(frozen=True, slots=True)
class Exact:
    """A cell's text as shown and the exact value its tooltip gives."""

    text: str
    value: float


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
        self._worker.completed.connect(self._history_loaded)
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel(
            "Numerical review only" if analysis.dataset is None else "Analysis history"
        )
        title.setObjectName("title")
        layout.addWidget(title)
        explanation = QtWidgets.QLabel(
            "The original spectrum could not be verified. Relink it to enable spectral views "
            "and new fits.\nRetained parameters, amplitudes, uncertainties, diagnostics and "
            "terminal corrections remain available. Experimental samples require the source."
            if analysis.dataset is None
            else "Latest ten completed runs retain fitted results and reconstruction evidence. "
            "Older runs are Summary only. "
            "Keep full evidence protects a run from automatic history pruning."
        )
        explanation.setWordWrap(True)
        explanation.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(explanation)
        actions = QtWidgets.QHBoxLayout()
        self.relink = themes.command("Relink source…")
        actions.addWidget(self.relink)
        details = QtWidgets.QToolButton()
        details.setText("Verification details")
        details.setCheckable(True)
        details.setVisible(bool(analysis.reason))
        actions.addWidget(details)
        actions.addStretch()
        numbers = run_numbers(analysis.journal)
        if analysis.completion:
            active = numbers.get(run_id(analysis.completion))
            status = QtWidgets.QLabel(
                (f"Active run {active} · " if active else "Active run · ")
                + ("Current" if analysis.current else "Outdated")
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
        vertical, horizontal = axis_names(analysis)
        peaks = self.table(
            ["Assignment", "Stable ID", f"{vertical} (ppm)", f"{horizontal} (ppm)"],
            peak_rows((p.name, p.id, p.positions) for p in analysis.peaks),
        )
        tabs.addTab(peaks, "Working peaks")
        self.run = QtWidgets.QComboBox()
        journal = {e["run_id"]: e for e in analysis.journal}

        # Named as on the Run page: "Run 3 · <time> · <short ID> · Full".
        def named(identity: str) -> str:
            return f"Run {numbers[identity]} · " if identity in numbers else ""

        for c in reversed(analysis.runs):
            identity = run_id(c)
            date = formatting.local_time(journal.get(identity, {}).get("completed_at", ""))
            self.run.addItem(f"{named(identity)}{date} · {identity[:8]} · Full", c)
        full_ids = {run_id(c) for c in analysis.runs}
        for entry in reversed(analysis.journal):
            identity = entry["run_id"]
            if identity not in full_ids:
                status = str(entry.get("status", ""))
                kind = "Summary only" if completed(entry) else f"{status.capitalize()} attempt"
                self.run.addItem(
                    f"{named(identity)}{formatting.local_time(entry.get('completed_at', ''))} · "
                    f"{identity[:8]} · {kind}",
                    entry,
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
            [
                "Assignment",
                "Stable ID",
                f"Initial {vertical} (ppm)",
                f"Initial {horizontal} (ppm)",
            ],
            [],
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
        # The unit in the text, not pyqtgraph's units: no SI prefix (Ga.u.) on the axis.
        self.profile.setLabel("left", "Signed fitted amplitude (a.u.)")
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
        self._sample_headers = [
            f"{vertical} index",
            f"{horizontal} index",
            "Observation",
            "Terminal correction",
            "Cluster model",
            "Reconstructed residual",
        ]
        self.samples = self.table(self._sample_headers, [])
        sample_layout.addWidget(self.samples)
        self.cluster.currentIndexChanged.connect(self.show_samples)
        self.plane.valueChanged.connect(self.show_samples)
        tabs.addTab(samples, "Fitting samples")
        self.export = themes.command("Export retained run…")
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
        """Native read-only tables with copyable cells; full precision in each tooltip."""
        table = QtWidgets.QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        NumericalReview.populate(table, rows)
        return table

    @staticmethod
    def populate(table, rows, headers=None):
        """Replace a compact numerical table; full precision in each number's tooltip.

        A float column of one order of magnitude shows its × 10ⁿ once, in the header
        (``headers`` gives the plain titles).
        """
        columns = list(zip(*rows, strict=True)) if rows else []
        texts: dict[int, list[str]] = {}
        for index, values in enumerate(columns):
            if all(isinstance(v, float) for v in values):
                shared = formatting.column([float(v) for v in values])
                texts[index] = [cell.value_text for cell in shared.cells]
                if headers is not None:
                    item = table.horizontalHeaderItem(index)
                    if item is not None:
                        item.setText(shared.header(headers[index]))
        table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                if isinstance(value, Exact):
                    text, tip = value.text, formatting.exact(value.value)
                elif column in texts:
                    text, tip = texts[column][row], formatting.exact(value)
                else:
                    text, tip = cell_text(value), str(value)
                cell = QtWidgets.QTableWidgetItem(text)
                cell.setToolTip(tip)
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
            self.keep.setEnabled(False)
            # Historical runs load on this background worker, as fitting's parent runs do.
            self._worker.submit("history", self._history_generation, raw.load)
            return
        if (
            completion is not None
            and not isinstance(completion, dict)
            and self.analysis.dataset is not None
            and any(
                not isinstance(s, FinalClusterSamples)
                for s in completion.result.completed_fit.samples
            )
        ):
            self.export.setEnabled(False)
            self.keep.setEnabled(False)
            self.details.setPlainText("Reconstructing requested samples from verified spectrum…")
            self._worker.submit(
                "reconstruct",
                self._history_generation,
                lambda: replace(
                    completion, result=reconstruct_result(completion.result, self.analysis.dataset)
                ),
            )
            return
        self.export.setEnabled(completion is not None and not isinstance(completion, dict))
        self.export.setToolTip("Full scientific export requires retained full evidence.")
        self.cluster.clear()
        self.samples.setRowCount(0)
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
                parameter_rows(
                    (p["name"], p["value"], p["unit"], p["standard_error"], p["min"], p["max"])
                    for p in (completion.get("parameters") or [])
                ),
            )
            self.populate(
                self.initial,
                peak_rows(
                    (p["name"], p["id"], p["positions"])
                    for p in completion.get("initial_peaks", [])
                ),
            )
            self.populate(
                self.starting_parameters,
                start_rows(
                    (p["name"], p["value"], p["unit"], p["min"], p["max"], p["vary"])
                    for p in completion.get("initial_parameters", [])
                ),
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
            start_rows(
                (p.name, p.value, p.unit, p.min, p.max, p.vary)
                for p in r.inputs.parameters.values()
            ),
        )
        self.plane.setRange(1, len(r.inputs.snapshot.plane_values))
        for c in r.outcome.clusters:
            self.cluster.addItem(f"Cluster {c.cluster_id} · {c.classification.value}", c.cluster_id)
        self.show_samples()
        self.populate(
            self.parameters,
            parameter_rows(
                (p.name, p.value, p.unit, p.standard_error, p.min, p.max)
                for p in r.outcome.final_nonlinear_parameters
            ),
        )
        self.populate(
            self.initial, peak_rows((p.name, p.id, p.positions) for p in r.inputs.snapshot.peaks)
        )
        self.details.setPlainText(
            "\n".join(
                [
                    f"Run {r.run_id} · "
                    + (
                        ("Current" if self.analysis.current else "Outdated")
                        if self.analysis.completion
                        and completion.result.run_id == self.analysis.completion.result.run_id
                        else "Historical run"
                    )
                    + " · Source verification is independent of currentness.",
                    # What the engine fitted: the settings with each peak's own limits.
                    "Effective settings: " + r.inputs.engine_configuration.model_dump_json(),
                    *[
                        f"Cluster {c.cluster_id}: {c.classification.value}; "
                        f"{c.optimizer_provenance.termination_message}; "
                        f"{c.unusable_reason or ''}; "
                        + (
                            statistics_text(c.analytical_evaluation.statistics)
                            if c.analytical_evaluation
                            else "No analytical evaluation"
                        )
                        for c in r.outcome.clusters
                    ],
                    "Sample coordinates and exact terminal corrections are retained. "
                    "Observed/model/residual values require the verified external spectrum.",
                ]
            )
        )
        self.peak.clear()
        by_engine = r.index.by_engine
        for c in r.outcome.clusters:
            if c.analytical_evaluation:
                for index, name in enumerate(c.peak_names):
                    self.peak.addItem(by_engine[name].assignment, (c.cluster_id, index))
        self.show_profile()

    def show_profile(self):
        """Plot each retained plane as a distinct observation, with signed amplitudes."""
        self.profile.clear()
        item = self.peak.currentData()
        if item is None:
            return
        c = self.selected_run().result.outcome.cluster(item[0])
        self.profile.plot(
            self.selected_run().result.inputs.snapshot.source_plane_indices,
            c.analytical_evaluation.amplitudes[item[1]],
            symbol="o",
            pen=None,
        )

    def export_run(self):
        """Reuse the canonical export workflow for retained numerical evidence."""
        completion = self.selected_run()
        if completion is not None and not isinstance(completion, dict):
            choose_and_export(
                self,
                completion.result,
                completion=completion,
                current=self.analysis.current
                and completion.result.run_id == self.analysis.completion.result.run_id,
                dataset=self.analysis.dataset,
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
        e = result.outcome.cluster(cid).sample_evaluation
        plane = self.plane.value() - 1
        available = isinstance(s, FinalClusterSamples)
        self.cluster.setEnabled(available)
        self.plane.setEnabled(available)
        if not available:
            self.samples.setRowCount(0)
            self.sample_note.setText("Relink the verified source to inspect fitting samples")
            return
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
            self._sample_headers,
        )

    def selected_run(self):
        """Return only the active or explicitly loaded historical record."""
        selected = self.run.currentData()
        if self._loaded_completion is not None:
            return self._loaded_completion
        if isinstance(selected, StoredRun):
            active = self.analysis.completion
            if active and selected.run_id == active.result.run_id:
                return active
            return self._loaded_completion
        return selected

    def _history_loaded(self, _kind, revision, outcome):
        if revision != self._history_generation:
            return
        if isinstance(outcome, Exception):
            self.details.setPlainText(f"Historical evidence unavailable: {outcome}")
        else:
            self._loaded_completion = outcome
            self.show_run()

    def release(self):
        """Release the selected historical record and invalidate late worker deliveries."""
        self._loaded_completion = None
        self._worker.close()

    def closeEvent(self, event):  # noqa: N802
        """Closing a historical inspector must not retain a decoded run."""
        self.release()
        super().closeEvent(event)
