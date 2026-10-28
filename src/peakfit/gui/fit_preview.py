"""Read-only fitting scope in the viewer's contextual right-hand workspace."""

from typing import TYPE_CHECKING

from PySide6 import QtCore, QtWidgets

if TYPE_CHECKING:
    from peakfit.gui.fit_process import FitProcessResult
    from peakfit.gui.fitting import PreparedLocalFit


class FitPreview(QtWidgets.QWidget):
    """Explain the exact local scope and starting model without running a fit."""

    dismissed = QtCore.Signal()
    run_requested = QtCore.Signal()
    cancel_requested = QtCore.Signal()
    prepare_requested = QtCore.Signal()
    result_requested = QtCore.Signal()

    def __init__(self) -> None:  # noqa: PLR0915
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        title = self.title = QtWidgets.QLabel("Local fitting · scope preview")
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)
        self.summary = QtWidgets.QLabel()
        self.models = QtWidgets.QLabel()
        self.region = QtWidgets.QLabel()
        assumptions = self.assumptions = QtWidgets.QLabel(
            "All planes share positions and lineshape parameters. Each peak's signed "
            "amplitude is solved independently in each plane.\n\n"
            "Isolated local fit: signals from other groups are not subtracted. "
            "The complete iterative cross-cluster correction procedure is not included. "
            "Check nearby peaks and the experimental model before fitting."
        )
        for label, lines in (
            (self.summary, 2),
            (self.models, 2),
            (self.region, 4),
            (assumptions, None),
        ):
            label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
            label.setWordWrap(True)
            label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            if lines is not None:
                label.setFixedHeight(label.fontMetrics().lineSpacing() * lines + 4)
                label.setSizePolicy(
                    QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed
                )
            layout.addWidget(label)
        self.status = QtWidgets.QLabel()
        self.status.setFixedHeight(self.status.fontMetrics().lineSpacing() + 4)
        self.status.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self.status)
        self._details = ""
        members_title = QtWidgets.QLabel("Included peaks · rings on spectrum")
        layout.addWidget(members_title)
        self.members = QtWidgets.QListWidget()
        self.members.setFixedHeight(110)
        layout.addWidget(self.members)
        self.table_title = QtWidgets.QLabel("Effective starting parameters · read-only")
        layout.addWidget(self.table_title)
        self.result_mode = QtWidgets.QComboBox()
        self.result_mode.addItems(
            ["Fitted nonlinear parameters", "Signed amplitudes · displayed plane"]
        )
        self.result_mode.hide()
        self.result_mode.currentIndexChanged.connect(self._result_table)
        layout.addWidget(self.result_mode)
        self._result: FitProcessResult | None = None
        self._plane = 0
        self.parameters = QtWidgets.QTableWidget(0, 6)
        self.parameters.setHorizontalHeaderLabels(
            ["Peak / axis", "Parameter", "Start", "Min", "Max", "Fit"]
        )
        self.parameters.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.parameters.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.parameters.verticalHeader().hide()
        self.parameters.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Interactive
        )
        layout.addWidget(self.parameters, 1)
        note = self.note = QtWidgets.QLabel(
            "Positions use unfolded ppm. Linewidths are in Hz (unapodized FWHM for "
            "processing-aware models). Amplitudes have no starting controls.\n"
            "Preparation only — no fit has been run."
        )
        note.setWordWrap(True)
        note.setFixedHeight(note.fontMetrics().lineSpacing() * 3 + 4)
        layout.addWidget(note)
        actions = QtWidgets.QHBoxLayout()
        self.run = QtWidgets.QPushButton("Run local fit")
        self.run.setEnabled(False)
        self.run.clicked.connect(self.run_requested)
        self.cancel = QtWidgets.QPushButton("Cancel fit")
        self.cancel.hide()
        self.cancel.clicked.connect(self.cancel_requested)
        self.prepare = QtWidgets.QPushButton("Prepare selected group")
        self.prepare.hide()
        self.prepare.clicked.connect(self.prepare_requested)
        self.retained = QtWidgets.QPushButton("Show retained result")
        self.retained.hide()
        self.retained.clicked.connect(self.result_requested)
        for button in (self.run, self.cancel, self.prepare, self.retained):
            actions.addWidget(button)
        layout.addLayout(actions)
        dismiss = QtWidgets.QPushButton("Back to Inspect")
        dismiss.clicked.connect(self.dismissed)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(dismiss, 1)
        self.details = QtWidgets.QPushButton("Details…")
        self.details.setFixedWidth(85)
        self.details.setEnabled(False)
        self.details.clicked.connect(self._show_details)
        buttons.addWidget(self.details)
        layout.addLayout(buttons)

    def show_preparation(self, prepared: PreparedLocalFit) -> None:
        """Project resolved parameters, preserving duplicate assignment identity."""
        from peakfit.gui.local_fit import execution_unavailable  # noqa: PLC0415

        self._result = None
        self.assumptions.show()
        self.members.setFixedHeight(110)
        self.title.setText("Local fitting · reviewed scope")
        self.table_title.setText("Effective starting parameters · read-only")
        self.result_mode.hide()
        self.run.show()
        self.cancel.hide()
        self.prepare.hide()
        reason = execution_unavailable(prepared)
        self.run.setEnabled(reason is None)
        self.run.setToolTip(reason or "Execute these exact inputs in a separate process")
        self.parameters.setColumnCount(6)
        self.parameters.setHorizontalHeaderLabels(
            ["Peak / axis", "Parameter", "Start", "Min", "Max", "Fit"]
        )
        self.note.setText(
            "Numerical fitting unavailable · Details explains why and how to proceed."
            if reason
            else (
                "Positions are unfolded ppm; linewidths are Hz. Signed amplitudes are solved "
                "independently for every plane. No fit has been run."
            )
        )
        group = prepared.scope
        self.summary.setText(
            f"{len(group.peaks)} peaks · {group.n_series} experimental planes · "
            f"{group.n_points} sampled spectral points per plane"
        )
        axes = group.peaks[0].shapes
        self.models.setText(
            "Models: "
            + " · ".join(
                f"{shape.axis}: {name}"
                for shape, name in zip(axes, prepared.shape_names, strict=True)
            )
            + "\nFrom NMRPipe processing metadata; verify suitability for the experiment."
        )
        self.region.setText(
            f"Amber cells: exact fitting samples across all planes.\n"
            f"Reviewed fitting threshold: {prepared.contour_level:.6g} "
            "(contour intensity at preparation).\n"
            f"Statistical noise: {prepared.noise:.6g}, estimated independently across all planes."
        )
        self.members.clear()
        labels = {}
        for number, identity in enumerate(prepared.identities, start=1):
            labels[identity.engine_name] = f"{number}. {identity.assignment}"
            peak = next(p for p in prepared.snapshot.peaks if p.id == identity.working_id)
            self.members.addItem(
                f"{labels[identity.engine_name]}  ({peak.positions[0]:.6f}, "
                f"{peak.positions[1]:.6f} ppm)"
            )
        self.parameters.setRowCount(0)
        self.parameters.setEnabled(prepared.parameters is not None)
        self._details = (
            (reason or "Numerical inputs prepared; no fit has been run.")
            + "\n\n"
            + self.models.text()
            + "\n\n"
            + self.region.text()
        )
        self.status.setText(
            "Scope shown · numerical preparation unavailable"
            if prepared.limitation
            else "Scope ready · numerical inputs prepared"
        )
        self.status.setToolTip(self._details)
        self.details.setEnabled(True)
        if prepared.parameters is None:
            return
        self.parameters.setRowCount(len(prepared.parameters))
        for row, parameter in enumerate(prepared.parameters.values()):
            identifier = parameter.param_id
            assert identifier is not None
            owner = labels.get(identifier.peak_name, "Group")
            label = {
                "cs": "Shift",
                "lw": "Linewidth",
                "eta": "Mixing",
                "j": "Coupling",
                "phase": "Phase",
            }.get(identifier.label, identifier.label)
            values = (
                f"{owner} / {identifier.axis}",
                f"{label} ({parameter.unit})" if parameter.unit else label,
                f"{parameter.value:.9g}",
                f"{parameter.min:.9g}",
                f"{parameter.max:.9g}",
                "Vary" if parameter.vary else "Fixed",
            )
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setToolTip(f"{parameter.name}\n{value}")
                self.parameters.setItem(row, column, item)
        self.parameters.resizeColumnsToContents()

    def show_status(self, message: str, *, clear: bool = False) -> None:
        """Use reserved status space; retain the previous scope while updating."""
        self._details = message
        self.status.setText("Scope unavailable · Details…" if clear else message)
        self.status.setToolTip(message)
        self.details.setEnabled(True)
        self.parameters.setEnabled(False)
        self.run.setEnabled(False)
        if clear:
            self.summary.setText("Scope unavailable")
            self.models.clear()
            self.region.clear()
            self.members.clear()
            self.parameters.setRowCount(0)

    def show_running(self, message: str) -> None:
        """Keep the submitted scope readable and provide real cancellation."""
        self.status.setText(message)
        self.run.hide()
        self.cancel.show()
        self.prepare.hide()
        self.note.setText(
            "Scientific edits and dataset replacement pause during fitting. Navigation and "
            "contour controls remain available. Submitted fitting scope is fixed."
        )

    def show_result(self, result: FitProcessResult, plane: int, *, outdated: bool) -> None:
        """Inspect retained scientific quantities without modifying working peaks."""
        self.show_preparation(result.scientific.inputs)
        self._result = result
        self.assumptions.hide()
        self.members.setFixedHeight(70)
        self._plane = plane
        outcome = result.scientific.outcome.clusters[0]
        provenance = outcome.optimizer_provenance
        label = {
            "converged": "Converged · numerically usable",
            "usable_non_converged": "Not converged · usable provisional result",
            "unusable": "Unusable result",
        }[outcome.classification.value]
        self.title.setText("Local fitting · numerical result")
        self.status.setText("Outdated · working inputs changed" if outdated else label)
        self.region.setText(
            f"{label}\n{provenance.termination_message or outcome.unusable_reason or ''}\n"
            f"Function evaluations: {provenance.function_evaluations} · "
            f"Jacobian evaluations: {provenance.jacobian_evaluations}"
        )
        evaluation = outcome.analytical_evaluation
        if evaluation is not None:
            stats = evaluation.statistics
            quality = (
                f"χ²: {stats.chi_squared:.6g} · reduced χ²: {stats.reduced_chi_squared:.6g}\n"
                f"Observations: {stats.n_observations:,} · fitted parameters: "
                f"{stats.n_fitted_parameters} · DOF: {stats.degrees_of_freedom}"
            )
        else:
            quality = outcome.unusable_reason or "No usable numerical quantities."
        self.models.setText(quality)
        self.table_title.setText("Read-only result · original identities retained")
        self.result_mode.show()
        self.run.hide()
        self.cancel.hide()
        self.prepare.show()
        self.parameters.setEnabled(True)
        self.note.setText(
            "Fitting has not changed working peaks. Result is retained in memory only; "
            "it is NOT saved to disk. Statistical noise and fitted mask remain those reviewed."
        )
        memory = result.scientific.peak_memory_bytes
        memory_label = f"{memory / 1024**2:.1f} MiB" if memory is not None else "unavailable"
        self._details += (
            f"\n\n{quality}\n{provenance.termination_message}\n"
            f"Optimality: {provenance.optimality}\n"
            f"Optimizer settings: {result.scientific.optimizer_config}\n"
            f"Numerical execution: {result.scientific.execution_seconds:.3f} s\n"
            f"Process startup: {result.startup_seconds:.3f} s\n"
            f"Input serialization/copy: {result.serialization_seconds:.3f} s "
            f"({result.input_bytes:,} bytes)\n"
            f"Wall time: {result.wall_seconds:.3f} s\n"
            f"Child peak RSS: {memory_label}\n"
            f"PeakFit: {result.scientific.software_version}\n"
            f"Run: {result.scientific.run_id}\nCompleted: {result.scientific.completed_at}\n"
            + "\n".join(result.scientific.warnings)
        )
        self.status.setToolTip(self._details)
        self._result_table()

    def set_result_plane(self, plane: int) -> None:
        """Amplitudes follow the main viewer's plane, without recomputing the fit."""
        self._plane = plane
        if self._result is not None and self.result_mode.currentIndex() == 1:
            self._result_table()

    def _result_table(self) -> None:
        if self._result is None:
            return
        scientific = self._result.scientific
        inputs = scientific.inputs
        outcome = scientific.outcome.clusters[0]
        labels = {
            identity.engine_name: f"{number}. {identity.assignment}"
            for number, identity in enumerate(inputs.identities, 1)
        }
        rows = []
        if self.result_mode.currentIndex() == 0:
            headers = ["Peak / axis", "Parameter", "Start", "Fitted", "Min", "Max", "Fit"]
            assert inputs.parameters is not None
            for parameter in outcome.final_nonlinear_parameters:
                start = inputs.parameters[parameter.name]
                identifier = start.param_id
                assert identifier is not None
                rows.append(
                    (
                        f"{labels.get(identifier.peak_name, 'Group')} / {identifier.axis}",
                        f"{identifier.label} ({parameter.unit})",
                        f"{start.value:.9g}",
                        f"{parameter.value:.9g}",
                        f"{parameter.min:.9g}",
                        f"{parameter.max:.9g}",
                        "Vary" if parameter.vary else "Fixed",
                    )
                )
        else:
            headers = ["Peak", "Plane", "Plane value", "Signed amplitude"]
            evaluation = outcome.analytical_evaluation
            if evaluation is not None:
                for index, name in enumerate(outcome.peak_names):
                    rows.append(
                        (
                            labels[name],
                            str(self._plane + 1),
                            f"{inputs.snapshot.plane_values[self._plane]:.9g}",
                            f"{evaluation.amplitudes[index, self._plane]:.9g}",
                        )
                    )
        self.parameters.setRowCount(0)
        self.parameters.setColumnCount(len(headers))
        self.parameters.setHorizontalHeaderLabels(headers)
        self.parameters.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                self.parameters.setItem(row, column, QtWidgets.QTableWidgetItem(value))
        self.parameters.resizeColumnsToContents()

    def _show_details(self) -> None:
        dialog = QtWidgets.QMessageBox(self)
        dialog.setWindowTitle("Fitting scope details")
        dialog.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        dialog.setText(self._details)
        dialog.exec()
