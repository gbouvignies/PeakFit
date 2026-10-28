"""Read-only fitting scope in the viewer's contextual right-hand workspace."""

from typing import TYPE_CHECKING

from PySide6 import QtCore, QtWidgets

if TYPE_CHECKING:
    from peakfit.gui.fitting import PreparedLocalFit


class FitPreview(QtWidgets.QWidget):
    """Explain the exact local scope and starting model without running a fit."""

    dismissed = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        title = QtWidgets.QLabel("Local fitting · scope preview")
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)
        self.summary = QtWidgets.QLabel()
        self.models = QtWidgets.QLabel()
        self.region = QtWidgets.QLabel()
        assumptions = QtWidgets.QLabel(
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
        layout.addWidget(QtWidgets.QLabel("Effective starting parameters · read-only"))
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
        note = QtWidgets.QLabel(
            "Positions use unfolded ppm. Linewidths are in Hz (unapodized FWHM for "
            "processing-aware models). Amplitudes have no starting controls.\n"
            "Preparation only — no fit has been run."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
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
            f"Clustering threshold: {prepared.contour_level:.6g} "
            "(lowest displayed contour intensity).\n"
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
            (prepared.limitation or "Numerical inputs prepared; no fit has been run.")
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
        if clear:
            self.summary.setText("Scope unavailable")
            self.models.clear()
            self.region.clear()
            self.members.clear()
            self.parameters.setRowCount(0)

    def _show_details(self) -> None:
        dialog = QtWidgets.QMessageBox(self)
        dialog.setWindowTitle("Fitting scope details")
        dialog.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        dialog.setText(self._details)
        dialog.exec()
