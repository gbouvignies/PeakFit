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
        for label in (self.summary, self.models, self.region, assumptions):
            label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
            label.setWordWrap(True)
            label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(label)
        members_title = QtWidgets.QLabel("Included peaks · rings on spectrum")
        layout.addWidget(members_title)
        self.members = QtWidgets.QListWidget()
        self.members.setMaximumHeight(110)
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
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
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
        layout.addWidget(dismiss)

    def show_preparation(self, prepared: PreparedLocalFit) -> None:
        """Project resolved parameters, preserving duplicate assignment identity."""
        group = prepared.cluster
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
            f"({prepared.contour_level / prepared.noise:g} × noise "
            f"{prepared.noise:.6g}). Independent of display contours."
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
