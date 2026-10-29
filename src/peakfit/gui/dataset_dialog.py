"""Review effective whole-dataset settings before submitting a scientific run."""

from dataclasses import replace
from typing import TYPE_CHECKING, cast, get_args

from PySide6 import QtCore, QtWidgets

from peakfit.engine.domain.config import LineshapeName
from peakfit.gui.background import ViewerComputation
from peakfit.gui.fitting import PreparedDatasetFit, prepare_dataset_fit

if TYPE_CHECKING:
    from peakfit.engine.domain.config import PeakFitConfig
    from peakfit.gui.viewer import SpectrumViewer


class DatasetFitDialog(QtWidgets.QDialog):
    """Prepare off Qt and show exactly the problem the Run action will submit."""

    def __init__(self, viewer: SpectrumViewer, configuration: PeakFitConfig) -> None:  # noqa: PLR0915
        super().__init__(viewer)
        self.setWindowTitle("Fit complete dataset")
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.resize(660, 400)
        self.configuration = configuration.model_copy(deep=True)
        self.prepared: PreparedDatasetFit | None = None
        self._revision = 0
        self._spectra = viewer.spectra.model_copy(
            update={
                "params": [axis.model_copy(deep=True) for axis in viewer.spectra.params],
                "z_values": viewer.spectra.z_values.copy(),
            }
        )
        self._peaks = viewer.peaks
        self._source_id = id(viewer.spectra)
        self._source_files = viewer.source_files
        self._worker = ViewerComputation(self)
        self._worker.completed.connect(self._prepared)
        self.finished.connect(self._worker.close)
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._prepare)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)
        heading = QtWidgets.QLabel("Fit the complete dataset")
        font = heading.font()
        font.setPointSize(font.pointSize() + 2)
        font.setBold(True)
        heading.setFont(font)
        layout.addWidget(heading)
        layout.addWidget(
            QtWidgets.QLabel(
                f"{len(self._peaks)} working peaks · "
                f"{viewer.spectra.data.shape[0]} experimental planes"
            )
        )
        form = QtWidgets.QFormLayout()
        self.passes = QtWidgets.QSpinBox()
        self.passes.setRange(1, 20)
        self.passes.setValue(configuration.fitting.refine_iterations)
        self.passes.setAccessibleName("Optimizer passes")
        self.threshold = QtWidgets.QDoubleSpinBox()
        self.threshold.setRange(0.1, 1000)
        self.threshold.setDecimals(2)
        self.threshold.setValue(configuration.clustering.contour_factor)
        self.threshold.setSuffix(" × statistical noise")
        self.threshold.setAccessibleName("Clustering threshold")
        self.threshold.setKeyboardTracking(False)
        form.addRow("Optimizer passes", self.passes)
        form.addRow("Clustering threshold", self.threshold)
        layout.addLayout(form)
        explanation = QtWidgets.QLabel(
            "Neighbor contributions are updated between passes. The clustering threshold "
            "uses all-plane statistical noise and is independent of display contours."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.status = QtWidgets.QLabel("Preparing scientific inputs…")
        self.status.setWordWrap(True)
        self.status.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self.advanced_toggle = QtWidgets.QToolButton()
        self.advanced_toggle.setText("Scientific settings and starting parameters")
        self.advanced_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setAutoRaise(True)
        layout.addWidget(self.advanced_toggle)
        self.advanced = QtWidgets.QWidget()
        advanced = QtWidgets.QVBoxLayout(self.advanced)
        advanced.setContentsMargins(0, 0, 0, 0)
        scientific = QtWidgets.QFormLayout()
        self.lineshape = QtWidgets.QComboBox()
        self.lineshape.addItems(list(get_args(LineshapeName)))
        self.lineshape.setCurrentText(configuration.fitting.lineshape)
        self.lineshape.setAccessibleName("Lineshape model")
        self.max_iterations = QtWidgets.QSpinBox()
        self.max_iterations.setRange(1, 100000)
        self.max_iterations.setValue(configuration.fitting.max_iterations)
        self.tolerance = QtWidgets.QDoubleSpinBox()
        self.tolerance.setDecimals(12)
        self.tolerance.setRange(1e-12, 0.1)
        self.tolerance.setValue(configuration.fitting.tolerance)
        self.tolerance.setKeyboardTracking(False)
        for name, control in (
            ("Lineshape model", self.lineshape),
            ("Maximum optimizer iterations", self.max_iterations),
            ("Optimizer tolerance", self.tolerance),
        ):
            scientific.addRow(name, control)
        advanced.addLayout(scientific)
        note = QtWidgets.QLabel(
            "VarPro optimizer · shared nonlinear parameters across planes · independently "
            "solved signed amplitudes. Auto models follow NMRPipe processing metadata; "
            "review their scientific suitability. Bounds below are the actual first-pass inputs."
        )
        note.setWordWrap(True)
        advanced.addWidget(note)
        self.parameters = QtWidgets.QTableWidget(0, 7)
        self.parameters.setHorizontalHeaderLabels(
            ["Peak / axis", "Parameter", "Start", "Min", "Max", "Unit", "Freedom"]
        )
        self.parameters.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.parameters.verticalHeader().hide()
        # Size from a representative visible-sized batch, not every native table
        # delegate in a large dataset. Every parameter remains present/scrollable.
        self.parameters.horizontalHeader().setResizeContentsPrecision(100)
        self.parameters.setMinimumHeight(160)
        advanced.addWidget(self.parameters)
        self.advanced.hide()
        layout.addWidget(self.advanced, 1)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        cancel = self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        cancel.setAutoDefault(False)
        cancel.setDefault(False)
        self.run = self.buttons.addButton(
            "Run dataset fit", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.run.setDefault(True)
        self.run.setEnabled(False)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        for control in (self.passes, self.threshold, self.max_iterations, self.tolerance):
            control.valueChanged.connect(self._changed)
        self.lineshape.currentTextChanged.connect(self._changed)
        self._changed()

    def _toggle_advanced(self, visible: bool) -> None:
        self.advanced.setVisible(visible)
        self.advanced_toggle.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if visible else QtCore.Qt.ArrowType.RightArrow
        )
        self.resize(self.width(), 660 if visible else self.minimumSizeHint().height())

    def _changed(self) -> None:
        self._revision += 1
        self.prepared = None
        self.run.setEnabled(False)
        self.parameters.setRowCount(0)
        self.status.setText("Preparing scientific inputs…")
        self._timer.start()

    def _prepare(self) -> None:
        settings = self.configuration.model_copy(deep=True)
        settings.fitting.refine_iterations = self.passes.value()
        settings.fitting.lineshape = cast("LineshapeName", self.lineshape.currentText())
        settings.fitting.max_iterations = self.max_iterations.value()
        settings.fitting.tolerance = self.tolerance.value()
        settings.clustering.contour_level = None
        settings.clustering.contour_factor = self.threshold.value()
        self.configuration = settings
        self._worker.submit(
            "fit",
            self._revision,
            lambda: prepare_dataset_fit(
                self._spectra,
                self._peaks,
                settings,
            ),
        )

    def _prepared(self, _kind: str, revision: int, value: object) -> None:
        if revision != self._revision or not self.isVisible():
            return
        if isinstance(value, Exception):
            self.status.setText(f"Cannot prepare fit: {value}")
            return
        assert isinstance(value, PreparedDatasetFit)
        self.prepared = replace(
            value,
            snapshot=replace(
                value.snapshot,
                spectrum_id=self._source_id,
                source_files=self._source_files,
            ),
        )
        samples = sum(len(cluster.grid_indices[0]) for cluster in value.clusters)
        self.status.setText(
            f"{len(value.clusters)} clusters · {samples:,} samples per plane\n"
            f"Models: {' / '.join(value.shape_names)} · Statistical noise: {value.noise:.6g}\n"
            f"Clustering threshold: {value.contour_level:.6g} intensity units"
        )
        labels = {
            identity.engine_name: f"{number}. {identity.assignment}"
            for number, identity in enumerate(value.identities, 1)
        }
        self.parameters.setRowCount(len(value.parameters))
        for row, parameter in enumerate(value.parameters.values()):
            identifier = parameter.param_id
            assert identifier is not None
            for column, text in enumerate(
                (
                    f"{labels.get(identifier.peak_name, 'Cluster')} / {identifier.axis}",
                    identifier.label,
                    f"{parameter.value:.9g}",
                    f"{parameter.min:.9g}",
                    f"{parameter.max:.9g}",
                    parameter.unit,
                    "Vary" if parameter.vary else "Fixed",
                )
            ):
                self.parameters.setItem(row, column, QtWidgets.QTableWidgetItem(text))
        self.parameters.resizeColumnsToContents()
        self.run.setEnabled(True)
        self.run.setDefault(True)
