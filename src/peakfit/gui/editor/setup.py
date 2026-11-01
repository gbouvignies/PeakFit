"""Fit setup in the inspector: the working scientific settings and their preparation.

The controls edit the working configuration only. What the setup reports about
statistical noise, clusters and samples comes from the same preparation that Fit
submits, never from display contours.
"""

import math
from typing import TYPE_CHECKING, cast, get_args

from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.engine.domain.config import LineshapeName
from peakfit.gui import theme as themes
from peakfit.gui.editor.labels import text_label

if TYPE_CHECKING:
    from peakfit.engine.domain.config import PeakFitConfig
    from peakfit.gui.fitting import PreparedDatasetFit

_LINESHAPES = {
    "auto": "Automatic",
    "gaussian": "Gaussian",
    "lorentzian": "Lorentzian",
    "pvoigt": "Pseudo-Voigt",
    "sp1": "SP1",
    "sp2": "SP2",
    "no_apod": "No apodization",
}

# Automatic models fall back to pseudo-Voigt without recognized apodization; the
# fit stays available, with a visible caution (docs/decisions.md).
DEFAULT_SOURCE = "no recognized apodization; default model"


def lineshape_text(name: str) -> str:
    """Readable model name; doublet variants keep their base name."""
    base = name.removesuffix("_doublet")
    text = _LINESHAPES.get(base, base)
    return f"{text} doublet" if name.endswith("_doublet") else text


class FitSetup(QtWidgets.QWidget):
    """Edit the working settings; show what their preparation resolves to."""

    changed = QtCore.Signal(object)  # PeakFitConfig
    regions_toggled = QtCore.Signal(bool)

    def __init__(self) -> None:  # noqa: PLR0915
        super().__init__()
        self._configuration: PeakFitConfig | None = None
        self._key = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(themes.SPACE_S)
        form = QtWidgets.QGridLayout()
        form.setHorizontalSpacing(themes.SPACE_M)
        form.setVerticalSpacing(6)
        form.setColumnStretch(1, 1)
        layout.addLayout(form)
        self.lineshape = QtWidgets.QComboBox()
        self.lineshape.setAccessibleName("Lineshape model")
        for name in get_args(LineshapeName):
            self.lineshape.addItem(lineshape_text(name), name)
        self.lineshape.currentIndexChanged.connect(self._edited)
        self.shapes = text_label("", "caption", wrap=True)
        self.shape_caution = text_label("", "warning", wrap=True)
        self.noise = text_label("", "", wrap=True)
        self.noise_source = text_label("", "caption", wrap=True)
        self.threshold = QtWidgets.QDoubleSpinBox()
        self.threshold.setRange(0.1, 1000)
        self.threshold.setDecimals(2)
        self.threshold.setSuffix(" × noise")
        self.threshold.setAccessibleName("Clustering threshold")
        self.threshold.setKeyboardTracking(False)
        self.threshold.valueChanged.connect(self._edited)
        self.level = text_label("", "caption", wrap=True)
        self.passes = QtWidgets.QSpinBox()
        self.passes.setRange(1, 20)
        self.passes.setAccessibleName("Optimizer passes")
        self.passes.setKeyboardTracking(False)
        self.passes.valueChanged.connect(self._edited)
        self.schedule = text_label("", "", wrap=True)
        self.passes_note = text_label("cross-talk updated between passes", "caption", wrap=True)
        rows = (
            ("Lineshapes", (self.lineshape, self.shapes, self.shape_caution)),
            ("Statistical noise", (self.noise, self.noise_source)),
            ("Clustering", (self.threshold, self.level)),
            ("Passes", (self.passes, self.schedule, self.passes_note)),
        )
        for row, (name, widgets) in enumerate(rows):
            title = text_label(name, "muted")
            form.addWidget(title, row, 0, QtCore.Qt.AlignmentFlag.AlignTop)
            cell = QtWidgets.QVBoxLayout()
            cell.setSpacing(2)
            for widget in widgets:
                cell.addWidget(widget)
            form.addLayout(cell, row, 1)
        self.advanced_toggle = QtWidgets.QToolButton()
        self.advanced_toggle.setObjectName("text")
        self.advanced_toggle.setText("Advanced")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        layout.addWidget(self.advanced_toggle, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        self.advanced = QtWidgets.QWidget()
        advanced = QtWidgets.QGridLayout(self.advanced)
        advanced.setContentsMargins(0, 0, 0, 0)
        advanced.setHorizontalSpacing(themes.SPACE_M)
        advanced.setVerticalSpacing(6)
        advanced.setColumnStretch(1, 1)
        self.max_iterations = QtWidgets.QSpinBox()
        self.max_iterations.setRange(1, 100000)
        self.max_iterations.setAccessibleName("Maximum optimizer iterations")
        self.max_iterations.setKeyboardTracking(False)
        self.max_iterations.valueChanged.connect(self._edited)
        self.tolerance = QtWidgets.QLineEdit()
        self.tolerance.setAccessibleName("Optimizer tolerance")
        validator = QtGui.QDoubleValidator(1e-12, 0.1, 15, self.tolerance)
        validator.setNotation(QtGui.QDoubleValidator.Notation.ScientificNotation)
        self.tolerance.setValidator(validator)
        self.tolerance.editingFinished.connect(self._edited)
        for row, (name, widget) in enumerate(
            (
                ("Solver", text_label("VarPro · signed amplitudes solved per plane", wrap=True)),
                ("Iteration limit", self.max_iterations),
                ("Tolerance", self.tolerance),
            )
        ):
            advanced.addWidget(text_label(name, "muted"), row, 0)
            advanced.addWidget(widget, row, 1)
        self.advanced.hide()
        layout.addWidget(self.advanced)
        # Fit the inspector width; numbers always use a decimal point.
        self.lineshape.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.lineshape.setMinimumContentsLength(6)
        for control in (self.lineshape, self.threshold, self.passes, self.max_iterations):
            control.setMinimumWidth(72)
            control.setLocale(QtCore.QLocale.c())
        validator.setLocale(QtCore.QLocale.c())
        self.preparation = text_label("", "", wrap=True)
        self.blocked = text_label("", "warning", wrap=True)
        layout.addSpacing(themes.SPACE_XS)
        layout.addWidget(self.preparation)
        layout.addWidget(self.blocked)
        self.regions = QtWidgets.QCheckBox("Show sampled fitting regions")
        self.regions.toggled.connect(self.regions_toggled)
        layout.addWidget(self.regions)
        layout.addWidget(
            text_label(
                "Clustering uses statistical noise from all planes; display contour "
                "settings never change clusters, samples or starts.",
                "caption",
                wrap=True,
            )
        )

    def _toggle_advanced(self, shown: bool) -> None:
        self.advanced.setVisible(shown)
        self.advanced_toggle.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if shown else QtCore.Qt.ArrowType.RightArrow
        )

    def show_state(
        self,
        configuration: PeakFitConfig,
        shapes: list[tuple[str, str, str]],
        preview: PreparedDatasetFit | None,
        error: str,
    ) -> None:
        """Reflect working settings and their preparation; cheap when nothing changed."""
        key = (configuration.model_dump_json(), tuple(shapes), id(preview), error)
        if key == self._key:
            return
        self._key = key
        self._configuration = configuration
        fitting = configuration.fitting
        with QtCore.QSignalBlocker(self.lineshape):
            self.lineshape.setCurrentIndex(self.lineshape.findData(fitting.lineshape))
        # Unchanged values are left alone so an edit in progress keeps its text.
        for control, value in (
            (self.passes, fitting.refine_iterations),
            (self.max_iterations, fitting.max_iterations),
        ):
            if control.value() != value:
                with QtCore.QSignalBlocker(control):
                    control.setValue(value)
        if self.threshold.value() != configuration.clustering.contour_factor:
            with QtCore.QSignalBlocker(self.threshold):
                self.threshold.setValue(configuration.clustering.contour_factor)
        if not self.tolerance.hasFocus():
            self.tolerance.setText(f"{fitting.tolerance:g}")
        self.shapes.setText(
            "\n".join(f"{axis}: {shape} · {source}" for axis, shape, source in shapes)
        )
        defaulted = [axis for axis, _shape, source in shapes if source == DEFAULT_SOURCE]
        self.shape_caution.setText(
            f"▲ {' and '.join(defaulted)}: no recognized apodization. Check that the "
            "pseudo-Voigt default suits these data, or choose a model."
            if defaulted
            else ""
        )
        self.shape_caution.setVisible(bool(defaulted))
        steps = fitting.steps
        self.passes.setVisible(not steps)
        self.passes_note.setVisible(not steps)
        self.schedule.setVisible(bool(steps))
        self.schedule.setText(f"{len(steps)}-step schedule (imported)")
        explicit = configuration.clustering.contour_level
        if preview is not None:
            self.noise.setText(f"{preview.noise:.4g}")
            self.level.setText(
                f"explicit level {preview.contour_level:.4g}; editing the factor replaces it"
                if explicit is not None
                else f"= {preview.contour_level:.4g} intensity"
            )
            samples = sum(len(cluster.grid_indices[0]) for cluster in preview.clusters)
            self.preparation.setText(
                f"{len(preview.clusters)} clusters · {samples:,} samples per plane".replace(
                    ",", " "
                )
            )
        else:
            self.noise.setText("–")
            self.level.setText("")
            self.preparation.setText("" if error else "Preparing…")
        self.noise_source.setText(
            "explicit value" if configuration.noise_level else "estimated from all planes"
        )
        self.blocked.setText(f"Fit unavailable: {error}" if error else "")
        self.blocked.setVisible(bool(error))

    def _edited(self) -> None:
        if self._configuration is None:
            return
        settings = self._configuration.model_copy(deep=True)
        fitting = settings.fitting
        fitting.lineshape = cast("LineshapeName", self.lineshape.currentData())
        fitting.refine_iterations = self.passes.value()
        fitting.max_iterations = self.max_iterations.value()
        try:
            tolerance = float(self.tolerance.text())
        except ValueError:
            tolerance = fitting.tolerance
        if math.isfinite(tolerance) and 0 < tolerance <= 0.1:  # noqa: PLR2004
            fitting.tolerance = tolerance
        factor = self.threshold.value()
        if factor != settings.clustering.contour_factor:
            settings.clustering.contour_level = None
            settings.clustering.contour_factor = factor
        if settings != self._configuration:
            self.changed.emit(settings)
