"""The inspector's Model page: the working fit settings and their preparation.

The controls edit the working configuration only; every committed change is one
undoable edit. What the page reports about statistical noise, clusters and samples
comes from the same preparation that Fit submits, never from display contours.
A setting that differs from the active run's is marked with •.
"""

import math
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.gui import formatting
from peakfit.gui import theme as themes
from peakfit.gui.editor.axis_table import DIFFERS, AxisInfo, AxisModelTable
from peakfit.gui.editor.labels import LABEL_WIDTH, disclosure, section, text_label

if TYPE_CHECKING:
    from peakfit.engine.domain.config import PeakFitConfig
    from peakfit.gui.fitting import PreparedDatasetFit

SOLVER = "VarPro · signed amplitudes solved per plane"


class ScaledValidator(QtGui.QValidator):
    """A number typed plain, in scientific notation or as the formatter shows it (1 × 10⁻⁸)."""

    def __init__(self, low: float, high: float, parent: QtCore.QObject) -> None:
        super().__init__(parent)
        self.low, self.high = low, high

    def validate(self, text: str, pos: int) -> object:
        """Acceptable once the text reads as a number within range."""
        try:
            value = formatting.parse_scaled(text)
        except ValueError:
            return QtGui.QValidator.State.Intermediate, text, pos
        inside = self.low <= value <= self.high
        state = QtGui.QValidator.State.Acceptable if inside else QtGui.QValidator.State.Intermediate
        return state, text, pos


def _form() -> QtWidgets.QGridLayout:
    grid = QtWidgets.QGridLayout()
    grid.setHorizontalSpacing(themes.SPACE_S)
    grid.setVerticalSpacing(6)
    grid.setColumnMinimumWidth(0, LABEL_WIDTH)  # the inspector's label column
    grid.setColumnStretch(1, 1)
    return grid


def _cell(*widgets: QtWidgets.QWidget) -> QtWidgets.QVBoxLayout:
    cell = QtWidgets.QVBoxLayout()
    cell.setSpacing(2)
    for widget in widgets:
        cell.addWidget(widget)
    return cell


class ModelPage(QtWidgets.QWidget):
    """Edit the working settings; show what their preparation resolves to."""

    changed = QtCore.Signal(object)  # PeakFitConfig
    invalid_changed = QtCore.Signal(str)
    regions_toggled = QtCore.Signal(bool)

    def __init__(self) -> None:  # noqa: PLR0915
        super().__init__()
        self._configuration: PeakFitConfig | None = None
        self._key = None
        self._planes = 0
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(themes.SPACE_S)
        self.differs = text_label("", "caption", wrap=True)
        self.differs.hide()
        layout.addWidget(self.differs)
        self.lineshape_title = section(layout, "Lineshape model")
        self.axes: AxisModelTable | None = None
        self._axes_slot = QtWidgets.QVBoxLayout()
        self._axes_slot.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._axes_slot)
        section(layout, "Clustering")
        clustering = _form()
        self.noise = text_label("", "")
        self.noise_source = text_label("explicit value", "caption")
        self.noise_title = text_label("Statistical noise", "muted")
        self.threshold = QtWidgets.QDoubleSpinBox()
        self.threshold.setRange(0.1, 1000)
        self.threshold.setDecimals(2)
        self.threshold.setSuffix(" × noise")
        self.threshold.setAccessibleName("Clustering threshold, in multiples of the noise")
        self.threshold.setKeyboardTracking(False)
        self.threshold.valueChanged.connect(self._edited)
        self.level = text_label("", "caption", wrap=True)
        self.threshold_title = text_label("Threshold", "muted")
        for row, (title, widgets) in enumerate(
            (
                (self.noise_title, (self.noise, self.noise_source)),
                (self.threshold_title, (self.threshold, self.level)),
            )
        ):
            clustering.addWidget(title, row, 0, QtCore.Qt.AlignmentFlag.AlignTop)
            clustering.addLayout(_cell(*widgets), row, 1)
        layout.addLayout(clustering)
        section(layout, "Fit")
        fit = _form()
        self.passes = QtWidgets.QSpinBox()
        self.passes.setRange(1, 20)
        self.passes.setAccessibleName("Optimizer passes")
        self.passes.setKeyboardTracking(False)
        self.passes.valueChanged.connect(self._edited)
        self.passes.setToolTip("Cross-talk between clusters is updated between passes")
        self.schedule = text_label("", "", wrap=True)
        self.passes_title = text_label("Passes", "muted")
        self.planes = text_label("", "", wrap=True)
        self.planes_title = text_label("Planes", "muted")
        for row, (title, widgets) in enumerate(
            (
                (self.passes_title, (self.passes, self.schedule)),
                (self.planes_title, (self.planes,)),
            )
        ):
            fit.addWidget(title, row, 0, QtCore.Qt.AlignmentFlag.AlignTop)
            fit.addLayout(_cell(*widgets), row, 1)
        layout.addLayout(fit)
        self.limits_toggle = disclosure("Optimizer limits")
        self.limits_toggle.toggled.connect(lambda shown: self.limits.setVisible(shown))
        self.limits_differs = text_label(DIFFERS, "muted")
        self.limits_differs.hide()
        toggle_row = QtWidgets.QHBoxLayout()
        toggle_row.setSpacing(2)
        toggle_row.addWidget(self.limits_toggle)
        toggle_row.addWidget(self.limits_differs)
        toggle_row.addStretch(1)
        layout.addLayout(toggle_row)
        self.limits = QtWidgets.QWidget()
        limits = _form()
        limits.setContentsMargins(0, 0, 0, 0)
        self.limits.setLayout(limits)
        self.max_iterations = QtWidgets.QSpinBox()
        self.max_iterations.setRange(1, 100000)
        self.max_iterations.setAccessibleName("Maximum optimizer iterations")
        self.max_iterations.setKeyboardTracking(False)
        self.max_iterations.valueChanged.connect(self._edited)
        self.tolerance = QtWidgets.QLineEdit()
        self.tolerance.setAccessibleName("Optimizer tolerance")
        self.tolerance.setValidator(ScaledValidator(1e-12, 0.1, self.tolerance))
        self.tolerance.editingFinished.connect(self._edited)
        self.iterations_title = text_label("Iteration limit", "muted")
        self.tolerance_title = text_label("Tolerance", "muted")
        for row, (title, widget) in enumerate(
            (
                (text_label("Solver", "muted"), text_label(SOLVER, wrap=True)),
                (self.iterations_title, self.max_iterations),
                (self.tolerance_title, self.tolerance),
            )
        ):
            limits.addWidget(title, row, 0)
            limits.addWidget(widget, row, 1)
        self.limits.hide()
        layout.addWidget(self.limits)
        for control in (self.threshold, self.passes, self.max_iterations):
            control.setMinimumWidth(72)
            control.setLocale(QtCore.QLocale.c())  # numbers always use a decimal point
        self.preparation = text_label("", "", wrap=True)
        self.blocked = text_label("", "warning", wrap=True)
        layout.addSpacing(themes.SPACE_XS)
        layout.addWidget(self.preparation)
        layout.addWidget(self.blocked)
        self.regions = QtWidgets.QCheckBox("Show sampled fitting regions")
        self.regions.setToolTip("Display contours never change clusters, samples or starts")
        self.regions.toggled.connect(self.regions_toggled)
        layout.addWidget(self.regions)

    def set_axes(self, axes: list[AxisInfo]) -> None:
        """Build the per-axis table for the dataset's spectral axes, once."""
        if self.axes is not None:
            return
        self.axes = AxisModelTable(axes)
        self.axes.changed.connect(self._axes_edited)
        self.axes.invalid_changed.connect(self.invalid_changed)
        self._axes_slot.addWidget(self.axes)

    @property
    def invalid(self) -> str:
        """Why the settings being edited cannot be fitted yet; empty when they can."""
        return self.axes.invalid if self.axes is not None else ""

    def show_state(
        self,
        configuration: PeakFitConfig,
        preview: PreparedDatasetFit | None,
        error: str,
        run: tuple[str, PeakFitConfig] | None = None,
    ) -> None:
        """Reflect working settings, their preparation and the active run's settings.

        ``run`` is the active run's short ID and recorded settings; cheap when
        nothing changed.
        """
        key = (configuration.model_dump_json(), id(preview), error, run and run[0], run and run[1])
        if key == self._key:
            return
        self._key = key
        self._configuration = configuration
        fitting = configuration.fitting
        recorded = run[1] if run else None
        labels = [row.axis.label for row in self.axes.rows] if self.axes is not None else []
        axes, differs = self._differences(configuration, recorded, labels)
        if self.axes is not None:
            self.axes.show_model(fitting, axes)
        self.differs.setText(
            f"{DIFFERS} differs from run {run[0]}" if run and (axes or differs) else ""
        )
        self.differs.setVisible(bool(run and (axes or differs)))
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
            self.tolerance.setText(formatting.setting(fitting.tolerance))
        steps = fitting.steps
        self.passes.setVisible(not steps)
        self.schedule.setVisible(bool(steps))
        self.schedule.setText(f"{len(steps)}-step schedule (imported)")
        excluded = configuration.exclude_planes
        explicit = configuration.clustering.contour_level
        if preview is not None:
            self.noise.setText(formatting.scaled(preview.noise))
            level = formatting.scaled(preview.contour_level)
            self.level.setText(
                f"explicit level {level}; editing the factor replaces it"
                if explicit is not None
                else f"= {level}"
            )
            samples = sum(len(cluster.grid_indices[0]) for cluster in preview.clusters)
            self.preparation.setText(
                f"{formatting.count(len(preview.clusters), 'cluster')} · "
                f"{formatting.count(samples, 'sample')} per plane"
            )
        else:
            self.noise.setText("–")
            self.level.setText("")
            self.preparation.setText("" if error else "Preparing…")
        self._show_planes(excluded)
        # The usual case goes in the tooltip; an explicit value is worth saying.
        self.noise_source.setVisible(bool(configuration.noise_level))
        self.noise.setToolTip(
            "Explicit value from the fit settings"
            if configuration.noise_level
            else "Estimated from all planes"
        )
        self.blocked.setText(f"Fit unavailable: {error}" if error else "")
        self.blocked.setVisible(bool(error))
        for title, name, key_ in (
            (self.noise_title, "Statistical noise", "noise"),
            (self.threshold_title, "Threshold", "threshold"),
            (self.passes_title, "Passes", "passes"),
            (self.planes_title, "Planes", "planes"),
            (self.iterations_title, "Iteration limit", "iterations"),
            (self.tolerance_title, "Tolerance", "tolerance"),
        ):
            tip = differs.get(key_)
            title.setText(f"{name} {DIFFERS}" if tip else name)
            title.setToolTip(f"Run {run[0]} used {tip}" if run and tip else "")
        limits = any(key_ in differs for key_ in ("iterations", "tolerance"))
        self.limits_differs.setVisible(limits)
        self.limits_differs.setToolTip("An optimizer limit differs from the active run's")

    def set_plane_count(self, planes: int) -> None:
        """The dataset's plane count, for the Planes row."""
        self._planes = planes

    def _show_planes(self, excluded: list[int]) -> None:
        total = self._planes
        if not excluded:
            self.planes.setText(f"All {formatting.grouped(total)} fitted")
            self.planes.setToolTip("")
            return
        self.planes.setText(f"{formatting.grouped(total - len(excluded))} of {total} fitted")
        self.planes.setToolTip(
            "Excluded by the settings: plane indices "
            + ", ".join(str(i) for i in sorted(excluded))
            + " (zero-based)"
        )

    @staticmethod
    def _differences(
        working: PeakFitConfig, recorded: PeakFitConfig | None, labels: list[str]
    ) -> tuple[frozenset[str], dict[str, str]]:
        """What differs from the run: the axes, and each other row with the run's value."""
        if recorded is None:
            return frozenset(), {}
        rows: dict[str, str] = {}
        w, r = working.fitting, recorded.fitting
        # Each axis's effective model, so a common family change marks every axis it moves.
        axes = frozenset(label for label in labels if w.axis_model(label) != r.axis_model(label))
        if working.noise_level != recorded.noise_level:
            rows["noise"] = (
                formatting.scaled(recorded.noise_level) if recorded.noise_level else "the estimate"
            )
        wc, rc = working.clustering, recorded.clustering
        if (wc.contour_factor, wc.contour_level) != (rc.contour_factor, rc.contour_level):
            rows["threshold"] = (
                f"level {formatting.scaled(rc.contour_level)}"
                if rc.contour_level is not None
                else f"{formatting.setting(rc.contour_factor)} × noise"
            )
        if (w.refine_iterations, w.steps) != (r.refine_iterations, r.steps):
            rows["passes"] = (
                f"a {len(r.steps)}-step schedule"
                if r.steps
                else formatting.count(r.refine_iterations, "pass", "passes")
            )
        if sorted(working.exclude_planes) != sorted(recorded.exclude_planes):
            rows["planes"] = (
                f"{len(recorded.exclude_planes)} excluded" if recorded.exclude_planes else "all"
            )
        if w.max_iterations != r.max_iterations:
            rows["iterations"] = formatting.grouped(r.max_iterations)
        if w.tolerance != r.tolerance:
            rows["tolerance"] = formatting.setting(r.tolerance)
        return axes, rows

    def _axes_edited(self, fitting) -> None:
        if self._configuration is None:
            return
        settings = self._configuration.model_copy(update={"fitting": fitting})
        if settings != self._configuration:
            self.changed.emit(settings)

    def _edited(self) -> None:
        if self._configuration is None:
            return
        settings = self._configuration.model_copy(deep=True)
        fitting = settings.fitting
        fitting.refine_iterations = self.passes.value()
        fitting.max_iterations = self.max_iterations.value()
        try:
            tolerance = formatting.parse_scaled(self.tolerance.text())
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
