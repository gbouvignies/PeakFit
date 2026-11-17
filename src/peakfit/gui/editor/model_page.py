"""The inspector's Model page: the working fit settings and their preparation.

The controls edit the working configuration only; every committed change is one
undoable edit. What the page reports about statistical noise, clusters and samples
comes from the same preparation that Fit submits, never from display contours.
A setting that differs from the active run's is marked with •.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.engine.domain.constraints import ParameterConfig
from peakfit.engine.domain.spectrum import format_planes
from peakfit.gui import formatting
from peakfit.gui import theme as themes
from peakfit.gui.editor.axis_table import DIFFERS, AxisInfo, AxisModelTable, axis_model_text
from peakfit.gui.editor.labels import LABEL_WIDTH, disclosure, section, text_label

if TYPE_CHECKING:
    from peakfit.engine.domain.config import PeakFitConfig
    from peakfit.gui.fitting import PreparedDatasetFit

SOLVER = "VarPro · signed amplitudes solved per plane"
TOLERANCE_RANGE = (1e-12, 0.1)  # an optimizer tolerance the setting accepts


@dataclass(frozen=True)
class RunSettings:
    """The active run's short ID and the settings it recorded."""

    run: str
    settings: PeakFitConfig


class Setting(Enum):
    """A Model page row below the axis table, with its label."""

    NOISE = "Statistical noise"
    THRESHOLD = "Threshold"
    PASSES = "Passes"
    PLANES = "Planes"
    CONSTRAINTS = "Constraints"
    ITERATIONS = "Iteration limit"
    TOLERANCE = "Tolerance"


LIMITS = (Setting.ITERATIONS, Setting.TOLERANCE)


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


def constraints_text(parameters: ParameterConfig, names: dict[str, str]) -> str:
    """Axis-wide and per-peak constraints in one line, or "None"."""
    if parameters == ParameterConfig():
        return "None"
    parts = []
    if parameters.position_window is not None:
        parts.append(f"positions ± {formatting.ppm(parameters.position_window)} ppm")
    windows = parameters.position_windows.model_dump(exclude_none=True)
    parts += [f"{names.get(axis, axis)} ± {formatting.ppm(w)} ppm" for axis, w in windows.items()]
    if parameters.defaults.patterns:
        parts.append(formatting.count(len(parameters.defaults.patterns), "default rule"))
    if parameters.peaks:
        parts.append(formatting.count(len(parameters.peaks), "peak") + " constrained")
    return " · ".join(parts)


def _rows(grid: QtWidgets.QGridLayout, rows) -> None:
    """Muted labels in the page's label column, each beside its stacked widgets."""
    for line, (title, widgets) in enumerate(rows):
        grid.addWidget(title, line, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        cell = QtWidgets.QVBoxLayout()
        cell.setSpacing(2)
        for widget in widgets:
            cell.addWidget(widget)
        grid.addLayout(cell, line, 1)


def _form() -> QtWidgets.QGridLayout:
    grid = QtWidgets.QGridLayout()
    grid.setHorizontalSpacing(themes.SPACE_S)
    grid.setVerticalSpacing(6)
    grid.setColumnMinimumWidth(0, LABEL_WIDTH)  # the inspector's label column
    grid.setColumnStretch(1, 1)
    return grid


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
        self.titles = {setting: text_label(setting.value, "muted") for setting in Setting}
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
        self.noise = text_label("", "")
        self.noise_source = text_label("explicit value", "caption")
        self.threshold = QtWidgets.QDoubleSpinBox()
        self.threshold.setRange(0.1, 1000)
        self.threshold.setDecimals(2)
        self.threshold.setSuffix(" × noise")
        self.threshold.setAccessibleName("Clustering threshold, in multiples of the noise")
        self.threshold.setKeyboardTracking(False)
        self.threshold.valueChanged.connect(self._edited)
        self.level = text_label("", "caption", wrap=True)
        clustering = _form()
        _rows(
            clustering,
            (
                (self.titles[Setting.NOISE], (self.noise, self.noise_source)),
                (self.titles[Setting.THRESHOLD], (self.threshold, self.level)),
            ),
        )
        layout.addLayout(clustering)
        section(layout, "Fit")
        self.passes = QtWidgets.QSpinBox()
        self.passes.setRange(1, 20)
        self.passes.setAccessibleName("Optimizer passes")
        self.passes.setKeyboardTracking(False)
        self.passes.valueChanged.connect(self._edited)
        self.passes.setToolTip("Cross-talk between clusters is updated between passes")
        self.schedule = text_label("", "", wrap=True)
        self.planes = text_label("", "", wrap=True)
        self.constraints = text_label("", "", wrap=True)
        fit = _form()
        _rows(
            fit,
            (
                (self.titles[Setting.PASSES], (self.passes, self.schedule)),
                (self.titles[Setting.PLANES], (self.planes,)),
                (self.titles[Setting.CONSTRAINTS], (self.constraints,)),
            ),
        )
        layout.addLayout(fit)
        self.limits_toggle = disclosure("Optimizer limits")
        self.limits_toggle.toggled.connect(lambda shown: self.limits.setVisible(shown))
        self.limits_differs = text_label(DIFFERS, "muted")
        self.limits_differs.setToolTip("An optimizer limit differs from the active run's")
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
        self.tolerance.setValidator(ScaledValidator(*TOLERANCE_RANGE, self.tolerance))
        self.tolerance.editingFinished.connect(self._edited)
        _rows(
            limits,
            (
                (text_label("Solver", "muted"), (text_label(SOLVER, wrap=True),)),
                (self.titles[Setting.ITERATIONS], (self.max_iterations,)),
                (self.titles[Setting.TOLERANCE], (self.tolerance,)),
            ),
        )
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

    def set_axes(self, axes: list[AxisInfo], planes: int) -> None:
        """Build the per-axis table for the dataset's spectral axes, once; its plane count."""
        self._planes = planes
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
        run: RunSettings | None = None,
    ) -> None:
        """Reflect working settings, their preparation and the active run's settings."""
        key = (configuration.model_dump_json(), id(preview), error, run)
        if key == self._key:
            return
        self._key = key
        self._configuration = configuration
        fitting = configuration.fitting
        axes, differs = self._differences(configuration, run)
        if self.axes is not None:
            self.axes.show_model(fitting, axes)
        marked = bool(run and (axes or differs))
        self.differs.setText(f"{DIFFERS} differs from run {run.run}" if run and marked else "")
        self.differs.setVisible(marked)
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
        self._show_planes(configuration.exclude_planes)
        self.constraints.setText(constraints_text(configuration.parameters, self._names()))
        # The usual case goes in the tooltip; an explicit value is worth saying.
        self.noise_source.setVisible(bool(configuration.noise_level))
        self.noise.setToolTip(
            "Explicit value from the fit settings"
            if configuration.noise_level
            else "Estimated from all planes"
        )
        self.blocked.setText(f"Fit unavailable: {error}" if error else "")
        self.blocked.setVisible(bool(error))
        for setting, title in self.titles.items():
            used = differs.get(setting)
            title.setText(f"{setting.value} {DIFFERS}" if used else setting.value)
            title.setToolTip(f"Run {run.run} used {used}" if run and used else "")
        self.limits_differs.setVisible(any(setting in differs for setting in LIMITS))

    def _names(self) -> dict[str, str]:
        return {row.axis.label: row.axis.name for row in self.axes.rows} if self.axes else {}

    def _show_planes(self, excluded: list[int]) -> None:
        total = self._planes
        if not excluded:
            self.planes.setText(f"All {formatting.grouped(total)} fitted")
            self.planes.setToolTip("")
            return
        self.planes.setText(f"{formatting.grouped(total - len(excluded))} of {total} fitted")
        self.planes.setToolTip(
            f"Excluded by the settings: planes {format_planes(sorted(excluded))} (zero-based)"
        )

    def _differences(
        self, working: PeakFitConfig, run: RunSettings | None
    ) -> tuple[dict[str, str], dict[Setting, str]]:
        """What differs from the run: by axis and by row, each with the run's value."""
        if run is None:
            return {}, {}
        recorded = run.settings
        w, r = working.fitting, recorded.fitting
        axes: dict[str, str] = {}
        for row in self.axes.rows if self.axes else ():
            label = row.axis.label
            # Each axis's effective model, so a common family change marks every axis.
            if w.axis_model(label) != r.axis_model(label):
                used = axis_model_text(r.axis_model(label), row.axis.detected)
                axes[label] = f"Run {run.run} used {used}"
        rows: dict[Setting, str] = {}
        if working.noise_level != recorded.noise_level:
            rows[Setting.NOISE] = (
                formatting.scaled(recorded.noise_level) if recorded.noise_level else "the estimate"
            )
        wc, rc = working.clustering, recorded.clustering
        if (wc.contour_factor, wc.contour_level) != (rc.contour_factor, rc.contour_level):
            rows[Setting.THRESHOLD] = (
                f"level {formatting.scaled(rc.contour_level)}"
                if rc.contour_level is not None
                else f"{formatting.setting(rc.contour_factor)} × noise"
            )
        if (w.refine_iterations, w.steps) != (r.refine_iterations, r.steps):
            rows[Setting.PASSES] = (
                f"a {len(r.steps)}-step schedule"
                if r.steps
                else formatting.count(r.refine_iterations, "pass", "passes")
            )
        if sorted(working.exclude_planes) != sorted(recorded.exclude_planes):
            rows[Setting.PLANES] = (
                f"planes {format_planes(sorted(recorded.exclude_planes))} excluded"
                if recorded.exclude_planes
                else "every plane"
            )
        if working.parameters != recorded.parameters:
            rows[Setting.CONSTRAINTS] = constraints_text(recorded.parameters, self._names())
        if w.max_iterations != r.max_iterations:
            rows[Setting.ITERATIONS] = formatting.grouped(r.max_iterations)
        if w.tolerance != r.tolerance:
            rows[Setting.TOLERANCE] = formatting.setting(r.tolerance)
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
        low, high = TOLERANCE_RANGE
        if math.isfinite(tolerance) and low <= tolerance <= high:
            fitting.tolerance = tolerance
        factor = self.threshold.value()
        if factor != settings.clustering.contour_factor:
            settings.clustering.contour_level = None
            settings.clustering.contour_factor = factor
        if settings != self._configuration:
            self.changed.emit(settings)
