"""The inspector's Model page: the working fit settings and their preparation.

The controls edit the working configuration only; every committed change is one
undoable edit. What the page reports about statistical noise, clusters and samples
comes from the same preparation that Fit submits, never from display contours.
A setting that differs from the active run's is marked with •. A typed value is a
draft until it is valid: an invalid one is explained under its setting, keeps the
committed value and blocks Fit.
"""

import math
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.engine.domain.constraints import ParameterConfig
from peakfit.engine.domain.spectrum import fitted_plane_indices
from peakfit.gui import formatting
from peakfit.gui import theme as themes
from peakfit.gui.editor.axis_table import DIFFERS, AxisInfo, AxisModelTable, axis_model_text
from peakfit.gui.editor.constraints import ConstraintsEditor, parse_number
from peakfit.gui.editor.labels import (
    LABEL_WIDTH,
    ElidedLabel,
    disclosure,
    mark_invalid,
    section,
    show_message,
    text_label,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

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
    """A Model page setting below the axis table, with its label."""

    NOISE = "Statistical noise"
    THRESHOLD = "Threshold"
    PASSES = "Passes"
    PLANES = "Planes"
    CONSTRAINTS = "Constraints"
    ITERATIONS = "Iteration limit"
    TOLERANCE = "Tolerance"


LIMITS = (Setting.ITERATIONS, Setting.TOLERANCE)
# The settings with a label in the page's label column (Constraints has a disclosure).
LABELLED = tuple(setting for setting in Setting if setting is not Setting.CONSTRAINTS)
# The settings typed into a field and validated by the page itself.
TYPED = (Setting.NOISE, Setting.THRESHOLD, Setting.PLANES)


class NoiseSource(StrEnum):
    """Where the statistical noise comes from."""

    ESTIMATED = "estimated"
    MANUAL = "manual"


class ThresholdKind(StrEnum):
    """How the clustering threshold is given."""

    FACTOR = "factor"
    LEVEL = "level"


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
        parts.append(formatting.count(len(parameters.defaults.patterns), "rule"))
    if parameters.peaks:
        parts.append(formatting.count(len(parameters.peaks), "peak") + " constrained")
    return " · ".join(parts) or "None"


def planes_fitted(total: int, excluded: Sequence[int]) -> str:
    """``All 131 fitted``, or ``128 of 131 fitted``."""
    if not excluded:
        return f"All {formatting.grouped(total)} fitted"
    return f"{formatting.grouped(total - len(excluded))} of {formatting.grouped(total)} fitted"


def _cell(*items: QtWidgets.QWidget | QtWidgets.QLayout) -> QtWidgets.QVBoxLayout:
    """Widgets and layouts stacked in one form cell."""
    cell = QtWidgets.QVBoxLayout()
    cell.setSpacing(2)
    for item in items:
        if isinstance(item, QtWidgets.QLayout):
            cell.addLayout(item)
        else:
            cell.addWidget(item)
    return cell


def _line(*widgets: QtWidgets.QWidget) -> QtWidgets.QHBoxLayout:
    """Widgets side by side, left-aligned."""
    line = QtWidgets.QHBoxLayout()
    line.setSpacing(4)
    for widget in widgets:
        line.addWidget(widget)
    line.addStretch(1)
    return line


def _rows(grid: QtWidgets.QGridLayout, rows) -> None:
    """Muted labels in the page's label column, each beside its cell."""
    for line, (title, cell) in enumerate(rows):
        grid.addWidget(title, line, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        grid.addLayout(cell, line, 1)


def _form() -> QtWidgets.QGridLayout:
    grid = QtWidgets.QGridLayout()
    grid.setHorizontalSpacing(themes.SPACE_S)
    grid.setVerticalSpacing(6)
    grid.setColumnMinimumWidth(0, LABEL_WIDTH)  # the inspector's label column
    grid.setColumnStretch(1, 1)
    return grid


def _value_edit(name: str) -> QtWidgets.QLineEdit:
    """A typed number, as wide as its text: a value is never shown cut."""
    edit = QtWidgets.QLineEdit()
    edit.setAccessibleName(name)
    edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
    metrics = edit.fontMetrics()
    least = metrics.horizontalAdvance("0.000 × 10⁰")

    def fit(text: str) -> None:
        edit.setFixedWidth(max(least, metrics.horizontalAdvance(text)) + 12)

    edit.textChanged.connect(fit)
    fit("")
    return edit


THRESHOLD_TIP = (
    "Selects the fitting regions: the points above it, in regions around the peaks, "
    "are the points fitted, and peaks that share a region are fitted together"
)
NOT_APPLIED = "Not applied: "  # an invalid entry keeps the setting in effect
ENTER_NOISE = "enter the statistical noise as a positive number."
ENTER_LEVEL = "enter the clustering level as a positive intensity."
NO_PLANE_LEFT = "every plane would be excluded. Keep at least one plane to fit."
_UNSET = object()


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
        self._estimate: float | None = None  # the noise estimated from the fitted planes
        self._level: float | None = None  # the clustering level the settings give
        self._committed: dict[Setting, object] = {}  # the value each field last showed
        self._exact: dict[QtWidgets.QLineEdit, tuple[str, float]] = {}  # shown text → value
        self.invalid = ""
        self.titles = {setting: text_label(setting.value, "muted") for setting in LABELLED}
        self.errors = {setting: text_label("", "error", wrap=True) for setting in TYPED}
        for message in self.errors.values():
            message.hide()
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
        # Each value, then where it comes from: one line per setting. Statistical
        # noise is the estimate, or a value entered instead of it.
        self.noise = text_label("", "")
        self.noise_value = _value_edit("Statistical noise")
        self.noise_value.editingFinished.connect(self._edited)
        self.noise_mode = themes.MenuChoice(
            "Statistical noise source",
            "Estimate the noise from the fitted planes, or enter it",
            (
                ("estimated", "Estimated from the fitted planes", NoiseSource.ESTIMATED),
                ("manual", "Manual value", NoiseSource.MANUAL),
            ),
        )
        self.noise_mode.currentIndexChanged.connect(self._noise_mode_chosen)
        self.noise_estimate = text_label("", "caption")
        # The threshold selects the fitting regions: the points above it are fitted,
        # and peaks whose regions touch share a cluster.
        self.threshold = QtWidgets.QDoubleSpinBox()
        self.threshold.setRange(0.1, 1000)
        self.threshold.setDecimals(2)
        self.threshold.setAccessibleName("Clustering threshold, in multiples of the noise")
        self.threshold.setKeyboardTracking(False)
        self.threshold.valueChanged.connect(self._edited)
        self.threshold.setFixedWidth(self.threshold.fontMetrics().horizontalAdvance("1000.00") + 30)
        self.level_value = _value_edit("Clustering level")
        self.level_value.editingFinished.connect(self._edited)
        self.threshold_mode = themes.MenuChoice(
            "Clustering threshold kind",
            THRESHOLD_TIP,
            (
                ("× noise", "A factor of the statistical noise", ThresholdKind.FACTOR),
                ("level", "An absolute intensity level", ThresholdKind.LEVEL),
            ),
        )
        self.threshold_mode.currentIndexChanged.connect(self._threshold_mode_chosen)
        self.level = text_label("", "caption", wrap=True)
        self.titles[Setting.THRESHOLD].setToolTip(THRESHOLD_TIP)
        clustering = _form()
        _rows(
            clustering,
            (
                (
                    self.titles[Setting.NOISE],
                    _cell(
                        _line(self.noise, self.noise_value, self.noise_mode),
                        self.noise_estimate,
                        self.errors[Setting.NOISE],
                    ),
                ),
                (
                    self.titles[Setting.THRESHOLD],
                    _cell(
                        _line(self.threshold, self.level_value, self.threshold_mode),
                        self.level,
                        self.errors[Setting.THRESHOLD],
                    ),
                ),
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
        # Planes left out of the fit, by the plane numbers navigation shows.
        self.excluded = QtWidgets.QLineEdit()
        self.excluded.setAccessibleName("Planes excluded from the fit, by plane number")
        self.excluded.setPlaceholderText("none")  # the caption says how planes are numbered
        self.excluded.setToolTip(
            "Plane numbers as in plane navigation, such as 1, 41–42. An excluded plane "
            "stays measured and shown, with no fitted amplitude."
        )
        self.excluded.editingFinished.connect(self._edited)
        self.planes = text_label("", "caption", wrap=True)
        fit = _form()
        _rows(
            fit,
            (
                (self.titles[Setting.PASSES], _cell(_line(self.passes), self.schedule)),
                (
                    self.titles[Setting.PLANES],
                    _cell(
                        _line(text_label("excluded", "muted"), self.excluded),
                        self.planes,
                        self.errors[Setting.PLANES],
                    ),
                ),
            ),
        )
        layout.addLayout(fit)
        # Constraints: a summary while collapsed, the editor once disclosed.
        self.constraints_toggle = disclosure("Constraints")
        self.constraints_toggle.toggled.connect(self._show_constraints)
        self.constraints_differs = text_label(DIFFERS, "muted")
        self.constraints_differs.hide()
        self.constraints_summary = ElidedLabel()
        self.constraints_summary.setObjectName("caption")
        constraints_row = QtWidgets.QHBoxLayout()
        constraints_row.setSpacing(2)
        left = QtCore.Qt.AlignmentFlag.AlignLeft
        constraints_row.addWidget(self.constraints_toggle, 0, left)
        constraints_row.addWidget(self.constraints_differs, 0, left)
        constraints_row.addSpacing(themes.SPACE_XS)
        constraints_row.addWidget(self.constraints_summary, 1)
        constraints_row.addStretch(0)
        layout.addLayout(constraints_row)
        self.constraints: ConstraintsEditor | None = None
        self._constraints_slot = QtWidgets.QVBoxLayout()
        self._constraints_slot.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._constraints_slot)
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
                (text_label("Solver", "muted"), _cell(text_label(SOLVER, wrap=True))),
                (self.titles[Setting.ITERATIONS], _cell(self.max_iterations)),
                (self.titles[Setting.TOLERANCE], _cell(self.tolerance)),
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
        self._show_modes()

    def set_axes(self, axes: list[AxisInfo], planes: int) -> None:
        """Build the per-axis table and constraints for the dataset's axes, once."""
        self._planes = planes
        if self.axes is not None:
            return
        self.axes = AxisModelTable(axes)
        self.axes.changed.connect(self._axes_edited)
        self.axes.invalid_changed.connect(self._update_invalid)
        self._axes_slot.addWidget(self.axes)
        self.constraints = ConstraintsEditor(axes)
        self.constraints.changed.connect(self._constraints_edited)
        self.constraints.invalid_changed.connect(self._update_invalid)
        self.constraints.setVisible(self.constraints_toggle.isChecked())
        self._constraints_slot.addWidget(self.constraints)
        # The editor joins the keyboard order where it is shown: after its toggle.
        QtWidgets.QWidget.setTabOrder(self.constraints_toggle, self.constraints)
        QtWidgets.QWidget.setTabOrder(self.constraints, self.limits_toggle)

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
        self._show_committed(configuration)
        self._show_preview(configuration, preview, error)
        self.noise.setToolTip("Estimated from the fitted planes")
        self._show_planes(configuration.exclude_planes)
        if self.constraints is not None:
            self.constraints.show_parameters(
                configuration.parameters, tuple(preview.parameters) if preview else None
            )
        self.blocked.setText(f"Fit unavailable: {error}" if error else "")
        self.blocked.setVisible(bool(error))
        for setting, title in self.titles.items():
            used = differs.get(setting)
            title.setText(f"{setting.value} {DIFFERS}" if used else setting.value)
            about = THRESHOLD_TIP if setting is Setting.THRESHOLD else ""
            title.setToolTip(f"Run {run.run} used {used}" if run and used else about)
        used = differs.get(Setting.CONSTRAINTS)
        self.constraints_differs.setVisible(bool(used))
        self.constraints_differs.setToolTip(f"Run {run.run} used {used}" if run and used else "")
        self.limits_differs.setVisible(any(setting in differs for setting in LIMITS))
        self._show_modes()
        self._validate()

    def _show_preview(
        self, configuration: PeakFitConfig, preview: PreparedDatasetFit | None, error: str
    ) -> None:
        """What the preparation resolved: noise, estimate, level and counts."""
        if preview is not None:
            # What Manual and Level start from: the last prepared values, kept while
            # the next preparation runs.
            self._estimate = preview.estimated_noise
            self._level = preview.contour_level
            self.noise.setText(formatting.scaled(preview.noise))
            self.noise_estimate.setText(
                f"estimate {formatting.scaled(preview.estimated_noise)}"
                if preview.estimated_noise is not None
                else "no estimate from these planes"
            )
            self.level.setText(
                f"= {formatting.scaled(preview.contour_level / preview.noise, 3)} × noise"
                if configuration.clustering.contour_level is not None
                else f"= {formatting.scaled(preview.contour_level)}"
            )
            samples = sum(len(cluster.grid_indices[0]) for cluster in preview.clusters)
            self.preparation.setText(
                f"{formatting.count(len(preview.clusters), 'cluster')} · "
                f"{formatting.count(samples, 'sample')} per plane"
            )
        else:
            self.noise.setText("–")
            self.noise_estimate.setText("")
            self.level.setText("")
            self.preparation.setText("" if error else "Preparing…")

    def discard_drafts(self) -> None:
        """Drop every value being typed: the next state shows the committed settings.

        Undo and Redo replace the settings, so nothing typed against the old ones stays.
        """
        self._committed.clear()
        self._key = None
        if self.constraints is not None:
            self.constraints.discard_drafts()

    def _show_committed(self, configuration: PeakFitConfig) -> None:
        """Show each field's committed value, replacing a draft only when the value changed."""
        noise = configuration.noise_level
        if self._committed.get(Setting.NOISE, _UNSET) != noise:
            self._committed[Setting.NOISE] = noise
            self._set_mode(
                self.noise_mode,
                NoiseSource.MANUAL if noise is not None else NoiseSource.ESTIMATED,
            )
            self._show_value(self.noise_value, noise)
        level = configuration.clustering.contour_level
        if self._committed.get(Setting.THRESHOLD, _UNSET) != level:
            self._committed[Setting.THRESHOLD] = level
            self._set_mode(
                self.threshold_mode,
                ThresholdKind.LEVEL if level is not None else ThresholdKind.FACTOR,
            )
            self._show_value(self.level_value, level)
        excluded = tuple(configuration.exclude_planes)
        if self._committed.get(Setting.PLANES, _UNSET) != excluded:
            self._committed[Setting.PLANES] = excluded
            self.excluded.setText(formatting.plane_numbers(excluded))

    def _show_value(self, edit: QtWidgets.QLineEdit, value: float | None) -> None:
        """A committed value in its shortest exact form; that text reads back exactly."""
        text = formatting.setting(value) if value is not None else ""
        edit.setText(text)
        if value is not None:
            self._exact[edit] = (text, value)

    @staticmethod
    def _set_mode(choice: themes.MenuChoice, mode: StrEnum) -> None:
        with QtCore.QSignalBlocker(choice):
            choice.setCurrentIndex(choice.findData(mode))

    @property
    def _noise_source(self) -> NoiseSource:
        return NoiseSource(self.noise_mode.currentData())

    @property
    def _threshold_kind(self) -> ThresholdKind:
        return ThresholdKind(self.threshold_mode.currentData())

    def _show_modes(self) -> None:
        manual = self._noise_source is NoiseSource.MANUAL
        self.noise.setVisible(not manual)
        self.noise_value.setVisible(manual)
        level = self._threshold_kind is ThresholdKind.LEVEL
        self.threshold.setVisible(not level)
        self.level_value.setVisible(level)

    def _names(self) -> dict[str, str]:
        return {row.axis.label: row.axis.name for row in self.axes.rows} if self.axes else {}

    def _show_planes(self, excluded: list[int]) -> None:
        # The convention, said once: navigation's numbers, never the TOML's indices.
        numbering = f"numbered\u00a01–{self._planes}"
        fitted = planes_fitted(self._planes, excluded).replace(" ", "\u00a0")
        self.planes.setText(f"{fitted} · {numbering}")

    # ------------------------------------------------------------- drafts
    def _number(self, edit: QtWidgets.QLineEdit) -> float | None:
        """The positive value a field holds, or None when it holds none."""
        text = edit.text().strip()
        shown = self._exact.get(edit)
        if shown is not None and shown[0] == text:
            return shown[1]  # unchanged: the committed value itself, not its rounding
        try:
            value = parse_number(text)
        except ValueError:
            return None
        return value if value is not None and value > 0 else None

    def _excluded(self) -> tuple[tuple[int, ...] | None, str]:
        """The zero-based planes the field excludes, or None and why it cannot."""
        text = self.excluded.text().strip()
        if not text:
            return (), ""
        try:
            planes = tuple(p - 1 for p in formatting.parse_planes(text, self._planes))
        except ValueError as error:
            return None, f"{error}. Enter plane numbers such as 1, 41–42."
        try:
            fitted_plane_indices(self._planes, planes)
        except ValueError:
            return None, NO_PLANE_LEFT
        return planes, ""

    def _field_errors(self) -> dict[Setting, str]:
        errors: dict[Setting, str] = {}
        if self._noise_source is NoiseSource.MANUAL and self._number(self.noise_value) is None:
            errors[Setting.NOISE] = ENTER_NOISE
        if self._threshold_kind is ThresholdKind.LEVEL and self._number(self.level_value) is None:
            errors[Setting.THRESHOLD] = ENTER_LEVEL
        _planes, why = self._excluded()
        if why:
            errors[Setting.PLANES] = why
        return errors

    def _validate(self) -> None:
        """Explain each invalid field under it, outline it, and update ``invalid``."""
        errors = self._field_errors()
        fields = {
            Setting.NOISE: self.noise_value,
            Setting.THRESHOLD: self.level_value,
            Setting.PLANES: self.excluded,
        }
        for setting, message in self.errors.items():
            text = errors.get(setting, "")
            show_message(message, NOT_APPLIED + text if text else "", hint=False)
            mark_invalid(fields[setting], bool(text))
        # A value derived from the setting in effect never sits beside an entry that
        # is not applied: it would describe a value the field does not show.
        manual = self._noise_source is NoiseSource.MANUAL
        self.noise_estimate.setVisible(
            manual and bool(self.noise_estimate.text()) and Setting.NOISE not in errors
        )
        self.level.setVisible(bool(self.level.text()) and Setting.THRESHOLD not in errors)
        self.planes.setVisible(Setting.PLANES not in errors)
        self._update_invalid()

    def _update_invalid(self, *_args) -> None:
        errors = self._field_errors()
        reasons = [
            self.axes.invalid if self.axes is not None else "",
            *(f"{setting.value}: {errors[setting]}" for setting in TYPED if setting in errors),
            self.constraints.invalid if self.constraints is not None else "",
        ]
        first = next((reason for reason in reasons if reason), "")
        if self.constraints is not None and self.constraints.invalid:
            self.constraints_summary.setText("▲ An entry is not applied")
        elif self._configuration is not None:
            parameters = self._configuration.parameters
            self.constraints_summary.setText(constraints_text(parameters, self._names()))
        if first != self.invalid:
            self.invalid = first
            self.invalid_changed.emit(first)

    # ------------------------------------------------------------- edits
    def _noise_mode_chosen(self) -> None:
        """Manual starts from the estimate as shown; Estimated drops the value."""
        if self._noise_source is NoiseSource.MANUAL:
            start = formatting.scaled(self._estimate) if self._estimate is not None else ""
            self.noise_value.setText(start)
            self.noise_value.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
        self._show_modes()
        self._edited()

    def _threshold_mode_chosen(self) -> None:
        """Level starts from the level the factor gives, as shown; Factor drops it."""
        if self._threshold_kind is ThresholdKind.LEVEL:
            level = self._level
            self.level_value.setText(formatting.scaled(level) if level is not None else "")
            self.level_value.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
        self._show_modes()
        self._edited()

    def _show_constraints(self, shown: bool) -> None:
        if self.constraints is not None:
            self.constraints.setVisible(shown)
        self.constraints_summary.setVisible(not shown)

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
                formatting.setting(recorded.noise_level)
                if recorded.noise_level is not None
                else "the estimate"
            )
        wc, rc = working.clustering, recorded.clustering
        if (wc.contour_factor, wc.contour_level) != (rc.contour_factor, rc.contour_level):
            rows[Setting.THRESHOLD] = (
                f"level {formatting.setting(rc.contour_level)}"
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
                f"planes {formatting.plane_numbers(recorded.exclude_planes)} excluded"
                if recorded.exclude_planes
                else "every plane"
            )
        if working.parameters != recorded.parameters:
            used = constraints_text(recorded.parameters, self._names())
            rows[Setting.CONSTRAINTS] = "no constraints" if used == "None" else used
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

    def _constraints_edited(self, parameters: ParameterConfig) -> None:
        if self._configuration is None:
            return
        settings = self._configuration.model_copy(update={"parameters": parameters})
        if settings != self._configuration:
            self.changed.emit(settings)

    def _edited(self) -> None:
        """Commit every valid field; an invalid one keeps its committed value."""
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
        if self._noise_source is NoiseSource.ESTIMATED:
            settings.noise_level = None
        elif (noise := self._number(self.noise_value)) is not None:
            settings.noise_level = noise
        clustering = settings.clustering
        if self._threshold_kind is ThresholdKind.FACTOR:
            clustering.contour_level = None
            clustering.contour_factor = self.threshold.value()
        elif (level := self._number(self.level_value)) is not None:
            clustering.contour_level = level
        excluded, _why = self._excluded()
        if excluded is not None:
            settings.exclude_planes = list(excluded)
        self._validate()
        if settings != self._configuration:
            self.changed.emit(settings)
