"""The per-axis lineshape model as a compact table: one row per spectral axis.

Columns are Axis · Family · J (Hz) · Fit J · Phase. A doublet's J range sits on a
line under its row, with × to remove it; a singlet's J cell offers **+ J** to add one.
Edits are drafts until they form a valid model. An invalid draft is explained under
its axis and blocks Fit; a valid one is committed as new working settings.
"""

import math
from dataclasses import dataclass
from itertools import pairwise

from PySide6 import QtCore, QtWidgets

from peakfit.engine.domain.config import PHASED_FAMILIES, AxisModel, Doublet, FitConfig
from peakfit.gui import formatting
from peakfit.gui import theme as themes
from peakfit.gui.editor.labels import text_label

FAMILY_NAMES = {
    "auto": "Automatic",
    "gaussian": "Gaussian",
    "lorentzian": "Lorentzian",
    "pvoigt": "Pseudo-Voigt",
    "sp1": "SP1",
    "sp2": "SP2",
    "no_apod": "No apodization",
}
# The automatic model falls back to pseudo-Voigt without recognized apodization; the
# fit stays available, with a visible caution (docs/decisions.md).
DEFAULT_FAMILY = "pvoigt"
DIFFERS = "•"  # marks a setting that differs from the active run's


@dataclass(frozen=True)
class AxisInfo:
    """One spectral axis: the engine's label, its name and what Automatic resolves to."""

    label: str
    name: str
    detected: str


@dataclass
class _Draft:
    """What one row shows; texts stay as typed until they form a valid doublet."""

    family: str
    doublet: tuple[str, str, str] | None  # J start, lower and upper bound
    vary: bool
    phase: bool
    adding: bool = False  # + J pressed, no J committed yet
    typed: bool = False  # something that is not a positive J was entered while adding


def _family_text(family: str, detected: str) -> str:
    return f"Auto · {FAMILY_NAMES[detected]}" if family == "auto" else FAMILY_NAMES[family]


def _parse_hz(text: str) -> float | None:
    try:
        value = formatting.parse_scaled(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def doublet_error(j: float | None, low: float | None, high: float | None) -> str:
    """Why a typed doublet is invalid, for the line under its axis; empty when valid."""
    if j is None or j <= 0:
        return "Enter J as a positive number of Hz."
    if low is None or high is None or low <= 0 or high <= 0:
        return "Enter both J bounds as positive numbers of Hz."
    hz = formatting.hz
    if low >= high:
        return f"The J range {hz(low)}–{hz(high)} Hz is empty. Make the lower bound smaller."
    if not low <= j <= high:
        return (
            f"J start {hz(j)} Hz is outside its range {hz(low)}–{hz(high)} Hz. "
            "Change the start or widen the range."
        )
    return ""


class _AxisRow:
    """The widgets of one axis: its main row, the J range line and its messages."""

    def __init__(self, table: AxisModelTable, axis: AxisInfo) -> None:
        name = axis.name
        self.axis = axis
        self.name = text_label(name)
        self.family = QtWidgets.QComboBox()
        self.family.setAccessibleName(f"{name} lineshape family")
        for key in FAMILY_NAMES:
            self.family.addItem(_family_text(key, axis.detected), key)
        self.family.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.family.setMinimumContentsLength(6)
        self.add = themes.link("+ J", f"Add a doublet on {name}: two lines split by J")
        self.add.setAccessibleName(f"Add doublet on {name}")
        self.j = _hz_edit(f"{name} J start (Hz)")
        self.j.setPlaceholderText("J")
        self.fit = QtWidgets.QCheckBox()
        self.fit.setAccessibleName(f"Fit J on {name}")
        self.fit.setToolTip(f"Fit J on {name} within its range; off holds J at its start")
        self.phase = QtWidgets.QCheckBox()
        self.phase.setAccessibleName(f"Fit phase on {name}")
        self.low = _hz_edit(f"{name} J lower bound (Hz)")
        self.high = _hz_edit(f"{name} J upper bound (Hz)")
        # A compact ×: the name and tooltip say what it removes.
        self.remove = themes.link("×", f"Remove the doublet on {name}")
        self.remove.setAccessibleName(f"Remove doublet on {name}")
        self.range = QtWidgets.QWidget()
        line = QtWidgets.QHBoxLayout(self.range)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(4)
        line.addWidget(text_label("J range", "muted"))
        line.addWidget(self.low)
        line.addWidget(text_label("–", "muted"))
        line.addWidget(self.high)
        line.addWidget(text_label("Hz", "muted"))
        line.addSpacing(themes.SPACE_XS)
        line.addWidget(self.remove)
        line.addStretch(1)
        self.caution = text_label("", "warning", wrap=True)
        self.error = text_label("", "error", wrap=True)
        self.family.currentIndexChanged.connect(table.commit)
        self.add.clicked.connect(lambda: table.start_doublet(self))
        self.j.editingFinished.connect(lambda: table.j_finished(self))
        for edit in (self.low, self.high):
            edit.editingFinished.connect(table.commit)
        self.fit.toggled.connect(table.commit)
        self.phase.toggled.connect(table.commit)
        self.remove.clicked.connect(lambda: table.remove_doublet(self))

    def tab_stops(self) -> list[QtWidgets.QWidget]:
        """J and its bounds come before the remaining settings of the axis."""
        return [
            self.family,
            self.add,
            self.j,
            self.low,
            self.high,
            self.remove,
            self.fit,
            self.phase,
        ]


def _hz_edit(name: str) -> QtWidgets.QLineEdit:
    edit = QtWidgets.QLineEdit()
    edit.setAccessibleName(name)
    edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
    edit.setFixedWidth(edit.fontMetrics().horizontalAdvance("000.0") + 16)
    return edit


class AxisModelTable(QtWidgets.QWidget):
    """Edit family, doublet and phase per spectral axis; commit only valid models."""

    changed = QtCore.Signal(object)  # FitConfig
    invalid_changed = QtCore.Signal(str)

    def __init__(self, axes: list[AxisInfo]) -> None:
        super().__init__()
        self.setObjectName("axisModel")
        self._fitting: FitConfig | None = None
        self._drafts: dict[str, _Draft] = {}
        self._updating = False
        self._pending: set[str] = set()
        self.invalid = ""
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(4)
        centre = QtCore.Qt.AlignmentFlag.AlignHCenter
        for column, text in enumerate(("Axis", "Family", "J (Hz)")):
            grid.addWidget(text_label(text, "caption"), 0, column)
        for column, text in ((3, "Fit J"), (4, "Phase")):  # over their check boxes
            grid.addWidget(text_label(text, "caption"), 0, column, centre)
        self.rows: list[_AxisRow] = []
        line = 1
        for axis in axes:
            row = _AxisRow(self, axis)
            self.rows.append(row)
            grid.addWidget(row.name, line, 0)
            grid.addWidget(row.family, line, 1)
            grid.addWidget(row.add, line, 2)
            grid.addWidget(row.j, line, 2)
            grid.addWidget(row.fit, line, 3, centre)
            grid.addWidget(row.phase, line, 4, centre)
            grid.addWidget(row.range, line + 1, 1, 1, 4)
            grid.addWidget(row.caution, line + 2, 1, 1, 4)
            grid.addWidget(row.error, line + 3, 1, 1, 4)
            line += 4
        grid.setColumnStretch(1, 1)
        stops = [stop for row in self.rows for stop in row.tab_stops()]
        for first, second in pairwise(stops):
            QtWidgets.QWidget.setTabOrder(first, second)

    def show_model(self, fitting: FitConfig, differs: frozenset[str] = frozenset()) -> None:
        """Show committed settings; ``differs`` names axes whose model the run did not use.

        Unchanged settings keep any draft in progress; changed ones (Undo, a reopened
        analysis) replace every draft.
        """
        for row in self.rows:
            marked = row.axis.label in differs
            row.name.setText(f"{row.axis.name} {DIFFERS}" if marked else row.axis.name)
            row.name.setToolTip("Differs from the active run's model" if marked else "")
        if fitting == self._fitting:
            return
        self._fitting = fitting
        for row in self.rows:
            model = fitting.axis_model(row.axis.label)
            d = model.doublet
            self._drafts[row.axis.label] = _Draft(
                family=str(model.family),
                doublet=(formatting.hz(d.j), formatting.hz(d.min), formatting.hz(d.max))
                if d
                else None,
                vary=d.vary if d else True,
                phase=model.phase,
            )
        self._render()

    # ------------------------------------------------------------- edits
    def start_doublet(self, row: _AxisRow) -> None:
        """+ J: open the J field; the doublet exists once a valid J is entered."""
        self._drafts[row.axis.label].adding = True
        self._render()
        row.j.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

    def j_finished(self, row: _AxisRow) -> None:
        """An empty J for a doublet being added cancels it; otherwise commit."""
        draft = self._drafts[row.axis.label]
        if draft.adding and not row.j.text().strip():
            draft.adding = False
            self._render()
            self._update_errors()
            return
        self.commit()

    def remove_doublet(self, row: _AxisRow) -> None:
        """Back to a singlet; the keyboard returns to + J."""
        draft = self._drafts[row.axis.label]
        draft.doublet, draft.adding, draft.vary = None, False, True
        self.commit()
        row.add.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

    def commit(self) -> None:
        """Read every row; emit the settings when all rows are valid."""
        if self._updating or self._fitting is None:
            return
        for row in self.rows:
            self._read(row)
        settings, errors = self._settings()
        self._render()
        if errors:
            return
        if settings != self._fitting:
            self.changed.emit(settings)

    def _read(self, row: _AxisRow) -> None:
        draft = self._drafts[row.axis.label]
        draft.family = row.family.currentData()
        draft.phase = row.phase.isChecked()
        if draft.adding and draft.doublet is None:
            parsed = _parse_hz(row.j.text().strip())
            if parsed is not None and parsed > 0:
                # A new doublet is fitted within [J/2, 2J] (ADR 0004), shown once committed.
                d = Doublet(j=parsed)
                draft.doublet = (row.j.text().strip(), formatting.hz(d.min), formatting.hz(d.max))
                draft.adding, draft.vary, draft.typed = False, True, False
            else:
                draft.typed = bool(row.j.text().strip())
            return  # still adding
        if draft.doublet is not None:
            draft.doublet = (row.j.text().strip(), row.low.text().strip(), row.high.text().strip())
            draft.vary = row.fit.isChecked()

    def _settings(self) -> tuple[FitConfig, dict[str, str]]:
        """The settings the drafts describe, and each invalid axis's reason."""
        assert self._fitting is not None
        errors: dict[str, str] = {}
        pending: set[str] = set()
        models: dict[str, AxisModel] = {}
        for row in self.rows:
            label, draft = row.axis.label, self._drafts[row.axis.label]
            doublet = None
            if draft.adding and draft.doublet is None:
                errors[label] = "Enter J as a positive number of Hz."
                if not draft.typed:
                    pending.add(label)  # nothing typed yet: blocks Fit, says nothing
            elif draft.doublet is not None:
                j, low, high = (_parse_hz(text) for text in draft.doublet)
                errors[label] = doublet_error(j, low, high)
                if j is not None and low is not None and high is not None and not errors[label]:
                    doublet = Doublet(j=j, min=low, max=high, vary=draft.vary)
            family = draft.family if draft.family != "auto" else row.axis.detected
            if draft.phase and family not in PHASED_FAMILIES and not errors.get(label):
                errors[label] = (
                    f"Phase needs a phased family ({_phased_names()}); "
                    f"{FAMILY_NAMES[family]} has none. Choose one or turn the phase off."
                )
            models[label] = AxisModel(doublet=doublet, phase=draft.phase)
        errors = {label: text for label, text in errors.items() if text}
        self._pending = pending
        families = {self._drafts[row.axis.label].family for row in self.rows}
        common = next(iter(families)) if len(families) == 1 else self._fitting.family
        axes = {label: model for label, model in self._fitting.axes.items() if label not in models}
        for label, model in models.items():
            own = self._drafts[label].family
            updated = model.model_copy(update={"family": own if own != common else None})
            if updated != AxisModel():
                axes[label] = updated
        return self._fitting.model_copy(update={"family": common, "axes": axes}), errors

    # ------------------------------------------------------------- display
    def _render(self) -> None:
        """Show the drafts; texts being typed are left as they are."""
        self._updating = True
        try:
            for row in self.rows:
                draft = self._drafts[row.axis.label]
                row.family.setCurrentIndex(row.family.findData(draft.family))
                family = draft.family if draft.family != "auto" else row.axis.detected
                row.family.setToolTip(
                    f"{row.axis.name}: Automatic resolves to {FAMILY_NAMES[family]} "
                    "from the apodization"
                    if draft.family == "auto"
                    else f"{row.axis.name}: {FAMILY_NAMES[family]}, chosen explicitly"
                )
                doublet = draft.doublet is not None
                row.add.setVisible(not doublet and not draft.adding)
                row.j.setVisible(doublet or draft.adding)
                row.fit.setVisible(doublet)
                row.range.setVisible(doublet)
                if doublet:
                    assert draft.doublet is not None
                    for edit, text in zip((row.j, row.low, row.high), draft.doublet, strict=True):
                        if edit.text() != text:
                            edit.setText(text)
                    row.fit.setChecked(draft.vary)
                    row.low.setEnabled(draft.vary)
                    row.high.setEnabled(draft.vary)
                elif not draft.adding:
                    row.j.clear()
                phased = family in PHASED_FAMILIES
                row.phase.setChecked(draft.phase)
                # A phase on a family without one stays switchable off.
                row.phase.setEnabled(phased or draft.phase)
                row.phase.setToolTip(
                    f"Fit one phase per cluster on {row.axis.name}"
                    if phased
                    else f"{FAMILY_NAMES[family]} has no phase; phase needs {_phased_names()}"
                )
                caution = draft.family == "auto" and row.axis.detected == DEFAULT_FAMILY
                row.caution.setText(
                    "▲ No recognized apodization. Check that the pseudo-Voigt default "
                    "suits these data, or choose a family."
                    if caution
                    else ""
                )
                row.caution.setVisible(caution)
        finally:
            self._updating = False
        self._update_errors()

    def _update_errors(self) -> None:
        _settings, errors = self._settings()
        for row in self.rows:
            label = row.axis.label
            text = errors.get(label, "") if label not in self._pending else ""
            row.error.setText(f"▲ {text}" if text else "")
            row.error.setVisible(bool(text))
            invalid = bool(text) and self._drafts[label].doublet is not None
            for edit in (row.j, row.low, row.high):
                edit.setProperty("invalid", invalid)
                edit.style().unpolish(edit)
                edit.style().polish(edit)
        first = next(
            (
                f"{row.axis.name}: {errors[row.axis.label]}"
                for row in self.rows
                if row.axis.label in errors
            ),
            "",
        )
        self._set_invalid(first)

    def _set_invalid(self, text: str) -> None:
        if text != self.invalid:
            self.invalid = text
            self.invalid_changed.emit(text)


def _phased_names() -> str:
    """``SP1, SP2 or No apodization``: the engine's phased families, by name."""
    names = [name for key, name in FAMILY_NAMES.items() if key in PHASED_FAMILIES]
    return f"{', '.join(names[:-1])} or {names[-1]}"
