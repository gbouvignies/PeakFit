"""Axis-wide parameter constraints: position windows and rules on a parameter kind.

Each rule is one ``[parameters.defaults]`` pattern, ``*.<axis>.<kind>`` with ``*``
for every axis, setting a start, a range or holding the parameter. Rules apply in
their order, so a later rule overrides an earlier one where both match. A peak's own
constraints are its working peak's limits, edited on its Peak page.

Edits are drafts until they form a valid configuration. An invalid draft is
explained under its rule and blocks Fit; a valid one is committed.
"""

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtWidgets

from peakfit.engine.domain.constraints import (
    ParameterConfig,
    ParameterConstraint,
    PositionWindowConfig,
    matches_pattern,
)
from peakfit.gui import formatting
from peakfit.gui import theme as themes
from peakfit.gui.editor.labels import mark_invalid, show_message, text_label
from peakfit.gui.fitting import desktop_rule

if TYPE_CHECKING:
    from collections.abc import Collection

    from peakfit.gui.editor.axis_table import AxisInfo

# The parameter kinds a rule constrains, with their unit. Positions are limited by
# the position window, and J and phase are set in the lineshape table.
KINDS: dict[str, tuple[str, str]] = {"lw": ("LW", "Hz"), "eta": ("η", "")}  # as tables say
ALL_AXES = "*"
WINDOW_AXES = set(PositionWindowConfig.model_fields)  # the axes a window can name
NEW_RULE = "Set a start, a range or Hold; × removes the rule."
_DESKTOP = re.compile(r"\*\.([^.]+)\.([^.]+)")  # the rules the desktop fit supports


def parse_number(text: str) -> float | None:
    """A typed finite number, or None; raises ValueError for text that is not one."""
    text = text.strip()
    if not text:
        return None
    value = formatting.parse_scaled(text)
    if not math.isfinite(value):
        raise ValueError(text)
    return value


INVALID = object()  # a typed window that is not a positive number
WINDOW = "window"  # the error key of the position windows, beside each rule's index


def window_value(text: str) -> float | object | None:
    """A typed position window: a positive number, None when empty, else ``INVALID``."""
    try:
        value = parse_number(text)
    except ValueError:
        return INVALID
    if value is None:
        return None
    return value if value > 0 else INVALID


def value_text(value: float | None) -> str:
    """A configured value as a field shows it; "" for none."""
    return "" if value is None else formatting.setting(value)


@dataclass
class _RuleDraft:
    """One rule as shown; texts stay as typed until they form a valid constraint."""

    axis: str  # an axis label or ALL_AXES
    kind: str
    start: str = ""
    low: str = ""
    high: str = ""
    held: bool = False
    free: bool | None = None  # the rule's vary when not held: None (default) or True
    raw: str = ""  # an imported pattern other than linewidth or η, shown as written

    @property
    def pattern(self) -> str:
        return self.raw or f"*.{self.axis}.{self.kind}"

    @classmethod
    def of(cls, pattern: str, constraint: ParameterConstraint) -> _RuleDraft:
        match = _DESKTOP.fullmatch(pattern)
        known = match is not None and match.group(2) in KINDS
        return cls(
            axis=match.group(1) if known and match else ALL_AXES,
            kind=match.group(2) if known and match else "lw",
            start=value_text(constraint.value),
            low=value_text(constraint.min),
            high=value_text(constraint.max),
            held=constraint.vary is False,
            free=True if constraint.vary else None,
            raw="" if known else pattern,
        )


def constraint_text(
    start: float | None,
    low: float | None,
    high: float | None,
    vary: bool | None,
    unit: str = "",
) -> str:
    """What a constraint sets, ``start 20 Hz · range 5–80 Hz · held``; "" when nothing."""
    show = formatting.setting
    shown = f" {unit}" if unit else ""
    parts = []
    if start is not None:
        parts.append(f"start {show(start)}{shown}")
    if low is not None or high is not None:
        lower = show(low) if low is not None else "−∞"
        upper = show(high) if high is not None else "∞"
        parts.append(f"range {lower}–{upper}{shown}")
    if vary is not None:
        parts.append("held" if vary is False else "fitted")
    return " · ".join(parts)


def rule_text(pattern: str, constraint: ParameterConstraint) -> str:
    """A rule as recorded: ``*.F2.lw: start 20 · range 5–80 · held``."""
    c = constraint
    return f"{pattern}: {constraint_text(c.value, c.min, c.max, c.vary) or 'no change'}"


ENTER_NUMBERS = "Enter the start and bounds as numbers, or leave them empty."


def values_error(start: str, low: str, high: str, unit: str) -> str:
    """Why a typed start and range cannot be fitted, or ""; empty fields set nothing."""
    try:
        first, lower, upper = (parse_number(t) for t in (start, low, high))
    except ValueError:
        return ENTER_NUMBERS
    shown = f" {unit}" if unit else ""
    show = formatting.setting
    if lower is not None and upper is not None and lower >= upper:
        span = f"{show(lower)}–{show(upper)}{shown}"
        return f"The range {span} is empty. Make the lower bound smaller."
    if first is not None and (
        (lower is not None and first < lower) or (upper is not None and first > upper)
    ):
        bounds = f"{value_text(lower) or '−∞'}–{value_text(upper) or '∞'}{shown}"
        return f"Start {show(first)}{shown} is outside its range {bounds}. Change one."
    return ""


def rule_error(draft: _RuleDraft) -> str:
    """Why a rule's values cannot be fitted, or ""; the pattern itself is checked apart."""
    unit = KINDS.get(draft.kind, ("", ""))[1]
    return values_error(draft.start, draft.low, draft.high, unit)


class _RuleRow(QtWidgets.QWidget):
    """A rule: axis, kind, Hold and ×; then start and range; then its message."""

    def __init__(self, editor: ConstraintsEditor, axes: list[AxisInfo]) -> None:
        super().__init__()
        self.axis = _choice(
            "Rule axis",
            "The axis the rule applies on, or all of them",
            [("All", ALL_AXES), *((a.name, a.label) for a in axes)],
        )
        self.kind = _choice(
            "Rule parameter",
            "LW: linewidth, in Hz; η: the Lorentzian fraction of a pseudo-Voigt",
            [(name, key) for key, (name, _unit) in KINDS.items()],
        )
        self.pattern = text_label("", "")  # an imported pattern, shown as written
        self.hold = QtWidgets.QCheckBox("Hold")
        self.hold.setToolTip("Hold the parameter at its start instead of fitting it")
        self.remove = themes.link("×", "Remove this rule")
        self.remove.setAccessibleName("Remove rule")
        self.start = number_edit("Rule start")
        self.low = number_edit("Rule lower bound")
        self.high = number_edit("Rule upper bound")
        self.unit = text_label("", "muted")
        self.message = text_label("", "error", wrap=True)
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(3)
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(4)
        top.addWidget(self.axis, 1)
        top.addWidget(self.kind, 1)
        top.addWidget(self.pattern, 2)
        top.addWidget(self.hold)
        top.addWidget(self.remove)
        values = QtWidgets.QHBoxLayout()
        values.setSpacing(4)
        values.addWidget(text_label("start", "muted"))
        values.addWidget(self.start)
        values.addSpacing(themes.SPACE_XS)
        values.addWidget(text_label("range", "muted"))
        values.addWidget(self.low)
        values.addWidget(text_label("–", "muted"))
        values.addWidget(self.high)
        values.addWidget(self.unit)
        values.addStretch(1)
        grid.addLayout(top, 0, 0)
        grid.addLayout(values, 1, 0)
        grid.addWidget(self.message, 2, 0)
        self.axis.currentIndexChanged.connect(editor.commit)
        self.kind.currentIndexChanged.connect(editor.commit)
        self.hold.toggled.connect(editor.commit)
        for edit in (self.start, self.low, self.high):
            edit.editingFinished.connect(editor.commit)
        self.remove.clicked.connect(lambda: editor.remove_rule(self))

    def show_draft(self, draft: _RuleDraft) -> None:
        imported = bool(draft.raw)
        self.axis.setVisible(not imported)
        self.kind.setVisible(not imported)
        self.pattern.setVisible(imported)
        self.pattern.setText(draft.raw)
        if not imported:
            self.axis.setCurrentIndex(max(self.axis.findData(draft.axis), 0))
            self.kind.setCurrentIndex(max(self.kind.findData(draft.kind), 0))
        for edit, value in (
            (self.start, draft.start),
            (self.low, draft.low),
            (self.high, draft.high),
        ):
            if edit.text() != value:
                edit.setText(value)
        self.hold.setChecked(draft.held)
        self.unit.setText(KINDS.get(draft.kind, ("", ""))[1])

    def show_message(self, text: str, *, hint: bool) -> None:
        """The line under the rule, outlining its values for an error."""
        show_message(self.message, text, hint=hint)
        for edit in (self.start, self.low, self.high):
            mark_invalid(edit, bool(text) and not hint)


def _choice(name: str, tip: str, items: list[tuple[str, str]]) -> QtWidgets.QComboBox:
    combo = QtWidgets.QComboBox()
    combo.setAccessibleName(name)
    combo.setToolTip(tip)
    for text, data in items:
        combo.addItem(text, data)
    combo.setSizeAdjustPolicy(
        QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    combo.setMinimumContentsLength(4)
    return combo


def number_edit(name: str) -> QtWidgets.QLineEdit:
    """A right-aligned field for one short number, named for screen readers."""
    edit = QtWidgets.QLineEdit()
    edit.setAccessibleName(name)
    edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
    edit.setFixedWidth(edit.fontMetrics().horizontalAdvance("00.00") + 14)
    return edit


class ConstraintsEditor(QtWidgets.QWidget):
    """Edit position windows per axis and axis-wide rules; commit only valid ones."""

    changed = QtCore.Signal(object)  # ParameterConfig
    invalid_changed = QtCore.Signal(str)

    def __init__(self, axes: list[AxisInfo]) -> None:
        super().__init__()
        self._axes = axes
        self._parameters: ParameterConfig | None = None
        self._names: Collection[str] | None = None
        self._drafts: list[_RuleDraft] = []
        self._windows: dict[str, str] = {}
        self._pending: _RuleDraft | None = None  # added, nothing set yet
        self._updating = False
        self.invalid = ""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(text_label("Position window: how far a peak may move", "caption"))
        windows = QtWidgets.QGridLayout()
        windows.setHorizontalSpacing(4)
        windows.setVerticalSpacing(3)
        self.windows: dict[str, QtWidgets.QLineEdit] = {}
        for line, axis in enumerate(a for a in axes if a.label in WINDOW_AXES):
            edit = number_edit(f"{axis.name} position window (ppm)")
            edit.setToolTip(f"{axis.name}: each position stays within ± this of its start")
            edit.editingFinished.connect(self.commit)
            self.windows[axis.label] = edit
            windows.addWidget(text_label(axis.name, "muted"), line, 0)
            windows.addWidget(text_label("±", "muted"), line, 1)
            windows.addWidget(edit, line, 2)
            windows.addWidget(text_label("ppm", "muted"), line, 3)
        windows.setColumnStretch(4, 1)
        layout.addLayout(windows)
        self.window_message = text_label("", "error", wrap=True)
        self.window_message.hide()
        layout.addWidget(self.window_message)
        layout.addSpacing(themes.SPACE_XS)
        layout.addWidget(text_label("Rules, in order: a later rule overrides", "caption"))
        self._rules = QtWidgets.QVBoxLayout()
        self._rules.setSpacing(themes.SPACE_S)
        layout.addLayout(self._rules)
        self.rules: list[_RuleRow] = []
        self.add_rule = themes.link("+ Rule", "Add a rule on one kind of parameter")
        self.add_rule.setAccessibleName("Add constraint rule")
        self.add_rule.clicked.connect(self.start_rule)
        layout.addWidget(self.add_rule, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        self.peaks = text_label("", "caption", wrap=True)
        self.peaks.hide()
        layout.addWidget(self.peaks)

    def show_own(self, count: int) -> None:
        """How many peaks have limits of their own, which their Peak pages edit."""
        verb, whose = ("has", "its") if count == 1 else ("have", "their")
        self.peaks.setText(
            f"{formatting.count(count, 'peak')} also {verb} constraints of {whose} own "
            "(Peak page ▸ Parameters)."
            if count
            else ""
        )
        self.peaks.setVisible(bool(count))

    def show_parameters(
        self, parameters: ParameterConfig, names: Collection[str] | None = None
    ) -> None:
        """Show committed constraints; ``names`` are the prepared model's parameter names.

        Unchanged constraints keep drafts in progress; changed ones (Undo, a reopened
        analysis) replace them.
        """
        # One stand-in per parameter kind and axis (``_.F2.lw``): rules are axis-wide, so
        # matching them needs the kinds, not every peak's parameter.
        kinds = frozenset(f"_.{n.partition('.')[2]}" for n in names) if names is not None else None
        names_changed = kinds is not None and kinds != self._names
        if kinds is not None:
            self._names = kinds
        if parameters == self._parameters:
            if names_changed:
                self._update_errors()
            return
        self._parameters = parameters
        self._drafts = [_RuleDraft.of(p, c) for p, c in parameters.defaults.patterns.items()]
        self._pending = None
        self._windows = {
            label: value_text(parameters.position_windows.get(label)) for label in self.windows
        }
        default = parameters.position_window
        for edit in self.windows.values():
            edit.setPlaceholderText(value_text(default) if default is not None else "none")
        self._render()

    def discard_drafts(self) -> None:
        """Drop rules and windows being typed; the next ``show_parameters`` shows them anew."""
        self._parameters = None

    # ------------------------------------------------------------- edits
    def start_rule(self) -> None:
        """+ Rule: a draft that exists once it sets something."""
        if self._pending is None:
            self._pending = _RuleDraft(ALL_AXES, "lw")
            self._drafts.append(self._pending)
        self._render()
        self.rules[-1].axis.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

    def remove_rule(self, row: _RuleRow) -> None:
        """×: drop the rule and commit; the keyboard returns to + Rule."""
        draft = self._drafts.pop(self.rules.index(row))
        if draft is self._pending:
            self._pending = None
        self._render()
        self.commit()
        self.add_rule.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

    def commit(self) -> None:
        """Read every field; emit the constraints when all are valid."""
        if self._updating or self._parameters is None:
            return
        for row, draft in zip(self.rules, self._drafts, strict=True):
            if not draft.raw:
                draft.axis, draft.kind = row.axis.currentData(), row.kind.currentData()
                row.unit.setText(KINDS[draft.kind][1])
            draft.start, draft.low, draft.high = (
                e.text().strip() for e in (row.start, row.low, row.high)
            )
            draft.held = row.hold.isChecked()
        if self._pending is not None and self._sets_something(self._pending):
            self._pending = None
        self._windows = {label: edit.text().strip() for label, edit in self.windows.items()}
        parameters, errors = self._settings()
        self._update_errors()
        if not errors and parameters != self._parameters:
            self.changed.emit(parameters)

    @staticmethod
    def _sets_something(draft: _RuleDraft) -> bool:
        return bool(draft.start or draft.low or draft.high or draft.held)

    def _settings(self) -> tuple[ParameterConfig, dict[int | str, str]]:
        """The constraints the drafts describe, and each invalid entry's reason."""
        assert self._parameters is not None
        errors: dict[int | str, str] = {}
        windows: dict[str, float | None] = {}
        for label, text in self._windows.items():
            value = window_value(text)
            if value is INVALID:
                errors[WINDOW] = "Enter each position window as a positive number of ppm."
            windows[label] = value if isinstance(value, float) else None
        patterns: dict[str, ParameterConstraint] = {}
        for index, draft in enumerate(self._drafts):
            if draft is self._pending:
                errors[index] = NEW_RULE
                continue
            error = self._pattern_error(draft) or rule_error(draft)
            if not error and draft.pattern in patterns:
                error = "Another rule already sets this parameter. Remove one of them."
            if error:
                errors[index] = error
                continue
            start, low, high = (parse_number(t) for t in (draft.start, draft.low, draft.high))
            vary = False if draft.held else draft.free
            patterns[draft.pattern] = ParameterConstraint(value=start, min=low, max=high, vary=vary)
        position_windows = self._parameters.position_windows.model_copy(update=windows)
        parameters = self._parameters.model_copy(
            update={
                "position_windows": position_windows,
                "defaults": self._parameters.defaults.model_copy(update={"patterns": patterns}),
            }
        )
        return parameters, errors

    def _pattern_error(self, draft: _RuleDraft) -> str:
        if not desktop_rule(draft.pattern):
            return (
                f"“{draft.raw}”: the desktop fit applies no rules for named peaks. "
                "Remove it, or fit with the CLI."
            )
        names = self._names
        if names is not None and not any(matches_pattern(n, draft.pattern) for n in names):
            if draft.raw:
                what = f"“{draft.raw}”"
            else:
                where = "any axis" if draft.axis == ALL_AXES else self._axis_name(draft.axis)
                what = f"{KINDS[draft.kind][0]} on {where}"
            return (
                f"{what} matches no parameter of the current model. "
                "Remove the rule or change the lineshape."
            )
        return ""

    def _axis_name(self, label: str) -> str:
        return next((a.name for a in self._axes if a.label == label), label)

    # ------------------------------------------------------------- display
    def _render(self) -> None:
        self._updating = True
        try:
            while len(self.rules) > len(self._drafts):
                row = self.rules.pop()
                self._rules.removeWidget(row)
                row.deleteLater()
            while len(self.rules) < len(self._drafts):
                row = _RuleRow(self, self._axes)
                self.rules.append(row)
                self._rules.addWidget(row)
                row.show()  # at once, not on the layout's queued show
            for row, draft in zip(self.rules, self._drafts, strict=True):
                row.show_draft(draft)
            for label, edit in self.windows.items():
                if edit.text() != self._windows.get(label, ""):
                    edit.setText(self._windows.get(label, ""))
        finally:
            self._updating = False
        self._update_errors()

    def _update_errors(self) -> None:
        if self._parameters is None:
            return
        _settings, errors = self._settings()
        window = errors.get(WINDOW, "")
        show_message(self.window_message, window, hint=False)
        for label, edit in self.windows.items():
            mark_invalid(edit, window_value(self._windows.get(label, "")) is INVALID)
        for index, row in enumerate(self.rules):
            draft = self._drafts[index]
            row.show_message(errors.get(index, ""), hint=draft is self._pending)
        first = window or next((errors[i] for i in range(len(self._drafts)) if i in errors), "")
        first = f"Constraints: {first}" if first else ""
        if first != self.invalid:
            self.invalid = first
            self.invalid_changed.emit(first)
