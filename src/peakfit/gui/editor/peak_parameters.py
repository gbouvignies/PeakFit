"""The Peak page's Parameters: the selected peak's starts, bounds, windows and Hold.

Each of the peak's parameters shows what the next fit submits, as the current
preparation resolves it from the Model settings. A value typed here is the peak's
own limit, kept on its working peak (by stable ID, so it follows renames and is
never shared with a peak of the same assignment), and overrides the settings for
this peak only, as ``[parameters.peaks."<name>"]`` does for ``peakfit fit``.

Edits are drafts until valid: an invalid one is explained under its parameter,
outlines its fields and blocks Fit; a valid one is committed as one undoable edit.
"""

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtWidgets

from peakfit.gui import formatting
from peakfit.gui import theme as themes
from peakfit.gui.editor.axis_table import DIFFERS
from peakfit.gui.editor.constraints import (
    INVALID,
    constraint_text,
    number_edit,
    parse_number,
    value_text,
    values_error,
    window_value,
)
from peakfit.gui.editor.labels import mark_invalid, show_message, text_label
from peakfit.gui.peaks import ParameterLimit

if TYPE_CHECKING:
    from peakfit.gui.peaks import PeakLimits, WorkingPeak

NOTE = "Empty fields follow the Model settings; typed values apply to this peak only."
ENTER_WINDOW = "Enter the position window as a positive number of ppm, or leave it empty."
WINDOW_TIP = "How far this peak's position may move from where it is"


@dataclass(frozen=True)
class ParameterRow:
    """One of the peak's parameters as the Model settings alone give it.

    These are the greyed values and Hold's default; the peak's own limits replace
    them field by field. ``window`` is a position's range as ± ppm around the peak
    (None when the settings' range is not symmetric); ``problem`` is why the peak's
    own limit cannot be fitted as given, or "". A limit on a parameter the model lacks
    has non-finite values.
    """

    key: str  # ``F3.lw``
    title: str  # ``LW ¹H F3``
    unit: str
    start: float
    lower: float
    upper: float
    vary: bool
    window: float | None = None
    problem: str = ""

    @property
    def position(self) -> bool:
        """Whether the parameter is a position, limited by a window around the peak."""
        return self.key.endswith(".cs")


def limit_text(limit: ParameterLimit, unit: str = "") -> str:
    """A peak's own limit as recorded: ``start 20 · range 5–80 · held``, ``± 0.05 ppm``."""
    parts = [constraint_text(limit.start, limit.lower, limit.upper, limit.vary, unit)]
    if limit.window is not None:
        parts.insert(0, f"± {formatting.setting(limit.window)} ppm")
    return " · ".join(part for part in parts if part)


@dataclass
class _Draft:
    """One parameter's own limit as shown; texts stay as typed until they are valid."""

    start: str = ""
    low: str = ""
    high: str = ""
    window: str = ""
    vary: bool | None = None

    @classmethod
    def of(cls, limit: ParameterLimit) -> _Draft:
        return cls(
            value_text(limit.start),
            value_text(limit.lower),
            value_text(limit.upper),
            value_text(limit.window),
            limit.vary,
        )

    def error(self, unit: str) -> str:
        """Why the typed values cannot be fitted, or ""."""
        if window_value(self.window) is INVALID:
            return ENTER_WINDOW
        return values_error(self.start, self.low, self.high, unit)

    def limit(self) -> ParameterLimit:
        """The limit the draft gives; only for a draft without an error."""
        start, low, high = (parse_number(t) for t in (self.start, self.low, self.high))
        window = window_value(self.window)
        return ParameterLimit(
            start, low, high, self.vary, window if isinstance(window, float) else None
        )


class _LimitRow(QtWidgets.QWidget):
    """A parameter: its name, Hold and ×; then its start and range, or its window."""

    def __init__(self, editor: PeakParameters, row: ParameterRow) -> None:  # noqa: PLR0915
        super().__init__()
        self.key = row.key
        self.title = text_label(row.title, "")
        self.hold = QtWidgets.QCheckBox("Hold")
        self.hold.setAccessibleName(f"Hold {row.title}")
        self.hold.setToolTip("Hold this peak's parameter at its start instead of fitting it")
        self.reset = themes.link("×", "Reset to the Model settings")
        self.reset.setAccessibleName(f"Reset {row.title}")
        # Hidden, it keeps its room, so Hold lines up on every parameter.
        policy = self.reset.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.reset.setSizePolicy(policy)
        self.start = number_edit(f"{row.title} start")
        self.low = number_edit(f"{row.title} lower bound")
        self.high = number_edit(f"{row.title} upper bound")
        self.window_edit = number_edit(f"{row.title} position window (ppm)")
        self.window_edit.setToolTip(WINDOW_TIP)
        self.at = text_label("", "muted")
        self.unit = text_label(row.unit, "muted")
        self.message = text_label("", "error", wrap=True)
        self.message.hide()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(4)
        top.addWidget(self.title, 1)
        top.addWidget(self.hold)
        top.addWidget(self.reset)
        values = QtWidgets.QHBoxLayout()
        values.setSpacing(4)
        if row.position:
            # A position starts where the peak is: moving it is the position editor's.
            values.addWidget(text_label("at", "muted"))
            values.addWidget(self.at)
            values.addSpacing(themes.SPACE_XS)
            values.addWidget(text_label("±", "muted"))
            values.addWidget(self.window_edit)
            values.addWidget(text_label("ppm", "muted"))
            fields = (self.window_edit,)
        else:
            values.addWidget(text_label("start", "muted"))
            values.addWidget(self.start)
            values.addSpacing(themes.SPACE_XS)
            values.addWidget(text_label("range", "muted"))
            values.addWidget(self.low)
            values.addWidget(text_label("–", "muted"))
            values.addWidget(self.high)
            values.addWidget(self.unit)
            fields = (self.start, self.low, self.high)
        values.addStretch(1)
        layout.addLayout(top)
        layout.addLayout(values)
        layout.addWidget(self.message)
        self.fields = fields
        for edit in fields:
            edit.editingFinished.connect(editor.commit)
        self.hold.toggled.connect(lambda held: editor.hold(self, held))
        self.reset.clicked.connect(lambda: editor.reset(self))
        self.row = row

    def show_row(self, row: ParameterRow, draft: _Draft, *, differs: str, stale: bool) -> None:
        """Placeholders from the preparation, typed values from the draft.

        A ``stale`` row has no current preparation: no values, only what was typed.
        """
        self.row = row
        self.title.setText(f"{row.title} {DIFFERS}" if differs else row.title)
        self.title.setToolTip(differs)
        known = math.isfinite(row.start)
        self.at.setText(formatting.ppm(row.start) if known else formatting.DASH)
        for edit, value, text in (
            (self.start, row.start, draft.start),
            (self.low, row.lower, draft.low),
            (self.high, row.upper, draft.high),
        ):
            edit.setPlaceholderText(value_text(value) if math.isfinite(value) else "")
            if edit.text() != text:
                edit.setText(text)
        # The field is narrow: the hint is short, the exact value in the tooltip. A
        # rule's absolute range is no ± window: say so, and give it in the tooltip.
        given = row.window is not None and math.isfinite(row.window) and not stale
        window = formatting.scaled(row.window, 2) if given and row.window else "range"
        self.window_edit.setPlaceholderText("" if stale else window)
        if given:
            tip = f"{WINDOW_TIP}; the Model settings give ± {value_text(row.window)} ppm"
        elif math.isfinite(row.lower) and not stale:
            tip = (
                f"The Model settings bound it to {formatting.ppm(row.lower)}–"
                f"{formatting.ppm(row.upper)} ppm; a window here replaces that"
            )
        else:
            tip = WINDOW_TIP
        self.window_edit.setToolTip(tip)
        if self.window_edit.text() != draft.window:
            self.window_edit.setText(draft.window)
        held = (draft.vary is False) if draft.vary is not None else not row.vary
        with QtCore.QSignalBlocker(self.hold):
            self.hold.setChecked(held)
        own = draft != _Draft()
        self.reset.setVisible(own)

    def show_message(self, text: str, *, invalid: bool) -> None:
        """The line under the parameter, outlining its fields for a typed error."""
        show_message(self.message, text, hint=False)
        for edit in self.fields:
            mark_invalid(edit, invalid)


class PeakParameters(QtWidgets.QWidget):
    """Edit one peak's own limits; commit only valid ones."""

    changed = QtCore.Signal(str, object)  # working ID, PeakLimits
    invalid_changed = QtCore.Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._peak: WorkingPeak | None = None
        self._rows: list[ParameterRow] = []
        self._drafts: dict[str, _Draft] = {}
        self._recorded: tuple[str, PeakLimits] | None = None  # the run's name and limits
        self._updating = False
        self._stale = False
        self.invalid = ""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(themes.SPACE_S)
        self.note = text_label("", "caption", wrap=True)
        layout.addWidget(self.note)
        self._list = QtWidgets.QVBoxLayout()
        self._list.setSpacing(themes.SPACE_S)
        layout.addLayout(self._list)
        self.rows: list[_LimitRow] = []

    @property
    def own(self) -> int:
        """How many of the shown peak's parameters have limits of its own."""
        return len(self._peak.limits) if self._peak is not None else 0

    @property
    def differs(self) -> bool:
        """Whether any of the peak's own limits differs from the active run's."""
        return any(self._differs(row.key) for row in self._rows)

    def show_peak(
        self,
        peak: WorkingPeak | None,
        rows: list[ParameterRow] | None,
        note: str,
        recorded: tuple[str, PeakLimits] | None = None,
    ) -> None:
        """Show ``peak``'s parameters; ``recorded`` is the active run's (name, own limits).

        Without ``rows`` (a preparation under way or failed) the same peak keeps its
        rows and drafts; another peak shows only the note. A changed peak or limits
        (a commit, Undo) replace the drafts.
        """
        before = self._peak
        same = peak is not None and before is not None and peak.id == before.id
        if not same or before is None or peak is None or peak.limits != before.limits:
            self._drafts = {k: _Draft.of(limit) for k, limit in peak.limits} if peak else {}
        self._peak = peak
        self._recorded = recorded
        self.note.setText(note)
        if rows is not None or not same:
            self._rows = rows or []
        # Kept rows show no values until the preparation is current again, never stale ones.
        self._stale = rows is None
        self._render()

    def discard_drafts(self) -> None:
        """Drop values being typed: Undo, Redo and a new selection show committed limits."""
        peak = self._peak
        self._drafts = {k: _Draft.of(limit) for k, limit in peak.limits} if peak else {}
        self._render()

    # ------------------------------------------------------------- edits
    def commit(self) -> None:
        """Read every field; emit the peak's limits when all are valid."""
        if self._updating or self._peak is None:
            return
        for row in self.rows:
            draft = self._drafts.setdefault(row.key, _Draft())
            draft.start, draft.low, draft.high, draft.window = (
                e.text().strip() for e in (row.start, row.low, row.high, row.window_edit)
            )
        self._emit()

    def hold(self, row: _LimitRow, held: bool) -> None:
        """Hold: this peak's parameter is held (or fitted against a rule that holds it).

        Back to what the Model settings give, the peak has no vary of its own, so a
        rule changed later still applies to it.
        """
        given = not row.row.vary
        self._drafts.setdefault(row.key, _Draft()).vary = None if held == given else not held
        self.commit()

    def reset(self, row: _LimitRow) -> None:
        """×: the parameter follows the Model settings again; the keyboard moves on."""
        self._drafts[row.key] = _Draft()
        self._render()
        self._emit()
        row.hold.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

    def _emit(self) -> None:
        peak = self._peak
        assert peak is not None
        errors = self._update_errors()
        if errors:
            return
        updated = peak
        for key, draft in self._drafts.items():
            updated = updated.with_limit(key, draft.limit())
        if updated.limits != peak.limits:
            self.changed.emit(peak.id, updated.limits)

    # ------------------------------------------------------------- display
    def _differs(self, key: str) -> str:
        """What the run used for ``key`` when this peak's own limit differs, or ""."""
        if self._recorded is None or self._peak is None:
            return ""
        run, limits = self._recorded
        used = dict(limits).get(key, ParameterLimit())
        if self._peak.limit(key) == used:
            return ""
        unit = next((r.unit for r in self._rows if r.key == key), "")
        return f"Run {run} used " + (limit_text(used, unit) or "the Model settings")

    def _render(self) -> None:
        self._updating = True
        try:
            keys = [row.key for row in self._rows]
            if [row.key for row in self.rows] != keys:
                for widget in self.rows:
                    self._list.removeWidget(widget)
                    widget.deleteLater()
                self.rows = []
                for row in self._rows:
                    widget = _LimitRow(self, row)
                    self.rows.append(widget)
                    self._list.addWidget(widget)
                    widget.show()  # at once, not on the layout's queued show
            for widget, row in zip(self.rows, self._rows, strict=True):
                draft = self._drafts.get(row.key, _Draft())
                shown = replace(row, start=math.nan, lower=math.nan, upper=math.nan, window=None)
                widget.show_row(
                    shown if self._stale else row,
                    draft,
                    differs=self._differs(row.key),
                    stale=self._stale,
                )
        finally:
            self._updating = False
        self._update_errors()

    def _update_errors(self) -> dict[str, str]:
        """Explain each typed error, else the preparation's problem, under its parameter."""
        errors = {}
        for widget, row in zip(self.rows, self._rows, strict=True):
            draft = self._drafts.get(row.key, _Draft())
            error = draft.error(row.unit)
            if error:
                errors[row.key] = f"{row.title}: {error}"
            committed = self._peak is not None and _Draft.of(self._peak.limit(row.key)) == draft
            problem = row.problem if committed else ""
            widget.show_message(error or problem, invalid=bool(error))
        first = next(iter(errors.values()), "")
        invalid = f"{self._peak.name}: {first}" if first and self._peak is not None else ""
        if invalid != self.invalid:
            self.invalid = invalid
            self.invalid_changed.emit(invalid)
        return errors
