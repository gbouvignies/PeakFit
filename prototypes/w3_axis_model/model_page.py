# ruff: noqa  -- throwaway prototype (W3 step 1, #215); delete with the folder
"""THROWAWAY PROTOTYPE (W3 step 1, issue #215): two layouts of the per-axis model.

Not product code. Nothing imports this module; it exists only to capture the two
layout alternatives in the real window, and is deleted once the user has chosen.

- ``TableModel`` (A): one row per spectral axis; columns for family, the doublet
  (J start, range, fitted or fixed) and phase.
- ``StackedModel`` (B): one group per spectral axis, the same controls stacked.

``ModelPage`` places either one where the Model page would put it, above the
other settings (clustering, noise, passes, planes, optimizer limits), so density
and the fold can be judged. Values below the per-axis model come from the real
preparation of the bundled dataset; the per-axis settings are the scenarios'.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

from PySide6 import QtCore, QtWidgets

from peakfit.gui import formatting
from peakfit.gui import theme as themes
from peakfit.gui.editor.labels import LABEL_WIDTH, disclosure, section, text_label

FAMILIES = {
    "auto": "Automatic",
    "gaussian": "Gaussian",
    "lorentzian": "Lorentzian",
    "pvoigt": "Pseudo-Voigt",
    "sp1": "SP1",
    "sp2": "SP2",
    "no_apod": "No apodization",
}
PHASED = {"sp1", "sp2", "no_apod"}


@dataclass
class J:
    """A doublet as the settings hold it: start, range, fitted or held."""

    j: float
    low: float
    high: float
    vary: bool = True

    def error(self) -> str:
        """Why the range is invalid, as the user would read it; empty when valid."""
        if self.low >= self.high:
            return f"the range {formatting.hz(self.low)}–{formatting.hz(self.high)} Hz is empty"
        if not self.low <= self.j <= self.high:
            return (
                f"J start {formatting.hz(self.j)} Hz is outside its range "
                f"{formatting.hz(self.low)}–{formatting.hz(self.high)} Hz"
            )
        return ""


@dataclass
class Axis:
    """One spectral axis of a scenario: its name parts and its model settings."""

    nucleus: str
    label: str
    family: str = "auto"
    resolved: str = "sp2"
    doublet: J | None = None
    phase: bool = False

    @property
    def name(self) -> str:
        return formatting.axis_name(SimpleNamespace(nucleus=self.nucleus, label=self.label))

    def family_text(self) -> str:
        """The family control's text: Automatic shows what it resolved to."""
        if self.family == "auto":
            return f"Auto · {FAMILIES[self.resolved]}"
        return FAMILIES[self.family]

    def source(self) -> str:
        return "from apodization" if self.family == "auto" else "explicit"


@dataclass
class Scenario:
    key: str
    title: str
    axes: list[Axis]
    note: str = ""
    planes: list[int] = field(default_factory=list)


def _n(label: str = "F2", **kw) -> Axis:
    return Axis("15N", label, **kw)


def _h(label: str = "F3", **kw) -> Axis:
    return Axis("1H", label, **kw)


N_DOUBLET = J(92.0, 46.0, 184.0)
SCENARIOS = (
    Scenario("singlets", "Two singlets, automatic family", [_n(), _h()]),
    Scenario(
        "doublet-phase",
        "¹⁵N doublet (J 92 Hz, range 46–184), phase fitted on ¹H",
        [_n(doublet=N_DOUBLET), _h(phase=True)],
    ),
    Scenario(
        "fixed-j",
        "Fixed-J ¹⁵N doublet; ¹H set to Lorentzian",
        [_n(doublet=J(92.0, 46.0, 184.0, vary=False)), _h(family="lorentzian")],
    ),
    Scenario(
        "homonuclear",
        "Homonuclear: two ¹H axes, a doublet on ¹H F3",
        [Axis("1H", "F2"), Axis("1H", "F3", doublet=J(7.0, 3.5, 14.0))],
        note="Model block mocked over the bundled ¹⁵N–¹H spectrum",
    ),
    Scenario(
        "three-axes",
        "Three spectral axes: ¹³C doublet, ¹⁵N singlet, ¹H phase",
        [
            Axis("13C", "F2", doublet=J(35.0, 17.5, 70.0)),
            _n("F3"),
            _h("F4", phase=True),
        ],
        note="Model block mocked over the bundled ¹⁵N–¹H spectrum",
    ),
    Scenario(
        "invalid-j",
        "Error: J start 92 Hz outside the range 100–184 Hz",
        [_n(doublet=J(92.0, 100.0, 184.0)), _h(phase=True)],
    ),
)


# ------------------------------------------------------------------ controls
def family_combo(axis: Axis) -> QtWidgets.QComboBox:
    combo = QtWidgets.QComboBox()
    combo.setAccessibleName(f"{axis.name} family")
    for key, text in FAMILIES.items():
        combo.addItem(f"Auto · {FAMILIES[axis.resolved]}" if key == "auto" else text, key)
    combo.setCurrentIndex(combo.findData(axis.family))
    combo.setSizeAdjustPolicy(
        QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    combo.setMinimumContentsLength(6)
    combo.setToolTip(f"{axis.name}: {axis.family_text()} · {axis.source()}")
    return combo


def hz_edit(
    value: float | None, name: str, *, error: bool, theme: themes.Theme
) -> QtWidgets.QLineEdit:
    edit = QtWidgets.QLineEdit("" if value is None else formatting.hz(value))
    edit.setAccessibleName(name)
    edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
    width = edit.fontMetrics().horizontalAdvance("000.0") + 16
    edit.setFixedWidth(width)
    if error:
        edit.setStyleSheet(f"QLineEdit {{ border: 1px solid {theme.error}; }}")
    return edit


def check(text: str, name: str, checked: bool) -> QtWidgets.QCheckBox:
    box = QtWidgets.QCheckBox(text)
    box.setAccessibleName(name)
    box.setChecked(checked)
    return box


def error_label(text: str) -> QtWidgets.QLabel:
    label = text_label(f"▲ {text}", "error", wrap=True)
    return label


# ------------------------------------------------------------------ (A) table
class TableModel(QtWidgets.QWidget):
    """(A) One row per axis: Axis · Family · J · Range · Fit J · Phase."""

    kind = "A"

    def __init__(self, axes: list[Axis], theme: themes.Theme) -> None:
        super().__init__()
        self.setObjectName("axisModel")
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        outer.addLayout(grid)
        headers = ("Axis", "Family", "J (Hz)", "Range (Hz)", "", "", "Fit J", "Phase")
        for column, text in enumerate(headers):
            if text:
                span = 3 if text.startswith("Range") else 1
                header = text_label(text, "caption")
                grid.addWidget(header, 0, column, 1, span)
        self.first: QtWidgets.QWidget | None = None
        errors = []
        for row, axis in enumerate(axes, start=1):
            name = text_label(axis.name, "")
            grid.addWidget(name, row, 0)
            combo = family_combo(axis)
            self.first = self.first or combo
            grid.addWidget(combo, row, 1)
            d = axis.doublet
            bad = bool(d and d.error())
            j = hz_edit(d.j if d else None, f"{axis.name} J start", error=bad, theme=theme)
            j.setPlaceholderText("none")
            j.setToolTip("Type a J to fit a doublet on this axis; clear it for a singlet")
            grid.addWidget(j, row, 2)
            if d:
                low = hz_edit(d.low, f"{axis.name} J lower bound", error=bad, theme=theme)
                high = hz_edit(d.high, f"{axis.name} J upper bound", error=bad, theme=theme)
                dash = text_label("–", "muted")
                for widget in (low, high):
                    widget.setEnabled(d.vary)
                grid.addWidget(low, row, 3)
                grid.addWidget(dash, row, 4)
                grid.addWidget(high, row, 5)
                fit = check("", f"{axis.name} fit J", d.vary)
                grid.addWidget(fit, row, 6, QtCore.Qt.AlignmentFlag.AlignHCenter)
                if bad:
                    errors.append(f"{axis.name}: {d.error()}.")
            phase = check("", f"{axis.name} fit phase", axis.phase)
            phase.setEnabled((axis.resolved if axis.family == "auto" else axis.family) in PHASED)
            grid.addWidget(phase, row, 7, QtCore.Qt.AlignmentFlag.AlignHCenter)
        grid.setColumnStretch(1, 1)
        # Where each family came from, once for the table.
        sources = sorted({f"{a.family_text()} {a.source()}" for a in axes})
        if len(sources) == 1 and len(axes) > 1:
            caption = f"All axes: {sources[0]}"
        else:
            caption = " · ".join(f"{a.name} {a.source()}" for a in axes)
        outer.addWidget(text_label(caption, "caption", wrap=True))
        for text in errors:
            outer.addWidget(error_label(f"{text} Change the start or widen the range."))


# ------------------------------------------------------------------ (A′) compact table
class CompactTableModel(QtWidgets.QWidget):
    """(A′) One row per axis: Axis · Family · J · Fit J · Phase; a doublet's range below it.

    The full table needs ~430 px; this keeps one row per axis within the 288 px the
    inspector has at 1280 px by moving the J range to a line under its row.
    """

    kind = "A′"

    def __init__(self, axes: list[Axis], theme: themes.Theme) -> None:
        super().__init__()
        self.setObjectName("axisModel")
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(4)
        outer.addLayout(grid)
        for column, text in enumerate(("Axis", "Family", "J (Hz)", "Fit J", "Phase")):
            header = text_label(text, "caption")
            align = (
                QtCore.Qt.AlignmentFlag.AlignHCenter if column >= 3 else QtCore.Qt.AlignmentFlag(0)
            )
            grid.addWidget(header, 0, column, align)
        self.first: QtWidgets.QWidget | None = None
        line = 1
        for axis in axes:
            grid.addWidget(text_label(axis.name, ""), line, 0)
            combo = family_combo(axis)
            self.first = self.first or combo
            grid.addWidget(combo, line, 1)
            d = axis.doublet
            bad = bool(d and d.error())
            j = hz_edit(d.j if d else None, f"{axis.name} J start", error=bad, theme=theme)
            j.setPlaceholderText("none")
            j.setToolTip("Type a J to fit a doublet on this axis; clear it for a singlet")
            grid.addWidget(j, line, 2)
            centre = QtCore.Qt.AlignmentFlag.AlignHCenter
            if d:
                grid.addWidget(check("", f"{axis.name} fit J", d.vary), line, 3, centre)
            phase = check("", f"{axis.name} fit phase", axis.phase)
            phase.setEnabled((axis.resolved if axis.family == "auto" else axis.family) in PHASED)
            grid.addWidget(phase, line, 4, centre)
            line += 1
            if d:
                bounds = QtWidgets.QHBoxLayout()
                bounds.setSpacing(4)
                bounds.addWidget(text_label("J range", "muted"))
                low = hz_edit(d.low, f"{axis.name} J lower bound", error=bad, theme=theme)
                high = hz_edit(d.high, f"{axis.name} J upper bound", error=bad, theme=theme)
                for widget in (low, high):
                    widget.setEnabled(d.vary)
                bounds.addWidget(low)
                bounds.addWidget(text_label("–", "muted"))
                bounds.addWidget(high)
                bounds.addWidget(text_label("Hz", "muted"))
                bounds.addStretch(1)
                grid.addLayout(bounds, line, 1, 1, 4)
                line += 1
                if bad:
                    text = f"▲ {d.error()[0].upper()}{d.error()[1:]}. Change the start or widen the range."
                    grid.addWidget(text_label(text, "error", wrap=True), line, 1, 1, 4)
                    line += 1
                elif not d.vary:
                    grid.addWidget(
                        text_label(f"J held at {formatting.hz(d.j)} Hz", "caption"), line, 1, 1, 4
                    )
                    line += 1
        grid.setColumnStretch(1, 1)
        sources = sorted({f"{a.family_text()} {a.source()}" for a in axes})
        if len(sources) == 1 and len(axes) > 1:
            caption = f"All axes: {sources[0]}"
        else:
            caption = " · ".join(f"{a.name} {a.source()}" for a in axes)
        outer.addWidget(text_label(caption, "caption", wrap=True))


# ------------------------------------------------------------------ (B) stacked
class StackedModel(QtWidgets.QWidget):
    """(B) One group per axis: its name, then Family, Doublet, Range and Phase rows."""

    kind = "B"

    def __init__(self, axes: list[Axis], theme: themes.Theme) -> None:
        super().__init__()
        self.setObjectName("axisModel")
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(themes.SPACE_S)
        self.first: QtWidgets.QWidget | None = None
        for axis in axes:
            group = QtWidgets.QWidget()
            grid = QtWidgets.QGridLayout(group)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(themes.SPACE_S)
            grid.setVerticalSpacing(4)
            grid.setColumnMinimumWidth(0, LABEL_WIDTH)
            grid.setColumnStretch(1, 1)
            title = text_label(axis.name, "value")
            grid.addWidget(title, 0, 0, 1, 2)
            line = 1

            def row(label: str, content, *, caption: str = "", style: str = "caption") -> None:
                nonlocal line
                grid.addWidget(text_label(label, "muted"), line, 0)
                if isinstance(content, QtWidgets.QLayout):
                    grid.addLayout(content, line, 1)
                else:
                    grid.addWidget(content, line, 1)
                line += 1
                if caption:
                    grid.addWidget(text_label(caption, style, wrap=True), line, 1)
                    line += 1

            combo = family_combo(axis)
            self.first = self.first or combo
            row("Family", combo, caption=axis.source())
            d = axis.doublet
            bad = bool(d and d.error())
            doublet = QtWidgets.QHBoxLayout()
            doublet.setSpacing(6)
            on = check("J doublet" if not d else "", f"{axis.name} doublet", d is not None)
            doublet.addWidget(on)
            if d:
                doublet.addWidget(hz_edit(d.j, f"{axis.name} J start", error=bad, theme=theme))
                doublet.addWidget(text_label("Hz", "muted"))
                doublet.addSpacing(6)
                doublet.addWidget(check("Fit J", f"{axis.name} fit J", d.vary))
            doublet.addStretch(1)
            row("Doublet", doublet)
            if d:
                bounds = QtWidgets.QHBoxLayout()
                bounds.setSpacing(6)
                low = hz_edit(d.low, f"{axis.name} J lower bound", error=bad, theme=theme)
                high = hz_edit(d.high, f"{axis.name} J upper bound", error=bad, theme=theme)
                for widget in (low, high):
                    widget.setEnabled(d.vary)
                bounds.addWidget(low)
                bounds.addWidget(text_label("–", "muted"))
                bounds.addWidget(high)
                bounds.addWidget(text_label("Hz", "muted"))
                bounds.addStretch(1)
                row(
                    "J range",
                    bounds,
                    caption=(
                        f"▲ {d.error()[0].upper()}{d.error()[1:]}. Change the start or "
                        "widen the range."
                        if bad
                        else ("" if d.vary else f"J held at {formatting.hz(d.j)} Hz")
                    ),
                    style="error" if bad else "caption",
                )
            phase = check("Fit phase", f"{axis.name} fit phase", axis.phase)
            phase.setEnabled((axis.resolved if axis.family == "auto" else axis.family) in PHASED)
            row("Phase", phase)
            outer.addWidget(group)


LAYOUTS = {"table": TableModel, "compact": CompactTableModel, "stacked": StackedModel}


# ------------------------------------------------------------------ the page
class ModelPage(QtWidgets.QWidget):
    """Where the per-axis model would sit: the Model page, before the first fit."""

    def __init__(
        self,
        layout: str,
        scenario: Scenario,
        theme: themes.Theme,
        *,
        noise: float,
        level: float,
        factor: float,
        clusters: int,
        samples: int,
        planes: int,
        passes: int,
    ) -> None:
        super().__init__()
        self.setObjectName("prototypeModelPage")
        page = QtWidgets.QVBoxLayout(self)
        page.setContentsMargins(0, 0, 0, 0)
        page.setSpacing(themes.SPACE_S)
        # The two-way switch of step 2: Run is unavailable before the first fit.
        switch = QtWidgets.QHBoxLayout()
        switch.setSpacing(0)
        run = themes.segment("Run", "first")
        model = themes.segment("Model", "last")
        for button in (run, model):
            button.setCheckable(True)
            switch.addWidget(button)
        model.setChecked(True)
        run.setEnabled(False)
        run.setToolTip("Available after the first fit")
        switch.addStretch(1)
        page.addLayout(switch)
        section(page, "Lineshape model")
        self.axes = LAYOUTS[layout](scenario.axes, theme)
        page.addWidget(self.axes)
        section(page, "Clustering")
        form = QtWidgets.QGridLayout()
        form.setHorizontalSpacing(themes.SPACE_S)
        form.setVerticalSpacing(6)
        form.setColumnMinimumWidth(0, LABEL_WIDTH)
        form.setColumnStretch(1, 1)
        threshold = QtWidgets.QDoubleSpinBox()
        threshold.setRange(0.1, 1000)
        threshold.setDecimals(2)
        threshold.setValue(factor)
        threshold.setSuffix(" × noise")
        threshold.setAccessibleName("Clustering threshold")
        threshold.setLocale(QtCore.QLocale.c())
        self.threshold = threshold
        rows: list[tuple[str, QtWidgets.QWidget, str]] = [
            ("Threshold", threshold, f"= {formatting.scaled(level)}"),
            (
                "Statistical noise",
                text_label(formatting.scaled(noise)),
                "estimated from all planes",
            ),
            (
                "Prepared",
                text_label(
                    f"{formatting.count(clusters, 'cluster')} · "
                    f"{formatting.count(samples, 'sample')} per plane",
                    wrap=True,
                ),
                "",
            ),
        ]
        self._rows(form, rows)
        page.addLayout(form)
        section(page, "Fit")
        fit = QtWidgets.QGridLayout()
        fit.setHorizontalSpacing(themes.SPACE_S)
        fit.setVerticalSpacing(6)
        fit.setColumnMinimumWidth(0, LABEL_WIDTH)
        fit.setColumnStretch(1, 1)
        passes_box = QtWidgets.QSpinBox()
        passes_box.setRange(1, 20)
        passes_box.setValue(passes)
        passes_box.setAccessibleName("Optimizer passes")
        planes_widget = QtWidgets.QWidget()
        planes_row = QtWidgets.QVBoxLayout(planes_widget)
        planes_row.setContentsMargins(0, 0, 0, 0)
        planes_row.setSpacing(2)
        planes_row.addWidget(text_label(f"All {planes} fitted", wrap=True))
        planes_row.addWidget(themes.link("Exclude planes…"), 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        self.planes = planes_widget
        self._rows(
            fit,
            [
                ("Passes", passes_box, "cross-talk updated between passes"),
                ("Planes", planes_widget, ""),
            ],
        )
        page.addLayout(fit)
        limits = disclosure("Optimizer limits")
        page.addWidget(limits, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        if scenario.note:
            page.addSpacing(themes.SPACE_S)
            page.addWidget(text_label(f"Prototype: {scenario.note}", "caption", wrap=True))

    @staticmethod
    def _rows(grid: QtWidgets.QGridLayout, rows) -> None:
        line = 0
        for name, widget, caption in rows:
            grid.addWidget(text_label(name, "muted"), line, 0, QtCore.Qt.AlignmentFlag.AlignTop)
            grid.addWidget(widget, line, 1)
            line += 1
            if caption:
                grid.addWidget(text_label(caption, "caption", wrap=True), line, 1)
                line += 1
