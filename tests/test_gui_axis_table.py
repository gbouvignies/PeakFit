"""The Model page's per-axis table: adding, editing and validating doublets per axis."""

import pytest
from PySide6 import QtCore, QtWidgets

from peakfit.engine.domain.config import AxisModel, Doublet, FitConfig
from peakfit.gui import theme as themes
from peakfit.gui.editor.axis_table import AxisInfo, AxisModelTable
from peakfit.gui.editor.labels import LABEL_WIDTH

pytestmark = pytest.mark.usefixtures("qapp")

HN = [AxisInfo("F2", "¹⁵N F2", "sp2"), AxisInfo("F3", "¹H F3", "sp2")]
INSPECTOR_CONTENT = 288  # logical px: a 320 px inspector less its margins at 1280×800
_WINDOWS: list[QtWidgets.QWidget] = []


def _table(axes=HN, fitting: FitConfig | None = None):
    """The table in a window with the application theme, as the inspector shows it."""
    window = QtWidgets.QWidget()
    themes.apply(window, themes.LIGHT)
    table = AxisModelTable(axes)
    table.setParent(window)
    emitted: list[FitConfig] = []
    table.changed.connect(emitted.append)
    table.show_model(fitting or FitConfig())
    window.show()
    _WINDOWS.append(window)  # alive for the test
    return table, emitted


def _type(edit: QtWidgets.QLineEdit, text: str) -> None:
    edit.setText(text)
    edit.editingFinished.emit()


def test_add_doublet_opens_j_and_commits_it_with_its_default_range():
    table, emitted = _table()
    nitrogen = table.rows[0]
    assert nitrogen.add.isVisible()
    assert not nitrogen.j.isVisible()
    assert nitrogen.add.accessibleName() == "Add doublet on ¹⁵N F2"
    nitrogen.add.click()
    assert nitrogen.j.isVisible()
    assert not nitrogen.add.isVisible()
    assert not emitted  # no doublet until J is entered
    # Nothing typed yet: no message under the axis, but Fit waits for J.
    assert nitrogen.error.isHidden()
    assert table.invalid == "¹⁵N F2: Enter J as a positive number of Hz."
    _type(nitrogen.j, "abc")
    assert nitrogen.error.text() == "▲ Enter J as a positive number of Hz."
    _type(nitrogen.j, "92")
    (settings,) = emitted
    assert settings.axes == {"F2": AxisModel(doublet=Doublet(j=92.0, min=46.0, max=184.0))}
    table.show_model(settings)
    assert nitrogen.range.isVisible()
    assert (nitrogen.low.text(), nitrogen.high.text()) == ("46.0", "184.0")
    assert nitrogen.fit.isChecked()


def test_an_empty_j_cancels_the_doublet_being_added():
    table, emitted = _table()
    row = table.rows[1]
    row.add.click()
    _type(row.j, "")
    assert row.add.isVisible()
    assert not row.j.isVisible()
    assert not emitted
    assert not table.invalid


def test_an_invalid_range_is_explained_under_its_axis_and_blocks_until_fixed():
    fitting = FitConfig(axes={"F2": AxisModel(doublet=Doublet(j=92.0))})
    table, emitted = _table(fitting=fitting)
    nitrogen, proton = table.rows
    _type(nitrogen.low, "100")
    assert not emitted
    assert nitrogen.error.isVisible()
    assert nitrogen.error.text() == (
        "▲ J start 92.0 Hz is outside its range 100.0–184.0 Hz. "
        "Change the start or widen the range."
    )
    assert proton.error.isHidden()
    assert table.invalid.startswith("¹⁵N F2: J start 92.0 Hz")
    # The message sits directly under that axis's range line.
    grid = table.layout()
    assert isinstance(grid, QtWidgets.QGridLayout)
    error_row = grid.getItemPosition(grid.indexOf(nitrogen.error))[0]
    range_row = grid.getItemPosition(grid.indexOf(nitrogen.range))[0]
    assert error_row > range_row
    assert error_row < grid.getItemPosition(grid.indexOf(proton.name))[0]
    assert nitrogen.low.property("invalid")
    _type(nitrogen.low, "40")
    (settings,) = emitted
    assert settings.axes["F2"].doublet == Doublet(j=92.0, min=40.0, max=184.0)
    assert nitrogen.error.isHidden()
    assert not table.invalid


def test_holding_j_disables_its_range_and_remove_returns_to_a_singlet():
    fitting = FitConfig(axes={"F2": AxisModel(doublet=Doublet(j=92.0))})
    table, emitted = _table(fitting=fitting)
    row = table.rows[0]
    row.fit.setChecked(False)
    held = emitted[-1]
    assert held.axes["F2"].doublet == Doublet(j=92.0, vary=False)
    table.show_model(held)
    assert not row.low.isEnabled()
    assert not row.high.isEnabled()
    assert row.remove.accessibleName() == "Remove doublet on ¹⁵N F2"
    row.remove.click()
    assert emitted[-1].axes == {}
    table.show_model(emitted[-1])
    assert row.add.isVisible()
    assert not row.range.isVisible()


def test_a_phase_on_a_family_without_one_is_refused_under_its_axis():
    fitting = FitConfig(axes={"F3": AxisModel(phase=True)})
    table, emitted = _table(fitting=fitting)
    proton = table.rows[1]
    assert proton.phase.isChecked()
    proton.family.setCurrentIndex(proton.family.findData("gaussian"))
    assert not emitted
    assert "Phase needs a phased family (SP1, SP2 or No apodization)" in proton.error.text()
    assert proton.phase.isEnabled()  # so that it can be turned off
    proton.phase.setChecked(False)
    (settings,) = emitted
    assert settings.axes == {"F3": AxisModel(family="gaussian")}
    assert not table.invalid


def test_one_family_for_every_axis_is_stored_once():
    table, emitted = _table()
    for row in table.rows:
        row.family.setCurrentIndex(row.family.findData("sp1"))
        table.show_model(emitted[-1])
    assert emitted[-1] == FitConfig(family="sp1")


def test_tab_reaches_j_and_its_range_before_the_rest_of_the_axis():
    fitting = FitConfig(axes={"F2": AxisModel(doublet=Doublet(j=92.0))})
    table, _emitted = _table(fitting=fitting)
    first = table.rows[0].family
    order, widget = [], first
    for _ in range(12):
        order.append(widget.accessibleName())
        widget = widget.nextInFocusChain()
        while not (
            widget.isVisible()
            and widget.isEnabled()
            and widget.focusPolicy() & QtCore.Qt.FocusPolicy.TabFocus
        ):
            widget = widget.nextInFocusChain()
        if widget is first:
            break
    assert order[:9] == [
        "¹⁵N F2 lineshape family",
        "¹⁵N F2 J start (Hz)",
        "¹⁵N F2 J lower bound (Hz)",
        "¹⁵N F2 J upper bound (Hz)",
        "Remove doublet on ¹⁵N F2",
        "Fit J on ¹⁵N F2",
        "Fit phase on ¹⁵N F2",
        "¹H F3 lineshape family",
        "Add doublet on ¹H F3",
    ]


def test_homonuclear_axes_keep_distinct_names_on_every_control():
    axes = [AxisInfo("F2", "¹H F2", "sp2"), AxisInfo("F3", "¹H F3", "sp2")]
    table, _emitted = _table(axes)
    names = [widget.accessibleName() for row in table.rows for widget in row.tab_stops()]
    assert len(names) == len(set(names))
    assert all("¹H F2" in name or "¹H F3" in name for name in names)


def test_three_doublet_axes_fit_the_inspector_without_scrolling_sideways():
    axes = [
        AxisInfo("F2", "¹³C F2", "sp2"),
        AxisInfo("F3", "¹⁵N F3", "no_apod"),
        AxisInfo("F4", "¹H F4", "pvoigt"),
    ]
    doublet = AxisModel(doublet=Doublet(j=184.0, vary=False), phase=True)
    fitting = FitConfig(family="no_apod", axes=dict.fromkeys(("F2", "F3", "F4"), doublet))
    table, _emitted = _table(axes, fitting)
    assert table.minimumSizeHint().width() <= INSPECTOR_CONTENT
    # Its axis column is narrower than the page's label column: rows stay compact.
    assert table.rows[0].name.sizeHint().width() < LABEL_WIDTH
