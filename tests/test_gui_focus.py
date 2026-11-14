"""Visible keyboard focus: one ring, distinct from selection, on every focusable control.

Specification §12 and UX-4: Fit, link buttons, the navigator list and the 2D canvas
show where Space, Enter or a shortcut will act. Assertions read the rendered pixels.
"""

from copy import deepcopy
from pathlib import Path

import pytest
from builders import two_cluster_dataset
from gui_helpers import wait_until
from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.gui import theme as themes
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.editor.workspace import EditorWorkspace
from peakfit.gui.spectrum import Dataset

pytestmark = pytest.mark.gui


def _luminance(value: str) -> float:
    """WCAG 2 relative luminance of a #rrggbb colour."""
    channels = [int(value[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a: str, b: str) -> float:
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


@pytest.mark.parametrize("theme", [themes.LIGHT, themes.DARK], ids=["light", "dark"])
def test_focus_token_reaches_three_to_one_against_what_it_touches(theme):
    # The ring sits on a surface (outside a control, past a one-pixel gap) or, inside
    # a list, beside its rows: plain, hovered or selected.
    for name in ("window", "chrome", "panel", "plot", "selection", "hover"):
        assert _contrast(theme.focus, getattr(theme, name)) >= 3, name
    # A dedicated token, not the accent that marks checked and selected states.
    assert theme.focus != theme.accent


@pytest.fixture(scope="module")
def scientific():
    spectra, peaks, config, prepared = two_cluster_dataset()
    result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    return spectra, peaks, config, result


@pytest.fixture(params=[themes.LIGHT, themes.DARK], ids=["light", "dark"])
def workspace(request, qapp, scientific):
    spectra, peaks, config, result = scientific
    dataset = Dataset(Path("synthetic.ft2"), spectra, peaks, (), True, 0.01)
    widget = EditorWorkspace(dataset)
    widget.fitting.configuration = config.model_copy(deep=True)
    widget.fitting.setup_seen = True
    themes.apply(widget, request.param)
    widget.apply_theme(request.param)
    widget.resize(1280, 750)
    widget.show()
    widget.activateWindow()
    fitting = widget.fitting
    fitting._submitted_revision = widget.edits.revision
    fitting.process.state = "completed"
    fitting._finished(0, result)
    wait_until(lambda: not fitting.busy)
    qapp.processEvents()
    yield widget
    widget.stop()
    widget.close()
    widget.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    qapp.processEvents()


def _focus_pixels(window: EditorWorkspace, area: QtCore.QRect) -> int:
    """Pixels in the focus colour within `area` (window coordinates), as rendered."""
    QtWidgets.QApplication.processEvents()
    image = window.grab().toImage()
    focus = QtGui.QColor(window.theme.focus).rgb()
    area = area.intersected(image.rect())
    return sum(
        image.pixel(x, y) == focus
        for x in range(area.left(), area.right() + 1)
        for y in range(area.top(), area.bottom() + 1)
    )


def _area(window: QtWidgets.QWidget, target: QtWidgets.QWidget) -> QtCore.QRect:
    return QtCore.QRect(target.mapTo(window, QtCore.QPoint(0, 0)), target.size())


def _ring_pixels(window: EditorWorkspace, target: QtWidgets.QWidget) -> int:
    """Focus-colour pixels on `target` and just around it."""
    return _focus_pixels(window, _area(window, target).adjusted(-4, -4, 4, 4))


def _focus(widget: QtWidgets.QWidget) -> None:
    widget.setFocus(QtCore.Qt.FocusReason.TabFocusReason)
    wait_until(widget.hasFocus)


def _perimeter(widget: QtWidgets.QWidget) -> int:
    return 2 * (widget.width() + widget.height())


def test_fit_button_shows_a_ring_only_while_focused(workspace):
    w = workspace
    assert _ring_pixels(w, w.fit_button) == 0
    _focus(w.fit_button)
    # A two-pixel ring (rounded corners) around the whole accent-filled button.
    assert _ring_pixels(w, w.fit_button) >= 1.5 * _perimeter(w.fit_button)
    _focus(w.navigator.tree)
    assert _ring_pixels(w, w.fit_button) == 0


def test_link_button_shows_a_ring_when_focused(workspace):
    w = workspace
    w.select_peak(w.peaks[0].id)
    links = [
        button
        for button in w.panel.findChildren(QtWidgets.QPushButton, "link")
        if button.isVisibleTo(w)
    ]
    assert links
    link = links[0]
    assert _ring_pixels(w, link) == 0
    _focus(link)
    assert _ring_pixels(w, link) >= 1.5 * _perimeter(link)


def test_navigator_list_shows_focus_on_the_view_distinct_from_selection(workspace):
    w = workspace
    tree = w.navigator.tree
    tree.setCurrentItem(tree.topLevelItem(0))
    _focus(w.fit_button)
    # A selected row alone is not focus.
    assert tree.selectedItems()
    assert _ring_pixels(w, tree) == 0
    _focus(tree)
    assert _ring_pixels(w, tree) >= 2 * _perimeter(tree) - 64


def test_checked_controls_do_not_look_focused(workspace):
    w = workspace
    checked = [b for b in w.mode_buttons if b.isChecked()]
    assert checked
    _focus(w.navigator.tree)
    for button in checked:
        assert _ring_pixels(w, button) == 0


def test_canvas_frames_its_plot_area_while_focused(workspace):
    w, c = workspace, workspace.canvas
    canvas = _area(w, c.view)
    _focus(w.fit_button)
    assert _focus_pixels(w, canvas) == 0
    _focus(c.view)
    x, y, width, height = (int(v) for v in c.arrangement.main)
    plot = QtCore.QRect(canvas.topLeft() + QtCore.QPoint(x, y), QtCore.QSize(width, height))
    # A two-pixel frame around the 2D plot area where A, Space and E/O/R act ...
    assert _focus_pixels(w, plot.adjusted(-3, -3, 3, 3)) >= 4 * (width + height) - 32
    # ... not around the whole canvas, which holds the slices too.
    for edge in (
        QtCore.QRect(canvas.left(), canvas.top(), canvas.width(), 2),
        QtCore.QRect(canvas.right() - 1, canvas.top(), 2, canvas.height()),
    ):
        assert _focus_pixels(w, edge) == 0
    _focus(w.navigator.tree)
    assert _focus_pixels(w, canvas) == 0
