"""Visible keyboard focus: one ring, distinct from selection, on every focusable control.

Specification §12 and UX-4: Fit, link buttons, the navigator list and the 2D canvas
show where Space, Enter or a shortcut will act. Assertions read the rendered pixels.
"""

from copy import deepcopy
from pathlib import Path

import pytest
from builders import two_cluster_dataset
from gui_helpers import wait_until
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from peakfit.gui import theme as themes
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.editor.workspace import EditorWorkspace
from peakfit.gui.peaks import WorkingEdits
from peakfit.gui.spectrum import Dataset

pytestmark = pytest.mark.gui


@pytest.mark.parametrize("theme", [themes.LIGHT, themes.DARK], ids=["light", "dark"])
def test_focus_token_reaches_three_to_one_against_what_it_touches(theme):
    # The ring sits on a surface (outside a control, past a one-pixel gap) or, inside
    # a list, beside its rows: plain, hovered or selected.
    for name in ("window", "chrome", "panel", "plot", "selection", "hover"):
        assert themes.contrast_ratio(theme.focus, getattr(theme, name)) >= 3, name
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
    widget.edits = WorkingEdits(widget.peaks, config.model_copy(deep=True))
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
    # The inspector lays out a new selection's page in a later turn of the event
    # loop; measure the ring around the link at its laid-out size, as a user sees it.
    wait_until(lambda: link.height() >= link.sizeHint().height())
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


def test_ring_marks_keyboard_focus_only_like_focus_visible(workspace):
    # A keyboard focus ring: a click focuses without a ring, the keyboard brings it.
    w = workspace
    tree = w.navigator.tree
    _focus(w.fit_button)
    QtTest.QTest.mouseClick(tree.viewport(), QtCore.Qt.MouseButton.LeftButton)
    wait_until(tree.hasFocus)
    assert _ring_pixels(w, tree) == 0
    QtTest.QTest.keyClick(tree, QtCore.Qt.Key.Key_Down)
    assert _ring_pixels(w, tree) >= 2 * _perimeter(tree) - 64
    QtTest.QTest.mouseClick(tree.viewport(), QtCore.Qt.MouseButton.LeftButton)
    assert _ring_pixels(w, tree) == 0
    QtTest.QTest.keyClick(tree, QtCore.Qt.Key.Key_Tab)
    wait_until(lambda: not tree.hasFocus())
    after = QtWidgets.QApplication.focusWidget()
    assert after is not None
    assert _ring_pixels(w, after) > 0


def test_canvas_frames_its_plot_area_after_a_click_too(workspace):
    # Single-key shortcuts act on the spectrum after a click, so its frame shows then.
    w, c = workspace, workspace.canvas
    _focus(w.fit_button)
    QtTest.QTest.mouseClick(c.view.viewport(), QtCore.Qt.MouseButton.LeftButton)
    wait_until(c.view.hasFocus)
    assert _focus_pixels(w, _area(w, c.view)) > 0


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


@pytest.mark.parametrize("theme", [themes.LIGHT, themes.DARK], ids=["light", "dark"])
def test_focused_checkbox_keeps_its_outline_and_gets_the_shared_ring(qapp, theme):
    # One focus treatment: the style draws no focus of its own, so the checkbox
    # keeps its token outline and the shared ring marks focus around it.
    host = QtWidgets.QWidget()
    host.setObjectName("navigator")  # a panel surface
    host.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
    layout = QtWidgets.QVBoxLayout(host)
    layout.setContentsMargins(12, 12, 12, 12)
    box = QtWidgets.QCheckBox()
    layout.addWidget(box)
    themes.apply(host, theme)
    host.resize(160, 60)
    host.show()
    host.activateWindow()
    _focus(box)
    QtTest.QTest.keyClick(box, QtCore.Qt.Key.Key_Tab)  # a keyboard focus change
    _focus(box)
    qapp.processEvents()
    image = box.grab().toImage()
    outline = QtGui.QColor(theme.outline).rgb()
    assert any(
        image.pixel(x, y) == outline for y in range(image.height()) for x in range(image.width())
    )
    ring = QtGui.QColor(theme.focus).rgb()
    whole = host.grab().toImage()
    area = QtCore.QRect(box.mapTo(host, QtCore.QPoint()), box.size()).adjusted(-4, -4, 4, 4)
    assert sum(
        whole.pixel(x, y) == ring
        for x in range(area.left(), area.right() + 1)
        for y in range(area.top(), area.bottom() + 1)
    ) >= 1.5 * _perimeter(box)
    host.close()


class _Shifting(QtWidgets.QWidget):
    """A container that moves its control when a child arrives, as a native relayout can."""

    def __init__(self) -> None:
        super().__init__()
        self.armed = False
        self.control = QtWidgets.QPushButton("Control", self)
        self.control.move(20, 20)
        self.armed = True

    def childEvent(self, event) -> None:  # noqa: N802
        if self.armed and event.type() == QtCore.QEvent.Type.ChildAdded:
            self.control.move(self.control.x() + 1, self.control.y())
        super().childEvent(event)


class _Events(QtCore.QObject):
    def __init__(self) -> None:
        super().__init__()
        self.kinds: list[QtCore.QEvent.Type] = []

    def eventFilter(self, _watched, event) -> bool:  # noqa: N802
        self.kinds.append(event.type())
        return False


def test_ring_is_not_placed_while_it_is_being_reparented(qapp):
    # Native macOS crashed when a move during the ring's reparent placed and raised
    # the ring before the reparent finished (Tab out of the inspector).
    host = QtWidgets.QWidget()
    inner = _Shifting()
    QtWidgets.QVBoxLayout(host).addWidget(inner)
    themes.apply(host, themes.LIGHT)
    host.resize(200, 120)
    host.show()
    host.activateWindow()
    ring = themes.FocusRing.install(qapp, themes.LIGHT)._painter()
    events = _Events()
    ring.installEventFilter(events)
    _focus(inner.control)
    qapp.processEvents()
    kinds = events.kinds
    reparented = kinds.index(QtCore.QEvent.Type.ParentChange)
    early = {QtCore.QEvent.Type.ZOrderChange, QtCore.QEvent.Type.Move, QtCore.QEvent.Type.Show}
    assert not early & set(kinds[:reparented])
    # Once reparented, the ring still lands on the control where it moved to.
    host_of_ring = ring.parentWidget()
    assert host_of_ring is not None
    origin = inner.control.mapTo(host_of_ring, QtCore.QPoint(0, 0))
    assert ring.geometry().contains(QtCore.QRect(origin, inner.control.size()))
    ring.removeEventFilter(events)
    host.close()


def test_toolbar_buttons_name_their_action_and_tooltips_add_the_keys(workspace):
    w = workspace
    assert w.add_button.accessibleName() == "Add peak"
    assert w.add_button.toolTip() == "Add peak (A)"
    assert w.pin_button.accessibleName() == "Pin slices at the sampled position"
    assert w.pin_button.toolTip().endswith("(Space)")
    assert w.profiles_button.accessibleName() == "Profiles: every cluster's amplitude profiles"
    assert w.inspector_button.accessibleName() == w.inspector_button.toolTip() == "Inspector"
    w.canvas.set_pinned(True)
    assert w.pin_button.accessibleName() == "Slices pinned · click to follow the pointer"
