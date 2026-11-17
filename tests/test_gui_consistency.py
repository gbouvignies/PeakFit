"""One vocabulary across the GUI: axis names, controls, shortcuts and stable geometry (#211)."""

import math
import re
from copy import deepcopy
from pathlib import Path

import pytest
from builders import two_cluster_dataset
from gui_helpers import wait_until
from PySide6 import QtCore, QtTest, QtWidgets

from peakfit.engine.domain.spectrum import get_dimension_label
from peakfit.gui import formatting
from peakfit.gui import theme as themes
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.editor.numerical import NumericalReview, parameter_rows, peak_rows
from peakfit.gui.editor.window import EditorWindow, menu_shortcuts, shortcut_sections
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.spectrum import Dataset

pytestmark = pytest.mark.gui

FOCUS = 0  # the overlapping pair's first peak: a two-member cluster
_AXIS_MENTION = re.compile(r"(?:(?<=\s)|^)(\S+ )?F[2-4]\b")


def _fit(nuclei: tuple[str | None, str | None]):
    """The two-cluster dataset with its axes labelled ``nuclei`` (vertical, horizontal)."""
    spectra, peaks, config, _ = two_cluster_dataset()
    for index, (axis, name) in enumerate(zip(spectra.spectral_params, nuclei, strict=True)):
        axis.nucleus = name
        axis.label = get_dimension_label(index)  # as the NMRPipe reader names them
    prepared = prepare_dataset_fit(spectra, peaks, config)
    result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    return spectra, peaks, config, result


@pytest.fixture(scope="module")
def heteronuclear():
    return _fit(("15N", "1H"))


def _window(qapp, scientific, size=(1280, 800)):
    spectra, peaks, config, _result = scientific
    window = EditorWindow()
    window.protect_edits = lambda **_: True  # ty: ignore[invalid-assignment]
    window.resize(*size)
    window.show()
    window.install_dataset(Dataset(Path("consistency.ft2"), spectra, peaks, (), True, 0.01))
    w = window.workspace
    assert w is not None
    w.fitting.configuration = config.model_copy(deep=True)
    w.fitting.setup_seen = True
    qapp.processEvents()
    return window, w


def _publish(w, result) -> None:
    fitting = w.fitting
    fitting._submitted_revision = w.edits.revision
    fitting.process.state = "completed"
    fitting._finished(0, result)
    wait_until(lambda: not fitting.busy)
    assert fitting.result is result, fitting.failure


def _close(qapp, window) -> None:
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    qapp.processEvents()


def _texts(root: QtWidgets.QWidget) -> list[str]:
    """Every visible label and button text, rich text reduced to plain.

    Qt shows a widget added to an already visible layout through a queued call, so
    rows a page has just filled appear on the next pass of the event loop. Deliver
    those first: the texts are what the window shows, not a half-delivered state.
    """
    QtCore.QCoreApplication.processEvents()
    found = []
    for widget in root.findChildren(QtWidgets.QWidget):
        if not widget.isVisible():
            continue
        if isinstance(widget, (QtWidgets.QLabel, QtWidgets.QAbstractButton)):
            plain = re.sub(r"<[^>]+>", "", widget.text()).replace("&nbsp;", " ")
            found.append(plain.replace(" ", " "))
    return found


def _peak_page_state(w) -> str:
    """What a failure needs to name its cause: panel, page, selection and Fitted rows."""
    panel = w.panel
    rows = [panel.fitted.itemAt(i).widget() for i in range(panel.fitted.count())]
    return (
        f"inspector visible {panel.isVisible()}, peak page {panel.peak_page.isVisible()}, "
        f"selected {w.selected_id}, Fitted title {panel.fitted_title.isVisible()}, "
        f"Fitted rows {[(r.text(), r.isVisible()) for r in rows if r is not None][:4]}"
    )


# ------------------------------------------------------------------ axis names
@pytest.mark.parametrize(
    "nuclei",
    [("15N", "1H"), ("13C", "1H"), ("1H", "1H")],
    ids=["HN", "HC", "homonuclear"],
)
def test_axis_names_are_distinct_and_identical_in_every_place(qapp, nuclei):
    scientific = _fit(nuclei)
    window, w = _window(qapp, scientific)
    _publish(w, scientific[-1])
    axes = w.dataset.spectra.spectral_params
    names = [formatting.axis_name(a) for a in axes]
    assert len(set(names)) == len(names)
    y_name, x_name = names
    c = w.canvas
    # The plot axes, the editor, the peak list and the fitted rows all use the name.
    assert c.bottom_axis.labelText == f"{x_name} (ppm)"
    assert c.left_axis.labelText == f"{y_name} (ppm)"
    # Let the published run's contours settle first: then nothing else spins the event
    # loop after the selection, which is when timing used to hide the bug below.
    wait_until(lambda: not c.contour_pending)
    w.select_peak(w.peaks[FOCUS].id)
    wait_until(lambda: not c.contour_pending)
    texts = _texts(window)
    for name in names:
        assert f"{name} (ppm)" in texts  # editor field and peak-list heading
        assert f"δ {name} (ppm)" in texts, _peak_page_state(w)  # fitted row
    w.select_peak(w.peaks[FOCUS].id, cluster=True)
    qapp.processEvents()
    texts += _texts(window)
    assert set(names) <= set(texts)  # the members table names each axis once
    w.select_peak(None)
    w.show_fit_setup()
    texts += _texts(window)
    texts.append(c.view.accessibleDescription())
    # Wherever an F-dimension is mentioned, it is one of the axis names, whole.
    mentions = {m.group(0).strip() for text in texts for m in _AXIS_MENTION.finditer(text)}
    assert mentions
    assert mentions <= set(names), mentions - set(names)
    assert not any(re.search(r"\b(direct|indirect)\b", t, re.IGNORECASE) for t in texts)
    _close(qapp, window)


# ------------------------------------------------------------------ geometry
@pytest.mark.parametrize("size", [(1160, 720), (1280, 800), (1920, 1080)], ids=str)
def test_the_2d_plot_never_moves_between_comparison_modes(qapp, heteronuclear, size):
    window, w = _window(qapp, heteronuclear, size)
    c = w.canvas
    # Before the first fit no mode shows lanes, whatever the preference.
    c.set_lanes(True)
    qapp.processEvents()
    assert not c.row_lane.isVisible()
    _publish(w, heteronuclear[-1])
    for view in ("overview", "focus"):
        if view == "focus":
            w.open_focus(w.peaks[FOCUS].id)
            qapp.processEvents()
        plots = {}
        for mode in (0, 1, 2, 1, 0):
            w.set_mode(mode)
            wait_until(lambda: not c.contour_pending)
            qapp.processEvents()
            assert c.comparison_mode == mode
            plots[mode] = c.box.sceneBoundingRect().toRect()
            # Once a run exists the lanes (asked for before the fit) show in every mode.
            assert c.row_lane.isVisible()
            assert c.column_lane.isVisible()
        assert len(set(plots.values())) == 1, (view, plots)
    _close(qapp, window)


# ------------------------------------------------------------------ values
_FROM_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
_KINDS = {"cs": "δ", "lw": "LW"}


def _within_shown_precision(value_text: str, sigma_text: str, power: int, recorded: float):
    """|shown − recorded| ≤ half the last digit shown (the σ's second significant digit)."""
    if sigma_text in ("fixed", formatting.NOT_ASSESSED):
        decimals = len(value_text.partition(".")[2])
        place = 10.0 ** (power - decimals)
    else:
        sigma = formatting.parse_scaled(sigma_text)
        place = 10.0 ** (math.floor(math.log10(sigma)) - 1 + power)
    shown = formatting.parse_scaled(value_text) * 10.0**power
    return abs(shown - recorded) <= 0.5 * place * (1 + 1e-9)


def test_every_shown_value_matches_the_outcome_at_its_precision(qapp, heteronuclear):
    window, w = _window(qapp, heteronuclear)
    result = heteronuclear[-1]
    _publish(w, result)
    names = {a.label: formatting.axis_name(a) for a in w.dataset.spectra.spectral_params}
    identities = {i.engine_name: i.working_id for i in result.inputs.identities}
    cluster = next(c for c in result.outcome.clusters if c.usable and len(c.peak_names) > 1)
    engine = cluster.peak_names[0]
    w.select_peak(identities[engine])
    fitted = w.panel.fitted
    rows = {}
    for row in range(1, fitted.rowCount()):
        cells = [fitted.itemAtPosition(row, column) for column in (0, 1, 2)]
        if all(cell is not None and cell.widget() is not None for cell in cells):
            title, value, sigma = (cell.widget().text() for cell in cells)
            rows[title] = (value, sigma)
    checked = 0
    for parameter in cluster.final_nonlinear_parameters:
        _peak, axis, kind = parameter.name.split(".")
        if _peak != engine:
            continue
        value, sigma = rows[f"{_KINDS.get(kind, kind)} {names[axis]} ({parameter.unit})"]
        assert _within_shown_precision(value, sigma, 0, parameter.value), parameter.name
        checked += 1
    assert checked >= 4  # positions and linewidths on both axes
    title, (value, sigma) = next((t, v) for t, v in rows.items() if t.startswith("Amplitude"))
    power = re.search(r"× 10([⁰-⁹⁻]+)", title)
    exponent = int(power.group(1).translate(_FROM_SUPERSCRIPT)) if power else 0
    recorded = float(cluster.analytical_evaluation.amplitudes[0, w.canvas.plane])
    assert _within_shown_precision(value, sigma, exponent, recorded)
    # χ²ᵣ: two decimals in the list and on the cluster page alike, the recorded value.
    review = next(r for r in w.reviews() if r.cluster_id == cluster.cluster_id)
    item = w.navigator._clusters[review.ids[0]]
    assert item.text(3) == formatting.chi2(review.chi2)
    assert abs(float(item.text(3)) - review.chi2) <= 0.005
    w.select_peak(identities[engine], cluster=True)
    qapp.processEvents()
    assert formatting.chi2(review.chi2) in _texts(w.panel.cluster_page)
    _close(qapp, window)


# ------------------------------------------------------------------ controls
def test_each_control_has_one_of_the_defined_looks(qapp, heteronuclear):
    window, w = _window(qapp, heteronuclear)
    _publish(w, heteronuclear[-1])
    kinds = {*themes.CONTROL_KINDS, "tab"}
    states = (
        lambda: w.select_peak(w.peaks[FOCUS].id),
        lambda: w.select_peak(w.peaks[FOCUS].id, cluster=True),
        lambda: w.open_focus(w.peaks[FOCUS].id),
        lambda: (w.close_focus(), w.show_view(profiles=True)),
    )
    for state in states:
        state()
        qapp.processEvents()
        for button in window.findChildren(QtWidgets.QPushButton):
            assert button.objectName() in kinds, (button.text(), button.objectName())
            if button.objectName() == themes.MENU:
                assert button.text().endswith(themes.MENU_ARROW)
    primaries = [
        b for b in window.findChildren(QtWidgets.QPushButton) if b.objectName() == themes.PRIMARY
    ]
    assert primaries == [w.fit_button]
    # A toggle is a checkbox: it never looks like a command.
    assert isinstance(w.focus_bar.regions, QtWidgets.QCheckBox)
    assert isinstance(w.panel.profile.show_all, QtWidgets.QCheckBox)
    _close(qapp, window)


# ------------------------------------------------------------------ shortcuts
def test_the_shortcuts_list_names_every_key(qapp, heteronuclear):
    window, _w = _window(qapp, heteronuclear)
    sections = dict(shortcut_sections(window.menuBar()))
    listed = {command for rows in sections.values() for _key, command in rows}
    for action in window.menuBar().actions():
        menu = action.menu()
        if isinstance(menu, QtWidgets.QMenu):
            assert {c for _k, c in menu_shortcuts(menu)} <= listed
    assert {"Fit dataset", "Save analysis", "Undo peak edit", "Focus cluster"} <= listed
    keys = {key for rows in sections.values() for key, _command in rows}
    assert {"E · O · R", "[ · ]", "N · Shift+N", "A", "Space", "Escape"} <= keys
    _close(qapp, window)


def test_single_keys_on_the_spectrum_choose_the_mode_and_the_plane(qapp, heteronuclear):
    window, w = _window(qapp, heteronuclear)
    _publish(w, heteronuclear[-1])
    view = w.canvas.view
    view.setFocus()
    for key, mode in (("R", 2), ("E", 0), ("O", 1)):
        QtTest.QTest.keyClick(view, getattr(QtCore.Qt.Key, f"Key_{key}"))
        assert w.canvas.comparison_mode == mode
    plane = w.canvas.plane
    QtTest.QTest.keyClick(view, QtCore.Qt.Key.Key_BracketRight)
    assert w.canvas.plane == plane + 1
    QtTest.QTest.keyClick(view, QtCore.Qt.Key.Key_BracketLeft)
    assert w.canvas.plane == plane
    # In a text field the same keys type.
    field = w.inspector.name
    w.select_peak(w.peaks[FOCUS].id)
    field.setFocus()
    QtTest.QTest.keyClick(field, QtCore.Qt.Key.Key_R)
    assert w.canvas.comparison_mode == 1
    _close(qapp, window)


# ------------------------------------------------------------------ numerical review
@pytest.mark.usefixtures("qapp")
def test_numerical_review_tables_share_the_formatter():
    rows = parameter_rows([("p.F2.cs", 120.20348, "ppm", 0.00059, 100.0, None)])
    _name, value, unit, sigma, low, high = rows[0]
    assert (value.text, unit, sigma.text, low.text, high.text) == (
        "120.20348",
        "ppm",
        "0.00059",
        "100.000",
        "–",
    )
    assert value.value == 120.20348  # the tooltip keeps the recorded value
    unassessed = parameter_rows([("p.F2.lw", 15.7312, "Hz", math.nan, 0.1, 200.0)])[0]
    assert (unassessed[1].text, unassessed[3].text) == ("15.7", formatting.NOT_ASSESSED)
    assert [c.text for c in peak_rows([("22N-H", "id", (120.20348, 7.98585))])[0][2:]] == [
        "120.203",
        "7.986",
    ]
    table = NumericalReview.table(["Observation"], [])
    NumericalReview.populate(table, [(3.1e9,), (1.2e9,)], ["Observation"])
    assert table.horizontalHeaderItem(0).text() == "Observation (× 10⁹)"
    assert [table.item(r, 0).text() for r in (0, 1)] == ["3.1", "1.2"]
    assert table.item(0, 0).toolTip() == formatting.exact(3.1e9)
