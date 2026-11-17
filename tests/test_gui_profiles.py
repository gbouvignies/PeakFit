"""The Profiles view: every cluster's profiles, in the navigator's order, sharing selection."""

import threading
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from builders import cpmg_like_dataset, decay_dataset, two_cluster_dataset
from gui_helpers import wait_until
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from peakfit.engine.algorithms.covariance_factors import CovarianceFactors
from peakfit.engine.scientific.profiles import Normalized, normalize
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.editor import workspace as workspace_module
from peakfit.gui.editor.profile import PeakProfile, ProfilePlot, peak_profile
from peakfit.gui.editor.profiles_view import (
    TILE_ASPECT,
    TILE_MAX_WIDTH,
    TILE_MIN_HEIGHT,
    TILE_MIN_WIDTH,
    ProfilesView,
    cluster_tiles,
    shown_rows,
    tile_grid,
)
from peakfit.gui.editor.review import HighlightRules, review_clusters
from peakfit.gui.editor.window import EditorWindow
from peakfit.gui.editor.workspace import EditorWorkspace
from peakfit.gui.peaks import WorkingEdits
from peakfit.gui.spectrum import Dataset

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def scientific():
    spectra, peaks, config, prepared = two_cluster_dataset()
    result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    return spectra, peaks, config, result


@pytest.fixture
def workspace(qapp, scientific):
    spectra, peaks, config, _ = scientific
    widget = EditorWorkspace(Dataset(Path("synthetic.ft2"), spectra, peaks, (), True, 0.01))
    widget.edits = WorkingEdits(widget.peaks, config.model_copy(deep=True))
    widget.fitting.model_seen = True
    widget.resize(1280, 750)
    widget.show()
    qapp.processEvents()
    yield widget
    widget.stop()
    widget.close()
    widget.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    qapp.processEvents()


def publish(workspace, result):
    fitting = workspace.fitting
    fitting._submitted_revision = workspace.edits.revision
    fitting.process.state = "completed"
    fitting._finished(0, result)
    wait_until(lambda: not fitting.busy)
    assert fitting.result is result, fitting.failure


def show_profiles(workspace):
    """Show the Profiles view and wait for its tiles, computed off the GUI thread."""
    workspace.show_view(profiles=True)
    wait_until(lambda: "Computing" not in workspace.profiles.caption.text())


def test_tiles_hold_every_member_in_cluster_order(scientific):
    result = scientific[-1]
    axes = list(scientific[0].spectral_params)
    reviews = review_clusters(result, axes, ["¹H", "¹⁵N"][: len(axes)], HighlightRules())
    tiles = cluster_tiles(result, reviews, None)
    assert set(tiles) == {r.cluster_id for r in reviews}
    for review in reviews:
        tile = tiles[review.cluster_id]
        assert tile.profile is not None
        # Members keep the cluster's order, so colours match the cluster page.
        assert tuple(s.assignment for s in tile.profile.series) == review.names


def test_profiles_follow_the_run_navigator_order_and_selection(workspace, scientific):
    w = workspace
    profiles = w.profiles_button
    assert not profiles.isEnabled()
    w.show_view(profiles=True)
    assert not w.profiles_shown  # nothing to show before a fit
    publish(w, scientific[-1])
    assert profiles.isEnabled()
    QtTest.QTest.mouseClick(profiles, QtCore.Qt.MouseButton.LeftButton)
    assert w.profiles_shown
    wait_until(lambda: "Computing" not in w.profiles.caption.text())
    # Spectrum tools step aside; the plane strip and selection stay shared.
    assert not any(tool.isVisible() for tool in w._spectrum_tools)
    order = [r.ids for r in w.navigator.shown_reviews()]
    assert w.profiles.tile_ids() == order
    # Sorting the cluster list by member count, largest first, reorders the tiles.
    w.navigator.tree.sortByColumn(2, QtCore.Qt.SortOrder.DescendingOrder)
    wait_until(lambda: w.profiles.tile_ids() == sorted(order, key=len, reverse=True))

    # Choosing a tile selects its cluster; a selection elsewhere marks its tile.
    last = len(w.profiles.tile_ids()) - 1
    point = w.profiles.list.visualRect(w.profiles.model.index(last)).center()
    viewport = w.profiles.list.viewport()
    QtTest.QTest.mouseClick(viewport, QtCore.Qt.MouseButton.LeftButton, pos=point)
    assert w.selected_id in w.profiles.tile_ids()[last]
    assert w.panel.cluster_view
    other = w.profiles.tile_ids()[0][0]
    w.select_peak(other)
    assert w.profiles.list.currentIndex().row() == 0

    w.set_plane(2)
    assert w.profiles.plane == 1


def test_tiles_are_computed_off_the_gui_thread_once_per_run(workspace, scientific, monkeypatch):
    w = workspace
    publish(w, scientific[-1])
    threads = []

    def recording(*args):
        threads.append(threading.current_thread())
        return cluster_tiles(*args)

    monkeypatch.setattr(workspace_module, "cluster_tiles", recording)
    show_profiles(w)
    assert len(threads) == 1
    assert threads[0] is not threading.main_thread()
    assert w.profiles.caption.text().startswith(f"{len(w.profiles.tile_ids())} clusters")
    # Leaving and returning to the view reuses the run's tiles.
    w.show_view(profiles=False)
    w.show_view(profiles=True)
    assert len(w.profiles.tile_ids()) == len(w.navigator.shown_reviews())
    assert len(threads) == 1


def test_a_tile_failure_says_what_went_wrong(workspace, scientific, monkeypatch):
    w = workspace
    publish(w, scientific[-1])

    def failing(*_args):
        raise ValueError("singular covariance")

    monkeypatch.setattr(workspace_module, "cluster_tiles", failing)
    show_profiles(w)
    assert not w.profiles.tile_ids()
    assert "Profiles could not be computed: singular covariance" in w.profiles.empty.text()
    assert w.profiles.empty.isVisible()


def test_a_filter_narrows_the_tiles_and_says_so(workspace, scientific):
    w = workspace
    publish(w, scientific[-1])
    show_profiles(w)
    total = len(w.profiles.tile_ids())
    assert w.profiles.caption.text().startswith(f"{total} clusters")
    w.navigator.filter.setCurrentIndex(w.navigator.filter.count() - 1)
    shown = w.navigator.shown_reviews()
    assert w.profiles.tile_ids() == [r.ids for r in shown]
    if len(shown) < total:
        assert w.profiles.caption.text().startswith(f"{len(shown)} of {total} clusters")
    if not shown:
        assert w.profiles.empty.isVisible()


@pytest.mark.usefixtures("settings")
def test_a_filtered_profiles_view_never_widens_the_window(qapp, scientific, monkeypatch):
    """The caption names the cluster-list filter; however long, it never widens the
    window or takes width from the navigator or the inspector (spec §2)."""
    spectra, peaks, config, result = scientific
    window = EditorWindow()
    monkeypatch.setattr(window, "protect_edits", lambda **_: True)
    window.resize(1160, 720)
    window.show()
    window.install_dataset(Dataset(Path("synthetic.ft2"), spectra, peaks, (), True, 0.01))
    w = window.workspace
    assert w is not None
    w.fitting.configuration = config.model_copy(deep=True)
    publish(w, result)
    w.navigator.filter.setCurrentIndex(w.navigator.filter.count() - 1)
    qapp.processEvents()
    minimum, size = window.minimumWidth(), window.size()
    navigator, _spectral, inspector = w.splitter.sizes()
    show_profiles(w)
    chosen = w.navigator.filter_text()
    assert chosen
    assert chosen in w.profiles.caption.text()
    for _ in range(3):  # let every layout request settle
        qapp.processEvents()
    assert window.minimumWidth() <= minimum
    assert window.size() == size
    assert w.splitter.sizes()[0] == navigator
    assert w.splitter.sizes()[2] == inspector
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def test_activating_a_tile_opens_its_cluster_in_focus(workspace, scientific):
    w = workspace
    publish(w, scientific[-1])
    show_profiles(w)
    ids = w.profiles.tile_ids()[0]
    index = w.profiles.model.index(0)
    point = w.profiles.list.visualRect(index).center()
    viewport = w.profiles.list.viewport()
    QtTest.QTest.mouseClick(viewport, QtCore.Qt.MouseButton.LeftButton, pos=point)
    QtTest.QTest.mouseDClick(viewport, QtCore.Qt.MouseButton.LeftButton, pos=point)
    assert not w.profiles_shown
    assert w.focus_open
    assert w.selected_id in ids
    assert all(tool.isVisible() for tool in w._spectrum_tools)
    assert not w.profiles_button.isChecked()


def test_an_edit_labels_the_tiles_historical(workspace, scientific):
    w = workspace
    result = scientific[-1]
    publish(w, result)
    show_profiles(w)
    assert "historical" not in w.profiles.caption.text()
    peak = w.peaks[0]
    w.move_peak(peak.id, tuple(p + 0.001 for p in peak.positions))
    assert not w.fitting.current
    assert f"Run {result.run_id[:8]} · historical" in w.profiles.caption.text()


def _tile(scientific, members):
    result = scientific[-1]
    axes = list(scientific[0].spectral_params)
    reviews = review_clusters(result, axes, ["¹H", "¹⁵N"][: len(axes)], HighlightRules())
    tiles = cluster_tiles(result, reviews, None)
    return next(t for t in tiles.values() if len(t.review.ids) == members)


def test_tiles_show_ratios_or_say_why_not(scientific):
    profile = _tile(scientific, 2).profile
    assert profile is not None
    raw = shown_rows(profile, (0,), normalized=False)
    assert not raw.ratio
    assert [i for i, _ in raw.rows] == [0, 1]
    ratio = shown_rows(profile, (0,), normalized=True)
    assert ratio.ratio
    for (index, values), series in zip(ratio.rows, profile.series, strict=True):
        expected = normalize(series.amplitudes, series.covariance, np.array([0]))
        assert isinstance(expected, Normalized)
        np.testing.assert_array_equal(values, expected.values)
        assert index in (0, 1)
    # A member whose reference is not distinguishable from its σ is left out and named.
    variance = float(profile.series[1].amplitudes[0]) ** 2
    weak = replace(profile.series[1], covariance=profile.series[1].covariance + variance)
    mixed = shown_rows(replace(profile, series=(profile.series[0], weak)), (0,), normalized=True)
    assert mixed.ratio
    assert [i for i, _ in mixed.rows] == [0]
    assert any(n.startswith(f"Not normalized: {weak.assignment}") for n in mixed.notes)
    # With no member normalizable, raw amplitudes are drawn and the tile says why.
    selected = profile.series[0]
    hopeless = selected.covariance + 1e4 * float(selected.amplitudes[0]) ** 2
    both = replace(profile, series=(replace(selected, covariance=hopeless), weak))
    fallback = shown_rows(both, (0,), normalized=True)
    assert not fallback.ratio
    assert any(n.startswith("Raw:") for n in fallback.notes)
    missing = shown_rows(profile, (10_000,), normalized=True)
    assert not missing.ratio
    assert "Raw: a reference plane is not in this run" in missing.notes


def test_one_reference_choice_serves_the_inspector_and_the_tiles(workspace, scientific):
    w = workspace
    publish(w, scientific[-1])
    show_profiles(w)
    w.panel.profile.set_reference((0,))
    assert w.profiles.reference == (0,)
    assert w.profiles.normalized
    assert w.profiles.mode.text() == "÷ plane 1 ▾"
    w.profiles.quantity_changed.emit((0,), False)
    assert not w.panel.profile.normalized
    assert w.panel.profile.reference == (0,)
    assert w.profiles.mode.text() == "Raw amplitudes ▾"


def test_exporting_writes_every_shown_tile_to_a_pdf(workspace, scientific, tmp_path):
    w = workspace
    publish(w, scientific[-1])
    show_profiles(w)
    target = tmp_path / "profiles.pdf"
    w.profiles.export_figure(str(target))
    data = target.read_bytes()
    assert data.startswith(b"%PDF")
    assert list(tmp_path.iterdir()) == [target]  # no partial file left beside it
    assert "Exported 2 tiles on 1 page to profiles.pdf" in w.profiles.caption.text()


def _engine_ids(result) -> dict[str, str]:
    return {i.engine_name: i.working_id for i in result.inputs.identities}


def _pair_cluster(result):
    return next(c for c in result.outcome.clusters if c.usable and len(c.peak_names) > 1)


def _tile_text(w, member: str) -> str:
    """The tooltip and accessible text of the tile holding this member."""
    row = next(r for r, ids in enumerate(w.profiles.tile_ids()) if member in ids)
    return w.profiles.model.index(row).data(QtCore.Qt.ItemDataRole.ToolTipRole)


def _review_value(page, label: str) -> QtWidgets.QLabel:
    grid = page.review
    for row in range(grid.rowCount()):
        item = grid.itemAtPosition(row, 0)
        if item is not None and item.widget().text() == label:
            return grid.itemAtPosition(row, 1).widget()
    raise KeyError(label)


def test_the_correlated_profile_flag_reads_alike_on_profile_tiles_and_cluster_page(
    workspace, scientific
):
    # The overlapping pair's within-plane |ρ| is 0.17, above 0.15: both are flagged.
    # The remote peak, alone in its cluster, never is.
    w = workspace
    result = scientific[-1]
    publish(w, result)
    ids = _engine_ids(result)
    pair = _pair_cluster(result)
    remote = next(c for c in result.outcome.clusters if len(c.peak_names) == 1)
    member = ids[pair.peak_names[0]]
    w.select_peak(member)
    profile = w.panel.profile
    note = profile.correlated.text()
    assert profile.correlated.isVisible()
    # One plain sentence naming the flagged members, selected peak first: the bars
    # do not show the covariance between planes. It never says σ is understated.
    names = {i.engine_name: i.assignment for i in result.inputs.identities}
    first, second = names[pair.peak_names[0]], names[pair.peak_names[1]]
    assert note == (
        f"Errors of {first} and {second} are correlated across planes; "
        "error bars alone do not show this covariance."
    )
    assert all(
        word not in note.lower()
        for word in ("understat", "underestimat", "too small", "discard", "bad", "unreliable")
    )
    # The findings and the downstream consequence stay one step away.
    assert "|ρ| 0.17" in profile.correlated.toolTip()
    assert "intensity_err" in profile.correlated.toolTip()
    profile.about_toggle.setChecked(True)
    about = profile.about.text()
    assert "intensity_err" in about
    assert "downstream" in about.lower()
    assert "not of the data" in about
    assert profile.legend.text().count("↔") == 2  # both members carry the mark
    show_profiles(w)
    assert "↔ errors correlated across planes" in _tile_text(w, member)
    assert "↔" not in _tile_text(w, ids[remote.peak_names[0]])
    w.select_peak(member, cluster=True)
    row = _review_value(w.panel.cluster_page, "Profile errors")
    assert row.text().startswith("↔ correlated across planes")
    assert "not of the data" in row.toolTip()
    w.select_peak(ids[remote.peak_names[0]])
    assert not profile.correlated.isVisible()
    assert "↔" not in profile.legend.text()
    w.select_peak(ids[remote.peak_names[0]], cluster=True)
    row = _review_value(w.panel.cluster_page, "Profile errors")
    assert row.text() == "within tolerance"


def test_withheld_lineshape_factors_leave_the_profile_not_assessed(workspace, scientific):
    w = workspace
    result = scientific[-1]
    pair = _pair_cluster(result)
    evaluation = pair.analytical_evaluation
    withheld = replace(
        evaluation,
        covariance_factors=CovarianceFactors(evaluation.amplitude_covariance, None, None, None),
    )
    edited = replace(pair, analytical_evaluation=withheld)
    outcome = replace(
        result.outcome,
        clusters=tuple(edited if c is pair else c for c in result.outcome.clusters),
    )
    publish(w, replace(result, completed_fit=replace(result.completed_fit, outcome=outcome)))
    member = _engine_ids(result)[pair.peak_names[0]]
    w.select_peak(member)
    note = w.panel.profile.correlated.text()
    assert "not assessed" in note
    assert "↔" not in w.panel.profile.legend.text()
    show_profiles(w)
    assert "correlation across planes not assessed" in _tile_text(w, member)
    w.select_peak(member, cluster=True)
    assert _review_value(w.panel.cluster_page, "Profile errors").text() == "not assessed"


def _activate(w):
    """Make the workspace the active window, so focus changes deliver focus events."""
    w.activateWindow()
    assert QtTest.QTest.qWaitForWindowActive(w)


def test_showing_profiles_keeps_the_selected_peak_both_ways(workspace, scientific):
    w = workspace
    publish(w, scientific[-1])
    _activate(w)
    # A member of the last cluster, so the first tile can never be mistaken for it.
    last = [r.ids for r in w.navigator.shown_reviews()][-1]
    chosen = last[-1]
    w.select_peak(chosen)
    assert w.selected_id == chosen

    show_profiles(w)
    wait_until(lambda: w.profiles.list.hasFocus())
    QtWidgets.QApplication.processEvents()
    assert w.selected_id == chosen
    current = w.profiles.list.currentIndex()
    assert current.isValid()
    assert chosen in w.profiles.tile_ids()[current.row()]
    assert w.profiles.list.selectionModel().isSelected(current)

    w.show_view(profiles=False)
    QtWidgets.QApplication.processEvents()
    assert w.selected_id == chosen
    show_profiles(w)  # a second opening reuses the built tiles
    QtWidgets.QApplication.processEvents()
    assert w.selected_id == chosen


def test_showing_profiles_with_nothing_selected_selects_nothing(workspace, scientific):
    w = workspace
    publish(w, scientific[-1])
    _activate(w)
    w.select_peak(None)
    show_profiles(w)
    wait_until(lambda: w.profiles.list.hasFocus())
    QtWidgets.QApplication.processEvents()
    assert w.selected_id is None
    assert not w.profiles.list.selectionModel().hasSelection()


def test_keys_and_clicks_on_tiles_still_choose_a_cluster(workspace, scientific):
    w = workspace
    publish(w, scientific[-1])
    _activate(w)
    show_profiles(w)
    wait_until(lambda: w.profiles.list.hasFocus())
    tiles = w.profiles.tile_ids()
    assert len(tiles) >= 2
    viewport = w.profiles.list.viewport()
    point = w.profiles.list.visualRect(w.profiles.model.index(1)).center()
    QtTest.QTest.mouseClick(viewport, QtCore.Qt.MouseButton.LeftButton, pos=point)
    assert w.selected_id in tiles[1]
    assert w.panel.cluster_view
    QtTest.QTest.keyClick(w.profiles.list, QtCore.Qt.Key.Key_Left)
    assert w.selected_id in tiles[0]
    QtTest.QTest.keyClick(w.profiles.list, QtCore.Qt.Key.Key_Right)
    assert w.selected_id in tiles[1]


# Adaptive tiles: the grid fills the width with as many readable tiles as fit.

WIDTHS = range(120, 4000, 13)


def test_tiles_stay_within_the_limits_at_every_width():
    for width in WIDTHS:
        columns, size = tile_grid(width)
        assert columns >= 1
        assert size.width() >= TILE_MIN_WIDTH, width
        if columns > 1:
            assert size.width() <= TILE_MAX_WIDTH, width
        assert size.height() >= TILE_MIN_HEIGHT
        assert size.height() == max(TILE_MIN_HEIGHT, round(size.width() * TILE_ASPECT))
        if width >= TILE_MIN_WIDTH:
            assert columns * size.width() <= width, width


def test_a_single_column_fills_the_width():
    # Between one maximum tile and two minimum tiles, one tile takes the whole width.
    for width in range(TILE_MAX_WIDTH + 1, 2 * TILE_MIN_WIDTH):
        columns, size = tile_grid(width)
        assert (columns, size.width()) == (1, width), width
    assert tile_grid(TILE_MIN_WIDTH - 50)[1].width() == TILE_MIN_WIDTH


def test_width_beyond_the_maximum_adds_columns():
    previous = 0
    for width in WIDTHS:
        columns, size = tile_grid(width)
        assert columns >= previous  # widening never removes a column
        previous = columns
        if width >= TILE_MIN_WIDTH:
            # Unused width never holds another minimum tile: it goes to a new column.
            assert width - columns * size.width() < TILE_MIN_WIDTH, width
    assert tile_grid(4 * TILE_MAX_WIDTH)[0] >= 4


def _many_tiles(scientific, count: int) -> list:
    """``count`` tiles, each cluster recorded under its own member IDs."""
    result = scientific[-1]
    axes = list(scientific[0].spectral_params)
    reviews = review_clusters(result, axes, ["¹H", "¹⁵N"][: len(axes)], HighlightRules())
    base = list(cluster_tiles(result, reviews, None).values())
    tiles = []
    for n in range(count):
        tile = base[n % len(base)]
        review = replace(tile.review, cluster_id=n, ids=tuple(f"{i}#{n}" for i in tile.review.ids))
        tiles.append(replace(tile, review=review))
    return tiles


@pytest.fixture
def grid(qapp, scientific):
    view = ProfilesView()
    view.set_tiles(_many_tiles(scientific, 40), "40 clusters", "")
    view.show()
    qapp.processEvents()
    yield view
    view.close()
    view.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def _layout(view) -> tuple[int, QtCore.QSize]:
    """Columns and tile size as the user sees them: tiles sharing the first row."""
    QtWidgets.QApplication.processEvents()
    rects = [view.list.visualRect(view.model.index(row)) for row in range(view.model.rowCount())]
    first = [r for r in rects if r.top() == rects[0].top()]
    return len(first), rects[0].size()


@pytest.mark.parametrize("width", [480, 700, 1000, 1400, 2000])
def test_the_view_fills_its_width_with_tiles_within_the_limits(grid, width):
    grid.resize(width, 700)
    columns, size = _layout(grid)
    assert TILE_MIN_WIDTH <= size.width() <= TILE_MAX_WIDTH
    assert size.height() >= TILE_MIN_HEIGHT
    available = grid.list.viewport().width()
    assert columns * size.width() <= available
    assert available - columns * size.width() < TILE_MIN_WIDTH  # no room left for another


def test_a_wider_view_adds_columns_not_larger_tiles(grid):
    grid.resize(3 * TILE_MAX_WIDTH, 700)
    narrow, _ = _layout(grid)
    grid.resize(6 * TILE_MAX_WIDTH, 700)
    wide, size = _layout(grid)
    assert wide > narrow
    assert size.width() <= TILE_MAX_WIDTH


def test_reflowing_keeps_the_selected_tile_selected_and_visible(grid):
    chosen = []
    grid.cluster_chosen.connect(chosen.append)
    grid.resize(1400, 500)
    QtWidgets.QApplication.processEvents()
    target = grid.tile_ids()[33][0]
    grid.select(target)
    for width in (500, 1900, 800):
        grid.resize(width, 500)
        QtWidgets.QApplication.processEvents()
        current = grid.list.currentIndex()
        assert current.row() == 33, width
        assert grid.list.selectionModel().isSelected(current)
        assert grid.list.viewport().rect().contains(grid.list.visualRect(current)), width
    assert chosen == []  # reflowing never chooses a cluster
    assert grid.selected == target


def test_pdf_pages_keep_their_layout_at_any_view_width(grid, tmp_path):
    for width in (480, 2000):
        grid.resize(width, 600)
        QtWidgets.QApplication.processEvents()
        grid.export_figure(str(tmp_path / f"w{width}.pdf"))
        # Twelve tiles per A4 page, whatever the on-screen grid.
        assert "Exported 40 tiles on 4 pages" in grid.caption.text()


@pytest.mark.parametrize(("build", "planes"), [(cpmg_like_dataset, 20), (decay_dataset, 10)])
def test_short_series_tiles_carry_every_plane(build, planes):
    # Fixtures for the tile-size validation: short series fit and tile like CEST.
    spectra, _peaks, _config, prepared = build()
    result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    axes = list(spectra.spectral_params)
    reviews = review_clusters(result, axes, ["¹H", "¹⁵N"][: len(axes)], HighlightRules())
    tiles = cluster_tiles(result, reviews, tuple(spectra.z_values))
    assert len(tiles) == 2
    for tile in tiles.values():
        assert tile.profile is not None
        assert tile.profile.x.size == planes
        assert all(s.amplitudes.size == planes for s in tile.profile.series)


def test_what_the_bars_are_and_what_is_off_scale_share_one_note(qapp):
    _spectra, _peaks, _config, prepared = cpmg_like_dataset()
    result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    peak = result.inputs.identities[0].working_id
    shown = peak_profile(result, peak, None)
    assert isinstance(shown, PeakProfile)
    # The first plane lies far from the rest, so it is framed out; it is the current one.
    x = np.concatenate(([-12000.0], np.arange(shown.x.size - 1) * 25.0))
    panel = ProfilePlot()
    panel.show()
    panel.show_profile(replace(shown, x=x), int(shown.source_planes[0]))
    qapp.processEvents()
    note = panel.note.text()
    assert "±1 profile σ" in note
    assert "1 point off scale" in note
    assert "current plane is off scale" in note
    # One paragraph that wraps only between phrases, not a line for each kind.
    assert "<br>" not in note
    panel.close()


def test_a_screen_reader_hears_words_and_one_comparison_choice(workspace, scientific):
    """What VoiceOver reads: names, roles and keyboard routes, never bare glyphs."""
    w = workspace
    a11y = QtGui.QAccessible.queryAccessibleInterface
    # The inspector takes focus so arrows scroll it; the stop must have a name.
    assert a11y(w.panel).text(QtGui.QAccessible.Text.Name) == "Inspector"
    publish(w, scientific[-1])
    # Experimental / Overlay / Residual are one choice (spec §5), as Peaks / Clusters.
    roles = {a11y(b).role() for b in (*w.mode_buttons, *w.navigator.tab_buttons)}
    assert roles == {QtGui.QAccessible.Role.RadioButton}
    overlay = w.mode_buttons[1]
    QtTest.QTest.mouseClick(overlay, QtCore.Qt.MouseButton.LeftButton)
    QtTest.QTest.mouseClick(overlay, QtCore.Qt.MouseButton.LeftButton)
    assert [b.isChecked() for b in w.mode_buttons] == [False, True, False]
    # The warning column speaks words; ▲ and ◇ stay visual.
    tree = w.navigator.tree
    spoken = [
        tree.topLevelItem(i).data(0, QtCore.Qt.ItemDataRole.AccessibleTextRole)
        for i in range(tree.topLevelItemCount())
    ]
    assert spoken
    assert all(s and not {"▲", "◇"} & set(s) for s in spoken)
    # A tile says how to focus its cluster from the keyboard, and Return does it.
    w.show_view(profiles=True)
    wait_until(lambda: "Computing" not in w.profiles.caption.text())
    tiles = w.profiles.list
    index = w.profiles.model.index(0)
    assert "Return to focus" in index.data(QtCore.Qt.ItemDataRole.AccessibleTextRole)
    tiles.setCurrentIndex(index)
    tiles.setFocus()
    QtTest.QTest.keyClick(tiles, QtCore.Qt.Key.Key_Return)
    wait_until(lambda: w.focus_open)
