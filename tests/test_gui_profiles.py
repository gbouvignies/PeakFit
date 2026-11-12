"""The Profiles view: every cluster's profiles, in the navigator's order, sharing selection."""

from copy import deepcopy
from pathlib import Path

import pytest
from builders import two_cluster_dataset
from gui_helpers import wait_until
from PySide6 import QtCore, QtTest

from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.editor.profiles_view import cluster_tiles
from peakfit.gui.editor.review import HighlightRules, review_clusters
from peakfit.gui.editor.workspace import EditorWorkspace
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
    widget.fitting.configuration = config.model_copy(deep=True)
    widget.fitting.setup_seen = True
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
    # Spectrum tools step aside; the plane strip and selection stay shared.
    assert not any(tool.isVisible() for tool in w._spectrum_tools)
    order = [r.ids for r in w.navigator.shown_reviews()]
    assert w.profiles.tile_ids() == order
    # Sorting the cluster list by member count, largest first, reorders the tiles.
    w.navigator.tree.sortByColumn(2, QtCore.Qt.SortOrder.DescendingOrder)
    wait_until(lambda: w.profiles.tile_ids() == sorted(order, key=len, reverse=True))

    # Choosing a tile selects its cluster; a selection elsewhere marks its tile.
    last = len(w.profiles.tile_ids()) - 1
    w.profiles.list.setCurrentIndex(w.profiles.model.index(last))
    assert w.selected_id in w.profiles.tile_ids()[last]
    assert w.panel.cluster_view
    other = w.profiles.tile_ids()[0][0]
    w.select_peak(other)
    assert w.profiles.list.currentIndex().row() == 0

    w.set_plane(2)
    assert w.profiles.plane == 1


def test_a_filter_narrows_the_tiles_and_says_so(workspace, scientific):
    w = workspace
    publish(w, scientific[-1])
    w.show_view(profiles=True)
    total = len(w.profiles.tile_ids())
    assert w.profiles.caption.text().startswith(f"{total} clusters")
    w.navigator.filter.setCurrentIndex(w.navigator.filter.count() - 1)
    shown = w.navigator.shown_reviews()
    assert w.profiles.tile_ids() == [r.ids for r in shown]
    if len(shown) < total:
        assert w.profiles.caption.text().startswith(f"{len(shown)} of {total} clusters")
    if not shown:
        assert w.profiles.empty.isVisible()


def test_activating_a_tile_opens_its_cluster_in_focus(workspace, scientific):
    w = workspace
    publish(w, scientific[-1])
    w.show_view(profiles=True)
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
    w.show_view(profiles=True)
    assert "historical" not in w.profiles.caption.text()
    peak = w.peaks[0]
    w.move_peak(peak.id, tuple(p + 0.001 for p in peak.positions))
    assert not w.fitting.current
    assert f"Run {result.run_id[:8]} · historical" in w.profiles.caption.text()
