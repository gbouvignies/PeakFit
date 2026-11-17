"""Every recorded per-cluster diagnostic is findable from the Clusters list (#217).

The list's mark column, its Show menu and N / Shift+N reach the same diagnostics the
cluster, peak and profile pages already show: engine numerical warnings, the
amplitude decomposition, the correlated-profile flag (↔) and highlights. Nothing is
recomputed for the list; it reads the same engine functions as those pages.
"""

import threading
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from builders import two_cluster_dataset
from gui_helpers import wait_until
from PySide6 import QtCore, QtTest

from peakfit.engine.algorithms.covariance_factors import profile_correlations
from peakfit.engine.diagnostics.identifiability import Cause, amplitude_identifiability
from peakfit.gui import run_diagnostics
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.editor.changes import compare_runs
from peakfit.gui.editor.review import HighlightRules, ReviewMark, review_clusters
from peakfit.gui.editor.workspace import EditorWorkspace
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.peaks import WorkingEdits
from peakfit.gui.spectrum import Dataset

pytestmark = pytest.mark.gui

_ROLE = QtCore.Qt.ItemDataRole.UserRole


def _fit(spectra, peaks, config):
    prepared = prepare_dataset_fit(spectra, peaks, config.model_copy(deep=True))
    return finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))


@pytest.fixture(scope="module")
def scientific():
    """The overlapping pair (↔, no warning or highlight) and a remote peak."""
    spectra, peaks, config, prepared = two_cluster_dataset()
    result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    return spectra, peaks, config, result


def _coincident_peaks(peaks):
    return (peaks[0], replace(peaks[1], positions=peaks[0].positions), *peaks[2:])


@pytest.fixture(scope="module")
def coincident(scientific):
    """The same data with the pair's peaks placed on one position: a degenerate split."""
    spectra, peaks, config, _ = scientific
    return _fit(spectra, _coincident_peaks(peaks), config)


@pytest.fixture
def workspace(request, qapp, scientific):
    """The editor on the synthetic data; ``coincident`` places the pair on one position."""
    spectra, peaks, config, _ = scientific
    if getattr(request, "param", None) == "coincident":
        peaks = _coincident_peaks(peaks)
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


def _reviews(result, spectra):
    axes = list(spectra.spectral_params)
    return review_clusters(result, axes, [f"axis {i}" for i in range(len(axes))])


def _pair(result):
    return next(c for c in result.outcome.clusters if c.usable and len(c.peak_names) > 1)


def _rows(navigator):
    tree = navigator.tree
    return [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]


def _row(navigator, cluster_id):
    return next(r for r in _rows(navigator) if r.data(0, _ROLE).cluster_id == cluster_id)


def _options(navigator):
    return [navigator.filter.itemText(i) for i in range(navigator.filter.count())]


def _choose(navigator, prefix):
    index = next(i for i, text in enumerate(_options(navigator)) if text.strip().startswith(prefix))
    navigator.filter.setCurrentIndex(index)


def _press_n(navigator, *, backward=False):
    navigator.tree.setFocus()
    modifier = (
        QtCore.Qt.KeyboardModifier.ShiftModifier
        if backward
        else QtCore.Qt.KeyboardModifier.NoModifier
    )
    QtTest.QTest.keyClick(navigator.tree, QtCore.Qt.Key.Key_N, modifier)


# The review: the same engine functions as the pages, nothing recomputed.


def test_the_review_carries_the_correlated_profile_flag_of_each_member(scientific):
    spectra, _, _, result = scientific
    ids = {i.engine_name: i.working_id for i in result.inputs.identities}
    for review in _reviews(result, spectra):
        cluster = result.outcome.by_cluster_id[review.cluster_id]
        evaluation = cluster.analytical_evaluation
        flags = profile_correlations(evaluation.covariance_factors, evaluation.amplitudes)
        expected = tuple(
            ids[name] for name, flag in zip(cluster.peak_names, flags, strict=True) if flag.flagged
        )
        assert review.correlated == expected
    pair = next(r for r in _reviews(result, spectra) if not r.single)
    assert set(pair.correlated) == set(pair.ids)  # both members of the overlapping pair
    assert not pair.warnings
    assert not pair.highlights


def test_an_undetermined_amplitude_split_is_a_numerical_warning_naming_its_members(
    scientific, coincident
):
    spectra = scientific[0]
    cluster = _pair(coincident)
    evaluation = cluster.analytical_evaluation
    causes = {m.cause for m in amplitude_identifiability(evaluation.shapes, evaluation.amplitudes)}
    assert causes == {Cause.ILL_CONDITIONED}
    review = next(r for r in _reviews(coincident, spectra) if r.cluster_id == cluster.cluster_id)
    warning = next(w for w in review.warnings if w.kind == "Ill-conditioned shapes")
    assert set(warning.detail.split(", ")) == set(review.names)
    # The run comparison names it once, as it did before the list could show it.
    changes = compare_runs(
        coincident,
        _reviews(coincident, spectra),
        scientific[-1],
        _reviews(scientific[-1], spectra),
        list(spectra.spectral_params),
        ["axis 0", "axis 1"],
    )
    gained = next(c for c in changes.changes if set(c.names) == set(review.names)).gained
    assert gained.count("▲ Ill-conditioned shapes") == 1


# The Clusters list: marks, Show menu and counts.


def test_the_list_marks_and_filters_the_correlated_profile_flag(workspace, scientific):
    w, navigator = workspace, workspace.navigator
    result = scientific[-1]
    publish(w, result)
    pair = _pair(result)
    remote = next(c for c in result.outcome.clusters if c is not pair)
    row = _row(navigator, pair.cluster_id)
    # The pair has no warning or highlight, yet the list now says what the pages say.
    assert row.text(0) == "↔"
    assert "↔ Profile errors correlated across planes: Duplicate, Duplicate" in row.toolTip(0)
    assert "Profile errors correlated across planes" in row.data(
        0, QtCore.Qt.ItemDataRole.AccessibleTextRole
    )
    assert _row(navigator, remote.cluster_id).text(0) == ""
    assert "↔" in navigator.tree.headerItem().toolTip(0)
    # Its own Show choice with its count; never counted as a warning.
    options = _options(navigator)
    assert "↔ Profile errors correlated (1)" in options
    assert "Any warning (0)" in options
    _choose(navigator, "↔ Profile errors correlated")
    shown = [r for r in _rows(navigator) if not r.isHidden()]
    assert shown == [row]
    assert navigator.cluster_count.text().startswith("1 of 2 clusters")
    assert not w.warnings.isVisible()


def test_one_mark_rule_ranks_warning_then_correlated_profile_then_highlight(workspace, scientific):
    w, navigator = workspace, workspace.navigator
    result = scientific[-1]
    publish(w, result)
    pair = _pair(result)
    review = _row(navigator, pair.cluster_id).data(0, _ROLE)
    assert review.mark is ReviewMark.CORRELATED
    # Warning first, then the correlated profile, then a highlight.
    assert replace(review, highlights=(object(),)).mark is ReviewMark.CORRELATED
    assert replace(review, correlated=()).mark is ReviewMark.NONE
    warned = next(r for r in _reviews(_with_warning(result), w.dataset.spectra) if r.warnings)
    assert warned.mark is ReviewMark.WARNING
    assert [m.rank for m in ReviewMark] == [0, 1, 2, 3]


def _with_warning(result):
    """The run with its first usable cluster's first varied parameter at its lower bound."""
    cluster = next(c for c in result.outcome.clusters if c.usable)
    parameter = next(p for p in cluster.final_nonlinear_parameters if p.vary)
    bounded = replace(
        cluster,
        final_nonlinear_parameters=tuple(
            replace(p, value=p.min) if p is parameter else p
            for p in cluster.final_nonlinear_parameters
        ),
    )
    outcome = replace(
        result.outcome,
        clusters=tuple(bounded if c is cluster else c for c in result.outcome.clusters),
    )
    return replace(result, completed_fit=replace(result.completed_fit, outcome=outcome))


@pytest.mark.parametrize("workspace", ["coincident"], indirect=True)
def test_an_undetermined_split_is_counted_and_marked_as_a_warning(workspace, coincident):
    w, navigator = workspace, workspace.navigator
    publish(w, coincident)
    pair = _pair(coincident)
    row = _row(navigator, pair.cluster_id)
    assert row.text(0) == "▲"
    assert "▲ Ill-conditioned shapes" in row.toolTip(0)
    assert "Ill-conditioned shapes (1)" in [o.strip() for o in _options(navigator)]
    assert w.warnings.property("count") == 1
    assert "amplitudes" in w.warnings.toolTip()


# N / Shift+N follow the Show choice.


def test_n_steps_through_the_rows_the_show_choice_selects(workspace, scientific):
    w, navigator = workspace, workspace.navigator
    result = scientific[-1]
    publish(w, result)
    pair = _pair(result)
    pair_ids = {i.working_id for i in result.inputs.identities if i.engine_name in pair.peak_names}
    w.select_peak(None)
    # All clusters: N visits numerical warnings only; here there are none, and the
    # status line points to the Show menu instead of doing nothing silently.
    _choose(navigator, "All clusters")
    _press_n(navigator)
    assert w.selected_id is None
    assert "Show" in w.status.text()
    # ↔ chosen: N reaches the flagged cluster, Shift+N too, and nothing hidden.
    _choose(navigator, "↔ Profile errors correlated")
    _press_n(navigator)
    assert w.selected_id in pair_ids
    assert "N: next" in navigator.cluster_count.text()
    w.select_peak(None)
    _press_n(navigator, backward=True)
    assert w.selected_id in pair_ids


def test_n_with_a_warning_filter_still_visits_only_warned_rows(workspace, scientific):
    w, navigator = workspace, workspace.navigator
    publish(w, _with_warning(scientific[-1]))
    _choose(navigator, "Any warning")
    shown = [r for r in _rows(navigator) if not r.isHidden()]
    assert shown
    visited = []
    for _ in shown:
        _press_n(navigator)
        visited.append(navigator._clusters[w.selected_id])
    assert visited == shown


def test_n_with_a_show_choice_steps_through_its_shown_rows_in_order(workspace, scientific):
    w, navigator = workspace, workspace.navigator
    publish(w, scientific[-1])
    w.set_rules(HighlightRules(0.0, 1.0, 1.0))  # every cluster highlighted
    _choose(navigator, "Any highlight")
    rows = [r for r in _rows(navigator) if not r.isHidden()]
    assert len(rows) == 2
    visited = []
    for _ in rows:
        _press_n(navigator)
        visited.append(navigator._clusters[w.selected_id])
    assert visited == rows
    _press_n(navigator, backward=True)
    assert navigator._clusters[w.selected_id] is rows[0]
    # A row the choice hides is never reached.
    rows[1].setHidden(True)
    for _ in range(3):
        _press_n(navigator)
        assert navigator._clusters[w.selected_id] is rows[0]


def test_the_run_diagnostics_are_computed_off_the_gui_thread(workspace, scientific, monkeypatch):
    # About 0.3 s on the bundled example: publication computes them in its worker.
    threads = []
    original = run_diagnostics.member_diagnostics

    def recorded(cluster):
        threads.append(threading.current_thread())
        return original(cluster)

    monkeypatch.setattr("peakfit.gui.dataset_fit.member_diagnostics", recorded)
    spectra, peaks, config, _ = scientific
    result = _fit(spectra, peaks, config)
    publish(workspace, result)
    assert threads
    assert threading.main_thread() not in threads
    assert "diagnostics" in vars(result)
    count = len(threads)
    workspace.navigator.filter.setCurrentIndex(0)
    workspace.reviews()
    assert len(threads) == count  # views only read the cached diagnostics
