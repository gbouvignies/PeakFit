"""Protect the fit → optimized working state → native comparison → refit loop."""

import csv
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import MappingProxyType

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6 import QtCore, QtWidgets
from test_gui_dataset import _dataset
from test_gui_editor import wait_for

import peakfit.gui.editor.canvas as canvas_module
import peakfit.gui.editor.review as review_module
from peakfit.engine.algorithms.evaluation import FitOutcomeClassification
from peakfit.gui import formatting
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.editor import fitting as fitting_module
from peakfit.gui.editor.profile import (
    Normalized,
    PeakProfile,
    ProfilePlot,
    framed_range,
    normalize,
    parse_planes,
    peak_profile,
)
from peakfit.gui.editor.setup import DEFAULT_SOURCE
from peakfit.gui.editor.window import EditorWindow
from peakfit.gui.editor.workspace import EditorWorkspace
from peakfit.gui.fit_display import dataset_model_plane
from peakfit.gui.fit_export import export_fit_result
from peakfit.gui.fit_process import FitProcessResult
from peakfit.gui.spectrum import Dataset
from peakfit.io.scientific import read_scientific

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def scientific():
    spectra, peaks, config, prepared = _dataset()
    result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    return spectra, peaks, config, prepared, result


@pytest.fixture
def workspace(scientific):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    spectra, peaks, config, _, _ = scientific
    dataset = Dataset(Path("synthetic.ft2"), spectra, peaks, (), True, 5.0, 0.01)
    widget = EditorWorkspace(dataset)
    widget.fitting.configuration = config.model_copy(deep=True)
    widget.fitting.setup_seen = True
    widget.resize(1280, 750)
    widget.show()
    app.processEvents()
    yield widget
    widget.stop()
    widget.close()
    widget.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    app.processEvents()


def publish(workspace, result):
    fitting = workspace.fitting
    fitting._submitted_revision = workspace.edits.revision
    fitting.process.state = "completed"
    fitting._finished(0, FitProcessResult(result, 0, 0, 0, 0))
    wait_for(lambda: not fitting.busy)
    assert fitting.result is result, fitting.failure


def test_atomic_completion_comparison_draft_edit_and_export(workspace, scientific, tmp_path):
    w, c = workspace, workspace.canvas
    result = scientific[-1]
    w.select_peak(w.peaks[0].id)
    following = c.pin_peak_id
    ranges = c.main.getViewBox().viewRange()
    seeds = w.peaks
    publish(w, result)
    assert w.fitting.current
    assert w.peaks == w.fitting.completion.working_peaks
    assert result.inputs.snapshot.peaks == seeds
    assert not w.edits.can_undo
    assert c.pin_peak_id == following
    assert c.selected_id == following
    np.testing.assert_allclose(c.main.getViewBox().viewRange(), ranges)
    assert c.comparison_mode == 1
    wait_for(lambda: not c.contour_pending)
    authoritative = dataset_model_plane(result, c.spectra.spectral_params, c.plane, cancel=Event())
    np.testing.assert_array_equal(c._model.values, authoritative.values)
    w.set_mode(2)
    wait_for(lambda: not c.contour_pending)
    residual = (c.spectra.data[c.plane] - authoritative.values) / result.outcome.noise
    np.testing.assert_array_equal(c.curve_x.getData()[1], residual[c.iy, :])
    np.testing.assert_array_equal(c.curve_y.getData()[0], residual[:, c.ix])
    jobs = c.contour_jobs
    limit = c.trace_limit
    for _ in range(500):
        c.sample(float(c.ppm_x[12]), float(c.ppm_y[13]))
    assert c.contour_jobs == jobs
    assert c.trace_limit == limit
    # A draft never invalidates accepted science; one commit does, including Undo.
    w.inspector.x_field.setText("8.123456789")
    assert w.fitting.current
    assert c.comparison_mode == 2
    w.inspector.show_peak(w.peaks[0], c.display(w.peaks[0]))
    peak = w.peaks[0]
    w.move_peak(peak.id, (peak.positions[0], peak.positions[1] + 0.00001))
    assert not w.fitting.current
    assert c.comparison_mode == 0
    w.history(False)
    assert w.peaks == w.fitting.completion.working_peaks
    assert not w.fitting.current
    path = export_fit_result(
        result, tmp_path / "result", completion=w.fitting.completion, current=w.fitting.current
    )
    outcome, _samples, context = read_scientific(path / "scientific.npz")
    assert outcome.n_optimizer_passes == 2
    assert context["export_state"]["currentness"] == "outdated"
    assert context["peak_identities"] == [
        {"engine_name": i.engine_name, "peak_id": i.working_id, "assignment": i.assignment}
        for i in result.inputs.identities
    ]


def test_repeat_actual_process_cancel_failure_and_fresh_masks(workspace):
    w = workspace
    w.fitting.fit()
    wait_for(lambda: not w.fitting.busy, timeout=20)
    assert w.fitting.current, w.fitting.failure
    previous = w.fitting.result
    seeds = w.peaks
    # Cancellation never publishes partial scientific values.
    w.fitting.fit()
    assert w.fit_busy
    w.delete_peaks()
    assert w.peaks == seeds
    w.fitting.cancel()
    wait_for(lambda: not w.fit_busy)
    assert w.fitting.result is previous
    assert w.peaks == seeds
    w.fitting.process.state = "failed"
    w.fitting._finished(0, "simulated child crash")
    assert w.fitting.result is previous
    assert w.fitting.current
    w.fitting.fit()
    wait_for(lambda: not w.fit_busy, timeout=20)
    assert w.fitting.current, w.fitting.failure
    assert w.fitting.result.inputs.snapshot.peaks == seeds
    assert w.fitting.result.inputs is not previous.inputs
    for _ in range(20):
        seeds = w.peaks
        w.fitting.fit()
        wait_for(lambda: not w.fit_busy, timeout=20)
        assert w.fitting.current, w.fitting.failure
        assert w.fitting.result.inputs.snapshot.peaks == seeds


def test_partial_usability_nonconvergence_and_free_pin(workspace, scientific):
    w, c = workspace, workspace.canvas
    result = scientific[-1]
    good, bad = result.outcome.clusters
    good = replace(good, optimizer_provenance=replace(good.optimizer_provenance, converged=False))
    bad = replace(
        bad,
        classification=FitOutcomeClassification.UNUSABLE,
        analytical_evaluation=None,
        unusable_reason="synthetic unusable",
    )
    clusters = (good, bad)
    mixed = replace(
        result,
        completed_fit=replace(
            result.completed_fit,
            outcome=replace(
                result.outcome,
                clusters=clusters,
                by_cluster_id=MappingProxyType({c.cluster_id: c for c in clusters}),
            ),
        ),
    )
    c.sample(float(c.ppm_x[7]), float(c.ppm_y[8]))
    c.set_pinned(True)
    original = (c.ix, c.iy)
    seeds = {p.id: p.positions for p in w.peaks}
    publish(w, mixed)
    assert w.fitting.current
    assert (c.ix, c.iy) == original
    assert c.pin_peak_id is None
    assert w.fitting.completion.warnings
    identities = {i.engine_name: i.working_id for i in mixed.inputs.identities}
    for name in bad.peak_names:
        assert (
            next(p for p in w.peaks if p.id == identities[name]).positions
            == seeds[identities[name]]
        )
    assert "Incomplete model" in w.summary.text()
    w.fitting.next_warning()
    assert "Not converged" in _review_text(w)
    w.fitting.next_warning()
    assert "synthetic unusable" in _review_text(w)
    # The navigator lists every recorded cluster and marks engine warnings.
    flagged = [
        w.navigator.tree.topLevelItem(i)
        for i in range(w.navigator.tree.topLevelItemCount())
        if w.navigator.tree.topLevelItem(i).text(0)
    ]
    assert w.navigator.tree.topLevelItemCount() == len(mixed.outcome.clusters)
    assert len(flagged) == 2


def test_latest_plane_wins_and_cache_survives_modes(workspace, scientific):
    w, c = workspace, workspace.canvas
    publish(w, scientific[-1])
    for _ in range(40):
        w.set_plane(_ % 3 + 1)
    w.set_plane(3)
    wait_for(lambda: not c.contour_pending)
    assert c.plane == 2
    expected = dataset_model_plane(scientific[-1], c.spectra.spectral_params, 2, cancel=Event())
    np.testing.assert_array_equal(c._model.values, expected.values)
    model = c._model
    w.set_mode(0)
    w.set_mode(2)
    wait_for(lambda: not c.contour_pending)
    assert c._model is model
    assert [layer.role for layer in c.paths] == ["residual"]


def test_all_unusable_completion_and_invalid_replacement(workspace, scientific):
    w = workspace
    result = scientific[-1]
    clusters = tuple(
        replace(
            c,
            classification=FitOutcomeClassification.UNUSABLE,
            analytical_evaluation=None,
            unusable_reason="unusable fixture",
        )
        for c in result.outcome.clusters
    )
    unusable = replace(
        result,
        completed_fit=replace(
            result.completed_fit,
            outcome=replace(
                result.outcome,
                clusters=clusters,
                by_cluster_id=MappingProxyType({c.cluster_id: c for c in clusters}),
            ),
        ),
    )
    seeds = w.peaks
    publish(w, unusable)
    assert w.peaks == seeds
    assert w.fitting.current
    assert "No usable model" in w.summary.text()
    assert not w.mode_buttons[1].isEnabled()
    assert w.canvas.comparison_mode == 0
    invalid = replace(
        result, inputs=replace(result.inputs, identities=result.inputs.identities[:-1])
    )
    w.fitting._finished(0, FitProcessResult(invalid, 0, 0, 0, 0))
    wait_for(lambda: not w.fit_busy)
    assert w.fitting.result is unusable
    assert w.peaks == seeds
    assert "stable-ID mapping" in w.fitting.failure


@pytest.mark.parametrize("mode", [1, 2])
def test_pending_comparison_keeps_current_experiment(workspace, scientific, monkeypatch, mode):
    w, c = workspace, workspace.canvas
    # Prime experimental geometry for this plane before requesting an uncached model.
    w.set_plane(3)
    wait_for(lambda: not c.contour_pending)
    entered, release = Event(), Event()
    project = canvas_module.dataset_model_plane

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return project(*args, **kwargs)

    monkeypatch.setattr(canvas_module, "dataset_model_plane", delayed)
    try:
        publish(w, scientific[-1])
        wait_for(entered.is_set)
        if mode == 2:
            w.set_mode(2)
        assert c.contour_pending
        assert c._model is None
        assert [layer.role for layer in c.paths] == ["measured"], (
            "Current-plane experimental contours must remain visible"
        )
        assert "Experimental" in c.pending_label.toPlainText()
        assert ("Residual" if mode == 2 else "Overlay") in c.pending_label.toPlainText()
        if mode == 2:
            assert c.curve_x.getData()[1] is None
    finally:
        release.set()
    wait_for(lambda: not c.contour_pending)


def test_result_export_preserves_adopted_peaks_for_close(
    workspace, scientific, monkeypatch, tmp_path
):
    w = workspace
    publish(w, scientific[-1])
    assert w.edits.dirty
    monkeypatch.setattr(fitting_module, "choose_and_export", lambda *_a, **_k: None)
    assert not w.fitting.export()
    assert w.edits.dirty
    monkeypatch.setattr(fitting_module, "choose_and_export", lambda *_a, **_k: tmp_path)
    assert w.fitting.export()
    assert not w.edits.dirty
    assert w.fitting.current
    window = EditorWindow()
    window.workspace = w

    # Canonical export preserves evidence, while Save analysis preserves the session.
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda _dialog: 0)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "clickedButton",
        lambda dialog: next(b for b in dialog.buttons() if b.text() == "Cancel"),
    )
    assert not window.protect_edits()
    # Further manual changes are still unpreserved, including when an old run is exported.
    peak = w.peaks[0]
    w.move_peak(peak.id, (peak.positions[0], peak.positions[1] + 0.00001))
    assert w.edits.dirty
    assert not w.fitting.current
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    assert w.fitting.export()
    assert w.edits.dirty
    assert not w.fitting.current
    w.plane_values_modified = True
    assert w.has_changes
    window.workspace = None
    window.close()


def test_pending_latest_plane_and_run_reject_stale_delivery(workspace, scientific, monkeypatch):
    w, c = workspace, workspace.canvas
    result = scientific[-1]
    publish(w, result)
    wait_for(lambda: not c.contour_pending)
    old_model = c._model
    entered, release = Event(), Event()
    project = canvas_module.dataset_model_plane

    def delayed(result, axes, plane, *, cancel):
        if plane == 1:
            entered.set()
            assert release.wait(3)
        return project(result, axes, plane, cancel=cancel)

    monkeypatch.setattr(canvas_module, "dataset_model_plane", delayed)
    try:
        # No cached experimental contours for this plane: prepare those first,
        # then keep them visible while the model calculation is held pending.
        w.set_mode(2)
        w.set_plane(2)
        wait_for(entered.is_set)
        assert [layer.role for layer in c.paths] == ["measured"]
        assert c._displaying_experiment
        assert c._model is None
        assert c.curve_x.getData()[1] is None
        obsolete = c._revision
        w.set_plane(3)
        # The fallback must never be carried across acquisition indices.
        assert not c.paths
        assert c._model is None
    finally:
        release.set()
    wait_for(lambda: not c.contour_pending)
    expected = project(result, c.spectra.spectral_params, 2, cancel=Event())
    np.testing.assert_array_equal(c._model.values, expected.values)
    current_model = c._model
    c._contours_ready("model", obsolete, (old_model, []))
    assert c._model is current_model
    # Replacing a run on the same plane can reuse only experimental geometry.
    obsolete = c._revision
    c.set_comparison(replace(result, run_id="replacement"), 2)
    assert c._model is None
    assert c._displaying_experiment
    assert [layer.role for layer in c.paths] == ["measured"]
    c._contours_ready("model", obsolete, (current_model, []))
    assert c._model is None
    wait_for(lambda: not c.contour_pending)
    assert all(key[0] == "replacement" for key in c._model_cache)
    residual = (c.spectra.data[2] - c._model.values) / result.outcome.noise
    np.testing.assert_array_equal(c.curve_x.getData()[1], residual[c.iy])


@pytest.fixture
def settings(tmp_path):
    """Keep application preferences out of the developer's own settings."""
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
    QtCore.QSettings.setPath(
        QtCore.QSettings.Format.IniFormat, QtCore.QSettings.Scope.UserScope, str(tmp_path)
    )
    QtCore.QSettings().clear()
    yield QtCore.QSettings()
    QtCore.QSettings().clear()
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.NativeFormat)


def _warned(result):
    """Attach an engine-style uncertainty warning to the first usable cluster."""
    cluster, *rest = result.outcome.clusters
    warning = "Unreliable local uncertainties: synthetic overlap."
    cluster = replace(
        cluster,
        analytical_evaluation=replace(cluster.analytical_evaluation, uncertainty_warning=warning),
    )
    clusters = (cluster, *rest)
    outcome = replace(
        result.outcome,
        clusters=clusters,
        by_cluster_id=MappingProxyType({c.cluster_id: c for c in clusters}),
    )
    return replace(result, completed_fit=replace(result.completed_fit, outcome=outcome))


def _review_text(w) -> str:
    page = w.panel.cluster_page if w.panel.cluster_view else w.panel.peak_page
    return " ".join(label.text() for label in page.findChildren(QtWidgets.QLabel))


def _selected_points(panel):
    # The selected peak is drawn last, above its cluster members.
    return [i for i in panel.plot.getPlotItem().items if isinstance(i, pg.ScatterPlotItem)][-1]


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([-12000.0, *np.arange(-1800.0, 1475.0, 25.0)], (-1800.0, 1450.0)),
        ([-12000.0, -11000.0, *range(10)], (0.0, 9.0)),
        ([5.0, -12000.0, 3.0, 1.0, 2.0, 4.0], (1.0, 5.0)),
        (list(range(10)), None),
        ([0.0, 0.0, 1.0, 1.0, 2.0, 2.0], None),
        ([-1000.0, *range(10), 1000.0], None),
        ([3.0], None),
    ],
)
def test_profile_frames_only_a_distant_minority(values, expected):
    assert framed_range(np.asarray(values)) == expected


def test_profile_shows_recorded_run_values_and_follows_selection(workspace, scientific, tmp_path):
    w, panel = workspace, workspace.panel.profile
    result = scientific[-1]
    publish(w, result)
    cluster = next(c for c in result.outcome.clusters if c.usable and len(c.peak_names) > 1)
    identities = {i.engine_name: i.working_id for i in result.inputs.identities}
    names = {i.engine_name: i.assignment for i in result.inputs.identities}
    w.select_peak(identities[cluster.peak_names[1]])
    profile = panel._profile
    evaluation = cluster.analytical_evaluation
    assert profile is not None
    assert evaluation is not None
    selected = profile.series[0]
    np.testing.assert_array_equal(selected.amplitudes, evaluation.amplitudes[1])
    assert selected.error == evaluation.scaled_amplitude_standard_errors[1]
    members = {s.assignment for s in profile.series}
    assert members == {names[n] for n in cluster.peak_names}
    snapshot = result.inputs.snapshot
    planes = np.asarray(snapshot.source_plane_indices)
    np.testing.assert_array_equal(profile.source_planes, planes)
    np.testing.assert_array_equal(profile.x, np.asarray(snapshot.plane_values)[planes])
    # The canonical CSV agrees within its own six-decimal precision.
    export = export_fit_result(result, tmp_path / "export", completion=w.fitting.completion)
    with next(export.rglob("intensities.csv")).open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["peak_name"] == cluster.peak_names[1]]
    np.testing.assert_allclose(
        [float(r["intensity"]) for r in rows], selected.amplitudes, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        [float(r["intensity_err"]) for r in rows], selected.error, rtol=1e-6, atol=1e-6
    )
    w.select_peak(identities[cluster.peak_names[0]])
    assert panel._profile is not None
    assert {s.assignment for s in panel._profile.series} == members
    points = _selected_points(panel)
    points.sigClicked.emit(points, [points.points()[2]], None)
    assert w.canvas.plane == planes[2]
    # Outdated evidence stays visible and labeled; a later peak was not in the run.
    peak = next(p for p in w.peaks if p.id == identities[cluster.peak_names[0]])
    w.move_peak(peak.id, (peak.positions[0], peak.positions[1] + 1e-5))
    assert "Outdated" in w.panel.status.text()
    assert panel._profile is not None
    w.place_peak(peak.positions)
    assert f"Not part of run {result.run_id[:8]}" in w.panel.status.text()


def test_profile_follows_only_accepted_selection(workspace, scientific):
    w = workspace
    result = scientific[-1]
    publish(w, result)
    first, second = w.peaks[0], w.peaks[1]
    w.select_peak(first.id)
    title, revision = w.panel.title.text(), w.edits.revision
    w.inspector.x_field.setText("not a number")
    w.select_peak(second.id)
    assert w.selected_id == first.id
    assert w.panel.title.text() == title
    assert w.edits.revision == revision
    # A valid draft is committed exactly once by the existing selection rule.
    w.inspector.x_field.setText(repr(first.positions[1] + 0.001))
    w.select_peak(second.id)
    assert w.selected_id == second.id
    assert w.edits.revision == revision + 1
    assert w.panel.title.text().startswith(second.name)
    assert w.edits.undo_label == f"move {first.name}"
    w.edits.undo()
    assert not w.edits.can_undo


def test_profile_keeps_repeated_plane_values_distinct(scientific):
    result = scientific[-1]
    snapshot = result.inputs.snapshot
    repeated = (1.0, 1.0, 2.0)
    run = replace(
        result, inputs=replace(result.inputs, snapshot=replace(snapshot, plane_values=repeated))
    )
    peak = result.inputs.identities[0].working_id
    numbered = peak_profile(run, peak, None)
    assert isinstance(numbered, PeakProfile)
    np.testing.assert_array_equal(numbered.x, np.asarray(snapshot.source_plane_indices) + 1)
    assert not numbered.plane_values
    changed = peak_profile(run, peak, (1.0, 1.0, 3.0))
    assert isinstance(changed, PeakProfile)
    assert not changed.plane_values
    panel = ProfilePlot()
    requested = []
    panel.plane_requested.connect(requested.append)
    shown = peak_profile(run, peak, repeated)
    assert isinstance(shown, PeakProfile)
    panel.show_profile(shown, 0)
    assert panel._profile is not None
    np.testing.assert_array_equal(panel._profile.x, repeated)
    points = _selected_points(panel)
    points.sigClicked.emit(points, points.points()[:2], None)
    actions = panel._chooser.actions()
    assert [a.text() for a in actions] == ["Plane 1 · value 1", "Plane 2 · value 1"]
    actions[1].trigger()
    assert requested == [2]


@pytest.mark.usefixtures("settings")
def test_compact_header_and_inspector_at_minimum_window(scientific, monkeypatch):
    app = QtWidgets.QApplication.instance()
    assert app is not None
    spectra, peaks, config, _, result = scientific
    window = EditorWindow()
    monkeypatch.setattr(window, "protect_edits", lambda **_: True)
    window.resize(1040, 720)
    window.show()
    name = "a-deliberately-long-dataset-name-for-the-compact-header-" * 2 + ".ft2"
    window.install_dataset(Dataset(Path(name), spectra, peaks, (), True, 5.0, 0.01))
    w = window.workspace
    assert w is not None
    w.fitting.configuration = config.model_copy(deep=True)
    w.fitting.setup_seen = True
    app.processEvents()
    publish(w, _warned(result))
    warned = result.outcome.clusters[0].peak_names[0]
    w.select_peak(next(i.working_id for i in result.inputs.identities if i.engine_name == warned))
    peak = next(p for p in w.peaks if p.id == w.selected_id)
    w.move_peak(peak.id, (peak.positions[0], peak.positions[1] + 1e-5))
    w.fitting._preparing = True
    w.fitting.attempt = "Preparing complete dataset"
    w.fitting.update()
    app.processEvents()

    def rect(widget):
        return QtCore.QRect(widget.mapTo(window, QtCore.QPoint(0, 0)), widget.size())

    buttons = [w.cancel_fit, w.inspector_button, w.peaks_button, w.add_button, w.fit_button]
    assert all(b.isVisible() and window.rect().contains(rect(b)) for b in buttons)
    # Settings cannot change during a fit; their chip yields its room to progress.
    assert not w.settings_button.isVisible()
    for label in (w.title, w.summary):
        assert not any(rect(label).intersects(rect(b)) for b in buttons)
    assert "Preparing complete dataset" in w.summary.displayed_text()
    w.fitting._preparing = False
    w.fitting.update()
    app.processEvents()
    buttons = [w.inspector_button, w.peaks_button, w.add_button, w.fit_button, w.settings_button]
    assert all(b.isVisible() and window.rect().contains(rect(b)) for b in buttons)
    for label in (w.title, w.summary):
        assert not any(rect(label).intersects(rect(b)) for b in buttons)
    shown = w.summary.displayed_text()
    assert "Outdated" in shown
    # Numerical warnings keep their own unelided label in the one-row toolbar.
    assert "numerical warning" in w.warnings.toolTip()
    assert w.warnings.text().startswith("▲")
    assert w.warnings.isVisible()
    assert window.rect().contains(rect(w.warnings))
    # The inspector names the selected peak's run state and its engine warning.
    assert "Outdated" in w.panel.status.text()
    assert "Uncertainty unreliable" in _review_text(w)
    assert w.panel.profile.isVisible()
    assert window.rect().contains(rect(w.panel.profile.plot))
    # Three columns never squeeze the spectrum below its minimum.
    assert w.canvas.main.width() >= w.canvas.main.minimumWidth()
    assert w.canvas.main.height() >= w.canvas.main.minimumHeight()
    window.close()
    window.deleteLater()


def test_profile_keeps_amplitudes_when_uncertainty_is_unavailable(scientific):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    result = scientific[-1]
    cluster = next(c for c in result.outcome.clusters if c.usable and len(c.peak_names) > 1)
    evaluation = cluster.analytical_evaluation
    assert evaluation is not None
    # A usable but poorly identifiable cluster can report NaN amplitude errors.
    errors = evaluation.scaled_amplitude_standard_errors.copy()
    errors[0] = np.nan
    weak = replace(
        cluster,
        analytical_evaluation=replace(evaluation, scaled_amplitude_standard_errors=errors),
    )
    clusters = tuple(weak if c is cluster else c for c in result.outcome.clusters)
    outcome = replace(
        result.outcome,
        clusters=clusters,
        by_cluster_id=MappingProxyType({c.cluster_id: c for c in clusters}),
    )
    run = replace(result, completed_fit=replace(result.completed_fit, outcome=outcome))
    peak = next(i.working_id for i in run.inputs.identities if i.engine_name == weak.peak_names[0])
    panel = ProfilePlot()
    panel.resize(320, 260)
    panel.show()
    shown = peak_profile(run, peak, None)
    assert isinstance(shown, PeakProfile)
    panel.show_profile(shown, 0)
    app.processEvents()
    assert not panel.plot.grab().isNull()
    assert panel._profile is not None
    np.testing.assert_array_equal(panel._profile.series[0].amplitudes, evaluation.amplitudes[0])
    assert np.isnan(panel._profile.series[0].error)
    bars = [i for i in panel.plot.getPlotItem().items if isinstance(i, pg.ErrorBarItem)]
    assert len(bars) == len(weak.peak_names) - 1
    assert np.all(np.isfinite(panel.plot.getViewBox().viewRange()))
    assert "σ unavailable" in panel.legend.text()
    assert "σ unavailable" in panel.legend.text()
    panel.close()


def test_navigator_clusters_and_inspector_show_recorded_values(workspace, scientific):
    w = workspace
    result = scientific[-1]
    assert not w.navigator.tab_buttons[1].isEnabled()
    publish(w, result)
    # The first result opens the cluster list; one row per recorded cluster.
    assert w.navigator.pages.currentIndex() == 1
    tree = w.navigator.tree
    assert tree.topLevelItemCount() == len(result.outcome.clusters)
    cluster = next(c for c in result.outcome.clusters if c.usable and len(c.peak_names) > 1)
    identities = {i.engine_name: i.working_id for i in result.inputs.identities}
    member = identities[cluster.peak_names[1]]
    row = w.navigator._clusters[member]
    evaluation = cluster.analytical_evaluation
    assert evaluation is not None
    assert row.text(2) == f"{evaluation.statistics.reduced_chi_squared:.2f}"
    # Choosing a cluster row selects one of its members, never another peak.
    tree.setCurrentItem(row)
    assert w.selected_id in {identities[n] for n in cluster.peak_names}
    w.select_peak(member)
    assert tree.currentItem() is row
    # Fitted values are the recorded parameters, formatted to their σ.
    parameter = next(
        p
        for p in cluster.final_nonlinear_parameters
        if p.name.startswith(cluster.peak_names[1] + ".")
    )
    value, sigma = formatting.with_sigma(parameter.value, parameter.standard_error)
    assert f"{value} ± {sigma}" in _review_text(w)
    assert w.edits.revision == 0


def test_dense_profile_click_opens_the_nearest_plane(scientific):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    result = scientific[-1]
    peak = result.inputs.identities[0].working_id
    shown = peak_profile(result, peak, None)
    assert isinstance(shown, PeakProfile)
    panel = ProfilePlot()
    panel.resize(320, 260)
    panel.show()
    panel.show_profile(shown, 0)
    app.processEvents()
    requested = []
    panel.plane_requested.connect(requested.append)
    points = _selected_points(panel)
    spots = points.points()[:3]
    target = spots[1].pos()

    class Click:
        def pos(self):
            return target

    # Several points under the pointer: the nearest one wins, with no chooser.
    points.sigClicked.emit(points, spots, Click())
    assert requested == [int(shown.source_planes[1]) + 1]


def test_selected_cluster_members_are_marked_without_changing_view_or_scope(workspace, scientific):
    w, c = workspace, workspace.canvas
    result = scientific[-1]
    publish(w, result)
    cluster = next(c for c in result.outcome.clusters if c.usable and len(c.peak_names) > 1)
    identities = {i.engine_name: i.working_id for i in result.inputs.identities}
    members = {identities[n] for n in cluster.peak_names}
    selected = identities[cluster.peak_names[0]]
    w.select_peak(selected)
    ranges, mode = c.main.getViewBox().viewRange(), c.comparison_mode
    model, jobs = c.comparison_result, c.contour_jobs
    # Choosing the cluster row keeps the selected member and marks the others.
    w.navigator.tree.setCurrentItem(w.navigator._clusters[selected])
    assert w.selected_id == selected
    assert set(c.cluster_ids) == members
    marked = {(round(x, 9), round(y, 9)) for x, y in zip(*c.cluster_rings.getData(), strict=True)}
    others = {
        tuple(round(v, 9) for v in c.display(p))
        for p in w.peaks
        if p.id in members and p.id != selected
    }
    assert marked == others
    # Highlighting changes neither the viewport, the comparison nor contour work.
    assert c.main.getViewBox().viewRange() == ranges
    assert (c.comparison_mode, c.comparison_result, c.contour_jobs) == (mode, model, jobs)
    # A peak outside the run has no recorded cluster to mark.
    w.place_peak(w.peaks[0].positions)
    assert c.cluster_ids == ()


def _tinted(c, alpha: float) -> set[tuple[int, int]]:
    image = c.region_image.image
    rows, columns = np.nonzero(image[..., 3] == round(255 * alpha))
    return set(zip(rows.tolist(), columns.tolist(), strict=True))


def test_first_fit_reveals_the_setup_then_runs_the_prepared_inputs(workspace, monkeypatch):
    w, fitting = workspace, workspace.fitting
    fitting.setup_seen = False
    w.select_peak(w.peaks[0].id)
    w.inspector_button.setChecked(False)
    w.toggle_inspector()
    fitting.fit()
    # Revealed, not run: the inspector opens at the setup with nothing selected.
    assert not fitting.busy
    assert fitting.result is None
    assert w.selected_id is None
    assert w.panel.isVisible()
    assert w.panel.analysis_page.isVisible()
    wait_for(lambda: fitting.prepared is not None)
    submitted = []
    monkeypatch.setattr(fitting, "start", submitted.append)
    fitting.fit()
    # The second Fit submits exactly the preparation the setup describes.
    assert submitted == [fitting.prepared]


def test_fit_is_disabled_with_the_preparation_reason(workspace):
    w, fitting = workspace, workspace.fitting
    wait_for(lambda: fitting.prepared is not None)
    invalid = fitting.configuration.model_copy(deep=True)
    invalid.exclude_planes = [1]
    fitting.configure(invalid)
    assert fitting.prepared is None
    wait_for(lambda: bool(fitting.blocked))
    assert "exclusions are unsupported" in fitting.blocked
    assert not w.fit_button.isEnabled()
    assert fitting.blocked in w.fit_button.toolTip()
    assert w.panel.setup.blocked.isVisible()
    assert fitting.blocked in w.panel.setup.blocked.text()
    w.select_peak(w.peaks[0].id)
    fitting.fit()
    assert not fitting.busy
    assert w.selected_id is None
    valid = invalid.model_copy(deep=True)
    valid.exclude_planes = []
    fitting.configure(valid)
    wait_for(lambda: fitting.prepared is not None)
    assert w.fit_button.isEnabled()
    assert not fitting.blocked


def test_setup_edits_mark_the_run_outdated_and_prepare_afresh(workspace, scientific):
    w, fitting, setup = workspace, workspace.fitting, workspace.panel.setup
    publish(w, scientific[-1])
    assert fitting.current
    setup.passes.setValue(3)
    assert fitting.configuration.fitting.refine_iterations == 3
    assert not fitting.current
    assert fitting.prepared is None
    wait_for(lambda: fitting.prepared is not None)
    assert '"refine_iterations":3' in fitting.prepared.snapshot.configuration
    setup.lineshape.setCurrentIndex(setup.lineshape.findData("gaussian"))
    assert fitting.configuration.fitting.lineshape == "gaussian"
    assert "Gaussian · explicit" in setup.shapes.text()


def test_fitting_regions_are_exact_hidden_by_default_and_labelled(workspace, scientific):
    w, c, fitting = workspace, workspace.canvas, workspace.fitting
    wait_for(lambda: fitting.prepared is not None)
    w._update_inspector()
    prepared = fitting.prepared
    # Member rings are the default cue; sampled regions appear only on request.
    assert not c.regions_visible
    assert not c.region_image.isVisible()
    assert "fitting regions" not in c.legend.text()
    w.panel.setup.regions.setChecked(True)
    assert c.region_image.isVisible()
    assert "fitting regions · preview for current settings" in c.legend.text()
    cluster = next(cl for cl in prepared.clusters if len(cl.peaks) > 1)
    ids = {i.engine_name: i.working_id for i in prepared.identities}
    w.select_peak(ids[cluster.peaks[0].name])
    view, mode, jobs = c.main.getViewBox().viewRange(), c.comparison_mode, c.contour_jobs
    w._update_inspector()
    # One tinted cell per sampled point, exactly; the selected cluster is stronger.
    samples = sum(len(cl.grid_indices[0]) for cl in prepared.clusters)
    selected = set(zip(*(np.asarray(i).tolist() for i in cluster.grid_indices), strict=True))
    assert _tinted(c, canvas_module.SELECTED_REGION_ALPHA) == selected
    assert len(_tinted(c, canvas_module.REGION_ALPHA)) == samples - len(selected)
    # Each cell is centred on its sample's ppm coordinates.
    row, column = next(iter(selected))
    centre = c.region_image.mapToParent(QtCore.QPointF(column + 0.5, row + 0.5))
    assert centre.x() == pytest.approx(c.ppm_x[column])
    assert centre.y() == pytest.approx(c.ppm_y[row])
    assert c.region_image.zValue() < 0  # behind the contours
    assert (c.comparison_mode, c.contour_jobs) == (mode, jobs)
    assert view == c.main.getViewBox().viewRange()
    # After a fit the recorded regions replace the preview, with the run named.
    result = scientific[-1]
    publish(w, result)
    w._update_inspector()
    assert c.region_source == f"recorded in run {result.run_id[:8]}"
    # A committed edit hides the preparation at once: no obsolete region is shown.
    peak = w.peaks[0]
    w.move_peak(peak.id, (peak.positions[0], peak.positions[1] + 0.001))
    assert not fitting.current
    assert fitting.prepared is None
    assert c.region_source == ""
    assert not np.any(c.region_image.image[..., 3])
    wait_for(lambda: fitting.prepared is not None)
    w._update_inspector()
    assert c.region_source == "preview for current settings"
    w.panel.setup.regions.setChecked(False)
    assert not c.region_image.isVisible()
    assert "fitting regions" not in c.legend.text()


def test_default_lineshape_fallback_is_a_caution_not_a_block(workspace):
    w, fitting, setup = workspace, workspace.fitting, workspace.panel.setup
    wait_for(lambda: fitting.prepared is not None)
    shapes = [("¹⁵N", "Pseudo-Voigt", DEFAULT_SOURCE), ("¹H", "SP2", "from NMRPipe apodization")]
    setup.show_state(fitting.configuration, shapes, fitting.prepared, fitting.blocked)
    assert setup.shape_caution.isVisible()
    assert setup.shape_caution.text().startswith("▲ ¹⁵N: no recognized apodization")
    assert w.fit_button.isEnabled()
    assert not fitting.blocked


def _review(result, spectra, rules=None):
    axes = list(spectra.spectral_params)
    titles = [f"axis {i}" for i in range(len(axes))]
    return review_module.review_clusters(
        result, axes, titles, rules or review_module.HighlightRules()
    )


def test_review_measures_are_recorded_and_heuristics_follow_their_definitions(scientific):
    spectra, _peaks, _config, _prepared, result = scientific
    reviews = _review(result, spectra)
    assert [r.cluster_id for r in reviews] == [c.cluster_id for c in result.outcome.clusters]
    for review, cluster in zip(reviews, result.outcome.clusters, strict=True):
        evaluation = cluster.analytical_evaluation
        if evaluation is None:
            continue
        assert review.chi2 == float(evaluation.statistics.reduced_chi_squared)
        assert review.samples == int(evaluation.statistics.n_observations)
        largest = np.max(np.abs(cluster.sample_evaluation.normalized_residuals))
        assert review.largest_residual == float(largest)
    # Shift: |fitted − initial| in Hz relative to the fitted linewidth, largest axis.
    cluster = next(c for c in result.outcome.clusters if c.usable)
    name = cluster.peak_names[0]
    identity = next(i for i in result.inputs.identities if i.engine_name == name)
    start = next(p for p in result.inputs.snapshot.peaks if p.id == identity.working_id)
    values = {p.name: p.value for p in cluster.final_nonlinear_parameters}
    expected = max(
        abs(values[f"{name}.F{i + 2}.cs"] - start.positions[i])
        * a.obs
        / values[f"{name}.F{i + 2}.lw"]
        for i, a in enumerate(spectra.spectral_params)
    )
    nothing = review_module.HighlightRules(1.0, 1e9, 0.0, 1e9)
    everything = review_module.HighlightRules(0.0, expected / 2, 1.0, 1.0)
    assert not any(r.highlights for r in _review(result, spectra, nothing))
    flagged = next(
        r for r in _review(result, spectra, everything) if r.cluster_id == cluster.cluster_id
    )
    shift = next(h for h in flagged.highlights if h.kind == "Position shift")
    assert shift.peak == identity.working_id
    assert shift.rule.startswith(f"shift {expected:.2f} × LW")
    # Thresholds equal to 1 × median flag every usable linewidth except the median.
    assert any(h.kind == "Linewidth outlier" for h in flagged.highlights)


def _with_bound(result):
    """The run with one varied parameter of its first usable cluster at its lower bound."""
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
    edited = replace(result, completed_fit=replace(result.completed_fit, outcome=outcome))
    return edited, cluster, parameter


def test_at_bound_is_a_numerical_warning(scientific):
    spectra, *_, result = scientific
    edited, cluster, parameter = _with_bound(result)
    review = next(r for r in _review(edited, spectra) if r.cluster_id == cluster.cluster_id)
    assert ("At bound", parameter.name) in [(w.kind, w.detail) for w in review.warnings]


def test_highlight_thresholds_filter_clusters_without_touching_the_run(workspace, scientific):
    w = workspace
    result = scientific[-1]
    publish(w, result)
    navigator = w.navigator
    revision, current, flagged = w.edits.revision, w.fitting.current, w.warnings.property("count")
    w.set_rules(review_module.HighlightRules(0.0, 0.0, 1.0, 1.0))
    reviews = w.reviews()
    highlighted = sum(bool(r.highlights) for r in reviews)
    assert highlighted
    options = [navigator.filter.itemText(i) for i in range(navigator.filter.count())]
    assert f"Any highlight ({highlighted})" in options
    assert any(o.strip().startswith("|ρ| ≥ 0") for o in options) or all(r.single for r in reviews)
    navigator.filter.setCurrentIndex(options.index(f"Any highlight ({highlighted})"))
    visible = [
        navigator.tree.topLevelItem(i)
        for i in range(navigator.tree.topLevelItemCount())
        if not navigator.tree.topLevelItem(i).isHidden()
    ]
    assert len(visible) == highlighted
    assert all(item.text(0) in {"◇", "▲"} for item in visible)
    total = len(reviews)
    expected = f"{total} clusters" if highlighted == total else f"{highlighted} of {total} clusters"
    assert navigator.cluster_count.text().startswith(expected)
    warned = sum(bool(r.warnings) for r in reviews)
    navigator.filter.setCurrentIndex(options.index(f"Any warning ({warned})"))
    if warned < total:
        assert navigator.cluster_count.text().startswith(f"{warned} of {total} clusters")
    # Display thresholds change no scientific state, warnings or run currentness.
    assert w.fitting.result is result
    assert (w.edits.revision, w.fitting.current) == (revision, current)
    assert w.warnings.property("count") == flagged
    # The inspector lists highlights apart from numerical warnings.
    review = next(r for r in reviews if r.highlights)
    w.select_peak(review.ids[0])
    text = _review_text(w)
    assert "Highlights" in text
    assert "Numerical warnings" in text
    assert review.highlights[0].rule in text


def test_starting_values_show_the_next_fit_inputs(workspace):
    w, fitting = workspace, workspace.fitting
    wait_for(lambda: fitting.prepared is not None)
    peak = w.peaks[0]
    w.select_peak(peak.id)
    w._update_inspector()
    prepared = fitting.prepared
    engine = next(i.engine_name for i in prepared.identities if i.working_id == peak.id)
    expected = [
        p for p in prepared.parameters.values() if p.param_id and p.param_id.peak_name == engine
    ]
    grid = w.panel.start_grid
    shown = [grid.itemAtPosition(2 * row, 1).widget().text() for row in range(len(expected))]
    assert len(shown) == len(expected)
    # Rows follow display order (horizontal axis first); compare independently of order.
    values = sorted(float(text.replace("−", "-")) for text in shown)
    assert values == pytest.approx(sorted(p.value for p in expected), rel=1e-3, abs=1e-3)
    assert "next fit" in w.panel.starts_note.text()
    # Without a current preparation the section says why instead of showing stale values.
    invalid = fitting.configuration.model_copy(deep=True)
    invalid.exclude_planes = [1]
    fitting.configure(invalid)
    wait_for(lambda: bool(fitting.blocked))
    w._update_inspector()
    assert grid.count() == 0
    assert fitting.blocked in w.panel.starts_note.text()


def test_warning_navigation_never_selects_a_hidden_cluster(workspace, scientific):
    w = workspace
    publish(w, _with_bound(scientific[-1])[0])
    navigator = w.navigator
    # Make every cluster highlighted, then warn about none that the filter shows.
    w.set_rules(review_module.HighlightRules(0.0, 0.0, 1.0, 1.0))
    options = [navigator.filter.itemText(i) for i in range(navigator.filter.count())]
    warned_choice = next(i for i, t in enumerate(options) if t.startswith("Any warning"))
    highlight_choice = next(i for i, t in enumerate(options) if t.startswith("Any highlight"))
    navigator.filter.setCurrentIndex(highlight_choice)
    rows = [navigator.tree.topLevelItem(i) for i in range(navigator.tree.topLevelItemCount())]
    warned = [r for r in rows if r.data(0, QtCore.Qt.ItemDataRole.UserRole).warnings]
    assert warned
    assert len(warned) < len(rows)
    # Hide every warned row: N must not select any of them.
    for row in warned:
        row.setHidden(True)
    before = w.selected_id
    w.fitting.next_warning()
    assert w.selected_id == before
    assert "current filter" in w.status.text()
    # With warned rows shown, N visits exactly them, in list order.
    navigator.filter.setCurrentIndex(warned_choice)
    shown = [r for r in rows if not r.isHidden()]
    visited = []
    for _ in shown:
        w.fitting.next_warning()
        visited.append(navigator._clusters[w.selected_id])
    assert visited == shown
    assert all(not item.isHidden() for item in visited)


def test_normalized_profile_draws_ratios_and_keeps_recorded_amplitudes(scientific):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    result = scientific[-1]
    peak = result.inputs.identities[0].working_id
    shown = peak_profile(result, peak, None)
    assert isinstance(shown, PeakProfile)
    recorded = shown.series[0].amplitudes.copy()
    panel = ProfilePlot()
    panel.resize(320, 260)
    panel.show()
    panel.show_profile(shown, 0)
    app.processEvents()
    # Raw by default; nothing is normalized until a reference is chosen.
    assert panel.mode.text() == "Raw amplitudes"
    assert panel._normalized is None
    reference = int(shown.source_planes[0])
    panel.set_reference((reference,))
    selected = shown.series[0]
    expected = normalize(
        selected.amplitudes, np.full(selected.amplitudes.size, selected.error), np.array([0])
    )
    assert isinstance(expected, Normalized)
    _x, y = _selected_points(panel).getData()
    np.testing.assert_allclose(y, expected.values)
    assert y[0] == 1.0
    assert panel.mode.text() == f"÷ plane {reference + 1}"
    assert f"A / A_ref (plane {reference + 1})" in panel.plot.getAxis("left").labelText
    assert "reference" in panel.legend.text()
    assert "full gradient" in panel.note.text()
    # The reference set persists across selections; recorded amplitudes never change.
    panel.show_profile(shown, 0)
    assert panel.normalized
    np.testing.assert_array_equal(shown.series[0].amplitudes, recorded)
    panel.set_normalized(False)
    _x, y = _selected_points(panel).getData()
    np.testing.assert_array_equal(y, recorded)


def test_unavailable_normalization_keeps_the_raw_profile_with_the_reason(scientific):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    result = scientific[-1]
    shown = peak_profile(result, result.inputs.identities[0].working_id, None)
    assert isinstance(shown, PeakProfile)
    selected = shown.series[0]
    zeroed = replace(
        shown,
        series=(replace(selected, amplitudes=np.zeros_like(selected.amplitudes)),),
    )
    panel = ProfilePlot()
    panel.show()
    panel.show_profile(zeroed, 0)
    panel.set_reference((int(shown.source_planes[0]),))
    app.processEvents()
    assert panel._normalized is None
    assert panel.fallback.isVisible()
    assert panel.fallback.text().startswith("▲ Showing raw amplitudes.")
    assert "is undefined: reference amplitude is zero" in panel.fallback.text()
    assert panel.mode.text().endswith("· not applied")
    assert "A / A_ref" not in panel.plot.getAxis("left").labelText
    _x, y = _selected_points(panel).getData()
    np.testing.assert_array_equal(y, zeroed.series[0].amplitudes)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("1", (1,)), ("1-3, 5", (1, 2, 3, 5)), (" 3 ,1 ", (1, 3))],
)
def test_reference_planes_are_plane_numbers(text, expected):
    assert parse_planes(text, 5) == expected


@pytest.mark.parametrize("text", ["", "0", "6", "a", "3-1"])
def test_invalid_reference_planes_are_rejected(text):
    with pytest.raises(ValueError, match=r"plane|Planes"):
        parse_planes(text, 5)


def test_normalization_is_unavailable_when_uncertainties_are_unreliable(scientific):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    result = scientific[-1]
    shown = peak_profile(result, result.inputs.identities[0].working_id, None)
    assert isinstance(shown, PeakProfile)
    # Inflated amplitudes and σ by one arbitrary scale, as in a degenerate pair.
    selected = shown.series[0]
    inflated = replace(
        shown,
        series=(
            replace(selected, amplitudes=selected.amplitudes * 1e3, error=selected.error * 1e3),
        ),
        warning="Unreliable local uncertainties",
    )
    panel = ProfilePlot()
    panel.show()
    panel.show_profile(inflated, 0)
    panel.set_reference((int(shown.source_planes[0]),))
    app.processEvents()
    assert panel._normalized is None
    assert "withheld as unreliable" in panel.fallback.text()
    assert "uncertainties unreliable" in panel.fallback.text()
    assert "uncertainties unreliable" not in panel.note.text()
    _x, y = _selected_points(panel).getData()
    np.testing.assert_array_equal(y, inflated.series[0].amplitudes)
    # The reference stays: a well-determined profile normalizes again at once.
    panel.show_profile(shown, 0)
    assert panel._normalized is not None
    assert not panel.fallback.isVisible()
    assert panel.mode.text() == f"÷ plane {int(shown.source_planes[0]) + 1}"


def _page_labels(page) -> list[str]:
    return [label.text() for label in page.findChildren(QtWidgets.QLabel) if label.isVisible()]


def test_cluster_row_opens_the_cluster_page_with_recorded_values(workspace, scientific):
    w, c = workspace, workspace.canvas
    result = scientific[-1]
    publish(w, result)
    cluster = next(cl for cl in result.outcome.clusters if cl.usable and len(cl.peak_names) > 1)
    identities = {i.engine_name: i for i in result.inputs.identities}
    members = [identities[n] for n in cluster.peak_names]
    row = w.navigator._clusters[members[0].working_id]
    w.navigator.tree.setCurrentItem(row)
    panel = w.panel
    # The cluster is the unit of review; a member stays selected for editing.
    assert panel.cluster_view
    assert panel.cluster_page.isVisible()
    assert not panel.peak_page.isVisible()
    assert w.selected_id in {m.working_id for m in members}
    assert panel.kind.text() == f"CLUSTER · {len(members)} PEAKS"
    review = next(r for r in w.reviews() if w.selected_id in r.ids)
    links = [
        b
        for b in panel.cluster_page.findChildren(QtWidgets.QPushButton)
        if b.objectName() == "link"
    ]
    assert [b.text() for b in links] == list(review.names)
    # Matrix cells are the recorded correlations in member order.
    matrix = review_module.correlation_matrix(cluster)
    assert matrix is not None
    engines = {i.working_id: i.engine_name for i in result.inputs.identities}
    order = [cluster.peak_names.index(engines[m]) for m in review.ids]
    grid = panel.cluster_page.matrix
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            text = grid.itemAtPosition(i + 1, j + 1).widget().text()
            expected = "1" if i == j else formatting.number(round(float(matrix[a, b]), 2) + 0.0, 2)
            assert text == expected
    # Parameter cells are the recorded values at display precision, σ in the tooltip.
    parameter = next(p for p in cluster.final_nonlinear_parameters if p.name.endswith(".lw"))
    cells = [
        label
        for label in panel.cluster_page.findChildren(QtWidgets.QLabel)
        if label.toolTip().startswith(parameter.name + ":")
    ]
    assert len(cells) == 1
    assert cells[0].text() == formatting.number(parameter.value, 1)
    # The shared profile sits on the cluster page; the view never moved.
    assert panel.profile_block.parent() is not None
    assert panel.cluster_page.isAncestorOf(panel.profile_block)
    view = c.main.getViewBox().viewRange()
    # A member link opens that peak's page; the peak page links back to its cluster.
    target = review.ids[-1]
    links[-1].click()  # assignments may repeat; links follow member order
    assert w.selected_id == target
    assert not panel.cluster_view
    assert panel.peak_page.isVisible()
    assert panel.peak_page.isAncestorOf(panel.profile_block)
    assert panel.cluster_link.isVisible()
    assert panel.cluster_link.text() == f"Review cluster of {len(members)} ›"
    panel.cluster_link.click()
    assert panel.cluster_view
    assert w.selected_id == target
    assert c.main.getViewBox().viewRange() == view
    # Selecting on the canvas or in the Peaks list returns to the peak page.
    w.select_peak(members[0].working_id)
    assert not panel.cluster_view


def test_cluster_page_explains_missing_correlations(workspace, scientific):
    w = workspace
    result = scientific[-1]
    single = next((c for c in result.outcome.clusters if len(c.peak_names) == 1), None)
    multi = next(c for c in result.outcome.clusters if c.usable and len(c.peak_names) > 1)
    evaluation = multi.analytical_evaluation
    hidden = replace(
        evaluation, amplitude_covariance=np.full_like(evaluation.amplitude_covariance, np.nan)
    )
    edited = replace(multi, analytical_evaluation=hidden)
    outcome = replace(
        result.outcome,
        clusters=tuple(edited if cl is multi else cl for cl in result.outcome.clusters),
    )
    publish(w, replace(result, completed_fit=replace(result.completed_fit, outcome=outcome)))
    identities = {i.engine_name: i.working_id for i in result.inputs.identities}
    w.select_peak(identities[multi.peak_names[0]], cluster=True)
    assert "Unavailable: the amplitude covariance was not recorded." in _page_labels(
        w.panel.cluster_page
    )
    if single is not None:
        w.select_peak(identities[single.peak_names[0]], cluster=True)
        assert "Not applicable: a single-peak cluster." in _page_labels(w.panel.cluster_page)
