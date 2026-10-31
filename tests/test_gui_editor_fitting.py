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
from peakfit.engine.algorithms.evaluation import FitOutcomeClassification
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.editor import fitting as fitting_module
from peakfit.gui.editor.profile import PeakProfile, ProfilePanel, framed_range, peak_profile
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
    widget.fitting.reviewed = True
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
    w.fitting.toggle_review()
    assert "Not converged" in w.fitting.review.toPlainText()
    w.fitting.next_warning(None)
    w.fitting.next_warning(None)
    assert "synthetic unusable" in w.fitting.review.toPlainText()


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
    assert len(c.paths) == 2


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
        assert len(c.paths) == 2, "Current-plane experimental contours must remain visible"
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
        assert len(c.paths) == 2
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
    assert len(c.paths) == 2
    c._contours_ready("model", obsolete, (current_model, []))
    assert c._model is None
    wait_for(lambda: not c.contour_pending)
    assert all(key[0] == "replacement" for key in c._model_cache)
    residual = (c.spectra.data[2] - c._model.values) / result.outcome.noise
    np.testing.assert_array_equal(c.curve_x.getData()[1], residual[c.iy])


@pytest.fixture
def settings(tmp_path):
    """Keep profile preferences out of the developer's own application settings."""
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
    w, panel = workspace, workspace.profile
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
    assert "Outdated" in panel.title.text()
    assert panel._profile is not None
    w.place_peak(peak.positions)
    assert panel.title.text().startswith(f"Not part of run {result.run_id[:8]}")


def test_profile_follows_only_accepted_selection(workspace, scientific):
    w = workspace
    result = scientific[-1]
    publish(w, result)
    first, second = w.peaks[0], w.peaks[1]
    w.select_peak(first.id)
    title, revision = w.profile.title.text(), w.edits.revision
    w.inspector.x_field.setText("not a number")
    w.select_peak(second.id)
    assert w.selected_id == first.id
    assert w.profile.title.text() == title
    assert w.edits.revision == revision
    # A valid draft is committed exactly once by the existing selection rule.
    w.inspector.x_field.setText(repr(first.positions[1] + 0.001))
    w.select_peak(second.id)
    assert w.selected_id == second.id
    assert w.edits.revision == revision + 1
    assert w.profile.title.text().startswith(second.name)
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
    panel = ProfilePanel()
    requested = []
    panel.plane_requested.connect(requested.append)
    panel.show_selection(run, peak, current=True, busy=False, plane_values=repeated, plane=0)
    assert panel._profile is not None
    np.testing.assert_array_equal(panel._profile.x, repeated)
    points = _selected_points(panel)
    points.sigClicked.emit(points, points.points()[:2], None)
    actions = panel._chooser.actions()
    assert [a.text() for a in actions] == ["Plane 1 · value 1", "Plane 2 · value 1"]
    actions[1].trigger()
    assert requested == [2]


def test_compact_header_and_profile_at_minimum_window(scientific, settings, monkeypatch):
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
    w.fitting.reviewed = True
    app.processEvents()
    assert not w.profile.toggle.isChecked()
    assert not w.profile.plot.isVisible()
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

    buttons = [w.cancel_fit, w.review_fit, w.peaks_button, w.add_button, w.fit_button]
    buttons.append(w.settings_button)
    assert all(b.isVisible() and window.rect().contains(rect(b)) for b in buttons)
    for label in (w.title, w.summary):
        assert not any(rect(label).intersects(rect(b)) for b in buttons)
    assert "Preparing complete dataset" in w.summary.displayed_text()
    w.fitting._preparing = False
    w.fitting.update()
    app.processEvents()
    shown = w.summary.displayed_text()
    assert "Outdated" in shown
    assert "uncertainty warnings" in shown
    assert w.profile.badge.isVisible()
    assert window.rect().contains(rect(w.profile.badge))
    assert "Outdated" in w.profile.title.toolTip()
    # An oversized remembered height never squeezes the spectrum below its minimum.
    settings.setValue("profile/height", 5000)
    w.profile.toggle.setChecked(True)
    app.processEvents()
    assert settings.value("profile/expanded", type=bool)
    assert w.canvas.main.height() >= w.canvas.main.minimumHeight()
    assert w.canvas.top.height() == w.canvas.top.minimumHeight()
    assert w.profile.plot.isVisible()
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
    panel = ProfilePanel()
    panel.set_expanded(True)
    panel.resize(800, 240)
    panel.show()
    panel.show_selection(run, peak, current=True, busy=False, plane_values=None, plane=0)
    app.processEvents()
    assert not panel.plot.grab().isNull()
    assert panel._profile is not None
    np.testing.assert_array_equal(panel._profile.series[0].amplitudes, evaluation.amplitudes[0])
    assert np.isnan(panel._profile.series[0].error)
    bars = [i for i in panel.plot.getPlotItem().items if isinstance(i, pg.ErrorBarItem)]
    assert len(bars) == len(weak.peak_names) - 1
    assert np.all(np.isfinite(panel.plot.getViewBox().viewRange()))
    assert "σ unavailable" in panel.title.text()
    assert "σ unavailable" in panel.basis.text()
    panel.close()
