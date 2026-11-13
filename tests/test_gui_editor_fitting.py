"""Protect the fit → optimized working state → native comparison → refit loop."""

import csv
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import Event

import numpy as np
import pyqtgraph as pg
import pytest
from builders import two_cluster_dataset
from gui_helpers import wait_until
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

import peakfit.gui.editor.review as review_module
import peakfit.gui.editor.spectral_view as canvas_module
import peakfit.gui.editor.workspace as workspace_module
from peakfit.engine.algorithms.covariance_factors import marginal_standard_errors
from peakfit.engine.algorithms.evaluation import FitOutcomeClassification
from peakfit.engine.diagnostics.identifiability import AmplitudeStatus, Cause, MemberAmplitude
from peakfit.engine.domain.config import FitConfig, PeakFitConfig
from peakfit.engine.domain.constraints import ParameterConfig
from peakfit.engine.scientific.profiles import Normalized, normalize
from peakfit.gui import formatting
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.editor import fitting as fitting_module
from peakfit.gui.editor import slices
from peakfit.gui.editor.profile import (
    PeakProfile,
    ProfilePlot,
    framed_range,
    parse_planes,
    peak_profile,
)
from peakfit.gui.editor.setup import DEFAULT_SOURCE
from peakfit.gui.editor.spectral_geometry import MAIN_MIN
from peakfit.gui.editor.unexplained import nearest_peak, unexplained_regions
from peakfit.gui.editor.window import EditorWindow
from peakfit.gui.editor.workspace import EditorWorkspace
from peakfit.gui.fit_display import dataset_model_plane
from peakfit.gui.fit_export import export_fit_result
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.project import StoredRun
from peakfit.gui.spectrum import Dataset, display_position, read_dataset
from peakfit.io.scientific import read_scientific

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def scientific():
    spectra, peaks, config, prepared = two_cluster_dataset()
    result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    return spectra, peaks, config, prepared, result


@pytest.fixture
def workspace(qapp, scientific):
    spectra, peaks, config, _, _ = scientific
    dataset = Dataset(Path("synthetic.ft2"), spectra, peaks, (), True, 0.01)
    widget = EditorWorkspace(dataset)
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


def publish(workspace, result, *, timeout=10):
    fitting = workspace.fitting
    fitting._submitted_revision = workspace.edits.revision
    fitting.process.state = "completed"
    fitting._finished(0, result)
    wait_until(lambda: not fitting.busy, timeout=timeout)
    assert fitting.result is result, fitting.failure


def test_atomic_completion_comparison_draft_edit_and_export(workspace, scientific, tmp_path):
    w, c = workspace, workspace.canvas
    result = scientific[-1]
    w.select_peak(w.peaks[0].id)
    following = c.pin_peak_id
    ranges = c.box.viewRange()
    seeds = w.peaks
    publish(w, result)
    assert w.fitting.current
    assert w.peaks == w.fitting.completion.working_peaks
    assert result.inputs.snapshot.peaks == seeds
    assert not w.edits.can_undo
    assert c.pin_peak_id == following
    assert c.selected_id == following
    np.testing.assert_allclose(c.box.viewRange(), ranges)
    assert c.comparison_mode == 1
    wait_until(lambda: not c.contour_pending)
    authoritative = dataset_model_plane(result, c.spectra.spectral_params, c.plane, cancel=Event())
    np.testing.assert_array_equal(c._model.values, authoritative.values)
    w.set_mode(2)
    wait_until(lambda: not c.contour_pending)
    residual = (c.spectra.data[c.plane] - authoritative.values) / result.outcome.noise
    np.testing.assert_array_equal(c.curve_x.getData()[1], residual[c.iy, :])
    np.testing.assert_array_equal(c.curve_y.getData()[0], residual[:, c.ix])
    jobs = c.contour_jobs
    for _ in range(500):
        c.sample(float(c.ppm_x[12]), float(c.ppm_y[13]))
    assert c.contour_jobs == jobs
    # Residual slices keep the lane scale wherever the crosshair goes.
    lane = [-canvas_module.LANE_SIGMA, canvas_module.LANE_SIGMA]
    np.testing.assert_allclose(c.row_box.viewRange()[1], lane)
    np.testing.assert_allclose(c.column_box.viewRange()[0], lane)
    # A draft never invalidates accepted science; one commit does, including Undo.
    w.inspector.x_field.setText("8.123456789")
    assert w.fitting.current
    assert c.comparison_mode == 2
    w.inspector.show_peak(w.peaks[0], c.display(w.peaks[0]))
    peak = w.peaks[0]
    w.move_peak(peak.id, (peak.positions[0], peak.positions[1] + 0.00001))
    assert not w.fitting.current
    # The Outdated run's residual stays, explicitly historical; nothing is refitted.
    assert c.comparison_mode == 2
    assert c.comparison_result is result
    assert c.comparison_label == f"Run {result.run_id[:8]} · historical"
    w.history(False)
    assert w.peaks == w.fitting.completion.working_peaks
    assert not w.fitting.current  # Undo never recertifies a run
    assert w.fitting.comparison.historical
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
    wait_until(lambda: not w.fitting.busy, timeout=20)
    assert w.fitting.current, w.fitting.failure
    previous = w.fitting.result
    seeds = w.peaks
    # Cancellation never publishes partial scientific values.
    w.fitting.fit()
    assert w.fitting.busy
    w.delete_peaks()
    assert w.peaks == seeds
    w.fitting.cancel()
    wait_until(lambda: not w.fitting.busy)
    assert w.fitting.result is previous
    assert w.peaks == seeds
    w.fitting.process.state = "failed"
    w.fitting._finished(0, "simulated child crash")
    assert w.fitting.result is previous
    assert w.fitting.current
    w.fitting.fit()
    wait_until(lambda: not w.fitting.busy, timeout=20)
    assert w.fitting.current, w.fitting.failure
    assert w.fitting.result.inputs.snapshot.peaks == seeds
    assert w.fitting.result.inputs is not previous.inputs
    for _ in range(20):
        seeds = w.peaks
        w.fitting.fit()
        wait_until(lambda: not w.fitting.busy, timeout=20)
        assert w.fitting.current, w.fitting.failure
        assert w.fitting.result.inputs.snapshot.peaks == seeds


def test_fitting_state_is_rendered_on_transitions_without_polling(workspace):
    w, fitting = workspace, workspace.fitting
    wait_until(lambda: fitting.prepared is not None)
    busy_at_transition = []
    fitting.state_changed.connect(lambda: busy_at_transition.append(fitting.busy))
    QtTest.QTest.qWait(500)
    # Idle: nothing changes, so nothing is emitted and the elapsed-time clock is stopped.
    assert busy_at_transition == []
    assert not w._fit_clock.isActive()
    fitting.fit()
    assert w._fit_clock.isActive()
    assert "Navigation available" in w.summary.text()
    wait_until(lambda: not fitting.busy, timeout=20)
    assert fitting.current, fitting.failure
    assert not w._fit_clock.isActive()
    assert busy_at_transition[0]
    assert not busy_at_transition[-1]
    assert w.summary.text().startswith(f"Run {fitting.result.run_id[:8]} · Current")
    # Counts only when something is not usable or not converged; always in the tooltip.
    clusters = fitting.result.outcome.clusters
    if all(c.usable and c.optimizer_provenance.converged for c in clusters):
        assert w.summary.text().endswith(f"all {len(clusters)} clusters usable")
    assert f"{len(clusters)}/{len(clusters)} usable" in w.summary.toolTip()
    # The new run is the next continuation source; its preparation is the last transition.
    wait_until(lambda: fitting.prepared is not None)
    count = len(busy_at_transition)
    QtTest.QTest.qWait(500)
    assert len(busy_at_transition) == count


def test_one_busy_state_locks_and_releases_every_edit(workspace):
    w = workspace
    w.select_peak(w.peaks[0].id)
    wait_until(lambda: w.fitting.prepared is not None)
    controls = (
        w.add_action,
        w.delete_action,
        w.add_button,
        w.fit_button,
        w.inspector,
        w.panel.setup,
    )
    assert all(control.isEnabled() for control in controls)
    w.fitting.fit()
    assert w.fitting.busy
    assert not any(control.isEnabled() for control in controls)
    assert not w.canvas.editing_enabled
    assert w.cancel_fit.isVisible()
    assert not w.settings_button.isVisible()
    wait_until(lambda: not w.fitting.busy, timeout=20)
    assert w.fitting.current, w.fitting.failure
    wait_until(lambda: w.fitting.prepared is not None)
    # Every control the fit locked is released, including the Edit menu's Add peak.
    assert all(control.isEnabled() for control in controls)
    assert w.canvas.editing_enabled
    assert not w.cancel_fit.isVisible()
    assert w.settings_button.isVisible()


def test_pinning_stays_available_in_focus_and_after_it(workspace, scientific):
    w = workspace
    result = scientific[-1]
    publish(w, result)
    assert w.pin_button.isEnabled()
    _open_focus(w, result)
    assert w.pin_button.isEnabled()
    w.close_focus()
    assert w.pin_button.isEnabled()


def test_showing_the_dataset_page_counts_as_seeing_the_setup(workspace):
    w, fitting = workspace, workspace.fitting
    w.select_peak(w.peaks[0].id)
    fitting.setup_seen = False
    w.select_peak(None)
    # The dataset page, with the fit setup, is on show: the first Fit runs directly.
    assert fitting.setup_seen


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
    wait_until(lambda: not c.contour_pending)
    assert c.plane == 2
    expected = dataset_model_plane(scientific[-1], c.spectra.spectral_params, 2, cancel=Event())
    np.testing.assert_array_equal(c._model.values, expected.values)
    model = c._model
    w.set_mode(0)
    w.set_mode(2)
    wait_until(lambda: not c.contour_pending)
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
    w.fitting._finished(0, invalid)
    wait_until(lambda: not w.fitting.busy)
    assert w.fitting.result is unusable
    assert w.peaks == seeds
    assert "stable-ID mapping" in w.fitting.failure


@pytest.mark.parametrize("mode", [1, 2])
def test_pending_comparison_keeps_current_experiment(workspace, scientific, monkeypatch, mode):
    w, c = workspace, workspace.canvas
    # Prime experimental geometry for this plane before requesting an uncached model.
    w.set_plane(3)
    wait_until(lambda: not c.contour_pending)
    entered, release = Event(), Event()
    project = canvas_module.dataset_model_plane

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return project(*args, **kwargs)

    monkeypatch.setattr(canvas_module, "dataset_model_plane", delayed)
    try:
        publish(w, scientific[-1])
        wait_until(entered.is_set)
        if mode == 2:
            w.set_mode(2)
        assert c.contour_pending
        assert c._model is None
        assert [layer.role for layer in c.paths] == ["measured"], (
            "Current-plane experimental contours must remain visible"
        )
        # The label says what is on show and what is still coming.
        assert c.pending_label.toPlainText() == (
            f"Experimental contours · {'residual' if mode == 2 else 'model'} pending"
        )
        if mode == 2:
            assert c.curve_x.getData()[1] is None
    finally:
        release.set()
    wait_until(lambda: not c.contour_pending)


@pytest.mark.usefixtures("qapp")
def test_toolbar_title_and_close_share_one_unsaved_state(scientific, monkeypatch, tmp_path):
    spectra, peaks, config, _, result = scientific
    window = EditorWindow()
    window.install_dataset(Dataset(Path("synthetic.ft2"), spectra, peaks, (), True, 0.01))
    w = window.workspace
    assert w is not None
    w.fitting.setup_seen = True

    def shown():
        # The toolbar, the window title and the close prompt answer the same question.
        assert window.isWindowModified() == bool(w.modified.text()) == bool(window.unsaved())
        return w.modified.toolTip()

    assert shown() == ""
    w.fitting.configure(config.model_copy(deep=True))
    assert shown() == "Unsaved: fit settings"
    publish(w, result)
    assert shown() == "Unsaved: peaks, fit settings, active run, run history"
    # Exporting results or peaks never marks the analysis saved.
    monkeypatch.setattr(fitting_module, "choose_and_export", lambda *_a, **_k: tmp_path)
    assert w.fitting.export()
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", lambda *_: (str(tmp_path / "peaks.csv"), "")
    )
    assert window.export_dialog()
    assert shown() == "Unsaved: peaks, fit settings, active run, run history"
    w.select_peak(w.peaks[0].id)  # selection is navigation, never an analysis change
    assert shown() == "Unsaved: peaks, fit settings, active run, run history"
    w.inspector.name.setText("draft")
    assert shown().endswith(", uncommitted draft")
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda _dialog: 0)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "clickedButton",
        lambda dialog: next(b for b in dialog.buttons() if b.text() == "Discard changes"),
    )
    asked = []
    original = QtWidgets.QMessageBox.addButton
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "addButton",
        lambda dialog, *args: asked.append(args) or original(dialog, *args),
    )
    assert window.protect_edits()
    assert asked  # the save offer was shown
    window.close()


def test_pending_latest_plane_and_run_reject_stale_delivery(workspace, scientific, monkeypatch):
    w, c = workspace, workspace.canvas
    result = scientific[-1]
    publish(w, result)
    wait_until(lambda: not c.contour_pending)
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
        wait_until(entered.is_set)
        assert [layer.role for layer in c.paths] == ["measured"]
        assert c._displaying_experiment
        assert c._model is None
        assert c.curve_x.getData()[1] is None
        obsolete = c._revision
        w.set_plane(3)
        # Nothing is carried across acquisition indices: whatever shows is plane 3's
        # (its measured contours may already be prepared), and no model yet.
        assert all(layer.key[2] == 2 for layer in c.paths)
        assert [layer.role for layer in c.paths] in ([], ["measured"])
        assert c._model is None
    finally:
        release.set()
    wait_until(lambda: not c.contour_pending)
    expected = project(result, c.spectra.spectral_params, 2, cancel=Event())
    np.testing.assert_array_equal(c._model.values, expected.values)
    current_model = c._model
    c._contours_ready("model", obsolete, (old_model, []))
    assert c._model is current_model
    # Replacing a run on the same plane can reuse only experimental geometry. The
    # workspace installs the replacement and binds the canvas, as a completed fit
    # does; binding the canvas alone would be undone by the next workspace refresh.
    obsolete = c._revision
    replacement = replace(result, run_id="replacement")
    w.fitting.completion = replace(w.fitting.completion, result=replacement)
    w.set_mode(2)
    assert c.comparison_result is replacement
    assert c._model is None
    assert c._displaying_experiment
    assert [layer.role for layer in c.paths] == ["measured"]
    c._contours_ready("model", obsolete, (current_model, []))
    assert c._model is None
    wait_until(lambda: not c.contour_pending)
    assert all(key[0] == "replacement" for key in c.model_planes.retained())
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
    )
    return replace(result, completed_fit=replace(result.completed_fit, outcome=outcome))


def _review_cell(grid, label: str) -> QtWidgets.QLabel:
    """The value beside a label in a review grid."""
    for row in range(grid.rowCount()):
        item = grid.itemAtPosition(row, 0)
        if item is not None and item.widget().text() == label:
            return grid.itemAtPosition(row, 1).widget()
    raise KeyError(label)


def _review_row(grid, label: str) -> str:
    return _review_cell(grid, label).text()


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
    assert selected.error == evaluation.amplitude_standard_errors[1]
    # σ-only: the bar is the unscaled analytical σ; χ²ᵣ is a measure, not a bar scale.
    chi2 = evaluation.statistics.reduced_chi_squared
    assert profile.reduced_chi_squared == chi2
    assert "±1σ conditional on lineshapes" in panel.note.text()
    assert "scaled" not in panel.note.text()
    assert "never rescaled by χ²ᵣ" in panel.note.toolTip()
    # The synthetic dataset has no NMRPipe header: its errors say so.
    assert result.outcome.noise_model.reason == "no NMRPipe processing header"
    assert "noise correlation not modelled" in panel.note.text()
    assert result.outcome.noise_model.description in panel.note.toolTip()
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


@pytest.mark.usefixtures("qapp")
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
def test_compact_header_and_inspector_at_minimum_window(qapp, scientific, monkeypatch):  # noqa: PLR0915
    spectra, peaks, config, _, result = scientific
    window = EditorWindow()
    monkeypatch.setattr(window, "protect_edits", lambda **_: True)
    window.resize(1040, 720)
    window.show()
    name = "a-deliberately-long-dataset-name-for-the-compact-header-" * 2 + ".ft2"
    window.install_dataset(Dataset(Path(name), spectra, peaks, (), True, 0.01))
    w = window.workspace
    assert w is not None
    # The toolbar names the application; the window title names the dataset.
    assert w.title.text() == "PeakFit"
    assert name in window.windowTitle()
    w.fitting.configuration = config.model_copy(deep=True)
    w.fitting.setup_seen = True
    qapp.processEvents()
    publish(w, _warned(result))
    warned = result.outcome.clusters[0].peak_names[0]
    w.select_peak(next(i.working_id for i in result.inputs.identities if i.engine_name == warned))
    peak = next(p for p in w.peaks if p.id == w.selected_id)
    w.move_peak(peak.id, (peak.positions[0], peak.positions[1] + 1e-5))
    w.fitting._preparing = True
    w.fitting.attempt = "Preparing complete dataset"
    w.render_fitting()
    qapp.processEvents()

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
    w.render_fitting()
    qapp.processEvents()
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
    # The window is never enlarged: the navigator gave way, the spectrum kept its minimum.
    assert window.width() == 1040
    assert w.navigator.isHidden()
    assert not w.peaks_button.isChecked()
    assert "The peak list is hidden to fit the window; widen it to show it again." in (
        w.error.text()
    )
    plot = w.canvas.arrangement.main
    assert plot[2] >= MAIN_MIN[0]
    assert plot[3] >= MAIN_MIN[1]
    window.close()
    window.deleteLater()


def _resize(qapp, window, width: int, height: int) -> None:
    """Resize once Qt has lowered the window's minimum after a panel was hidden."""
    for _ in range(3):
        qapp.processEvents()
        window.resize(width, height)
    qapp.processEvents()
    assert window.size() == QtCore.QSize(width, height)


@pytest.mark.usefixtures("settings")
def test_side_panels_open_only_where_they_fit_and_never_enlarge_the_window(
    qapp, scientific, monkeypatch
):
    spectra, peaks, config, _, result = scientific
    window = EditorWindow()
    monkeypatch.setattr(window, "protect_edits", lambda **_: True)
    window.resize(1160, 720)  # the reference: both side panels and slices with lanes
    window.show()
    window.install_dataset(Dataset(Path("reference.ft2"), spectra, peaks, (), True, 0.01))
    w = window.workspace
    assert w is not None
    w.fitting.configuration = config.model_copy(deep=True)
    qapp.processEvents()
    publish(w, result)
    w.open_focus(w.peaks[0].id)
    qapp.processEvents()
    assert not w.navigator.isHidden()
    assert not w.panel.isHidden()
    assert w.canvas.arrangement.sizes.lanes
    assert w.canvas.arrangement.main[2] >= MAIN_MIN[0]
    assert w.canvas.arrangement.main[3] >= MAIN_MIN[1]
    assert window.size() == QtCore.QSize(1160, 720)
    w.close_focus()

    # Space for the navigator comes from the spectrum's slack, then the inspector.
    w.peaks_button.click()
    assert w.navigator.isHidden()
    inspector = w.panel.minimumWidth() + 180
    w.splitter.setSizes([0, w.splitter.width() - inspector, inspector])
    qapp.processEvents()
    w.peaks_button.click()
    qapp.processEvents()
    assert not w.navigator.isHidden()
    _navigator, spectral, inspector = w.splitter.sizes()
    assert spectral >= w.canvas.minimum()[0]
    assert w.panel.minimumWidth() <= inspector
    assert window.width() == 1160

    # With no room even at every minimum, opening is declined and says why.
    w.peaks_button.click()  # hide, so the window may narrow
    _resize(qapp, window, 980, 720)
    w.peaks_button.click()
    qapp.processEvents()
    assert w.navigator.isHidden()
    assert not w.peaks_button.isChecked()
    assert w.error.text() == "No room for the peak list: widen the window or hide the inspector."
    assert window.width() == 980
    # Hiding the inspector makes room.
    w.inspector_button.click()
    w.peaks_button.click()
    qapp.processEvents()
    assert not w.navigator.isHidden()
    assert w.error.isHidden()
    window.close()
    window.deleteLater()


@pytest.mark.usefixtures("settings")
def test_slices_and_focus_have_menu_and_keyboard_alternatives(qapp, scientific, monkeypatch):
    spectra, peaks, config, _, result = scientific
    window = EditorWindow()
    monkeypatch.setattr(window, "protect_edits", lambda **_: True)
    window.resize(1280, 800)
    window.show()
    window.install_dataset(Dataset(Path("menus.ft2"), spectra, peaks, (), True, 0.01))
    w = window.workspace
    assert w is not None
    w.fitting.configuration = config.model_copy(deep=True)
    qapp.processEvents()
    publish(w, result)
    c = w.canvas
    actions = {a.text(): a for a in window.findChildren(QtGui.QAction)}
    assert actions["Residual lanes"].shortcut().toString() == "Ctrl+Shift+L"
    actions["Residual lanes"].trigger()
    assert c.sizes.lanes
    row, column = c.sizes.row, c.sizes.column
    actions["Taller row slice"].trigger()
    actions["Narrower column slice"].trigger()
    assert (c.sizes.row, c.sizes.column) == (row + 24, column - 24)
    actions["Default slice sizes"].trigger()
    assert (c.sizes.row, c.sizes.column) == (row, column)
    actions["Magnify slice intensities"].trigger()
    assert c.gain > 1
    actions["Default slice intensities"].trigger()
    assert c.gain == 1
    actions["Reduce residual lanes"].trigger()
    assert c.lane_sigma > canvas_module.LANE_SIGMA
    actions["Default residual lanes"].trigger()
    assert c.lane_sigma == canvas_module.LANE_SIGMA
    # Focus from the menu on the selected peak's cluster, and back.
    w.select_peak(w.peaks[0].id)
    view = c.box.viewRange()
    actions["Focus cluster"].trigger()
    assert w.focus_open
    actions["Return to spectrum"].trigger()
    assert not w.focus_open
    assert c.box.viewRange() == view
    window.close()
    window.deleteLater()


@pytest.mark.usefixtures("settings")
def test_overlay_after_residual_never_enlarges_a_narrowed_window(qapp, scientific, monkeypatch):
    spectra, peaks, config, _, result = scientific
    window = EditorWindow()
    monkeypatch.setattr(window, "protect_edits", lambda **_: True)
    window.resize(1160, 720)
    window.show()
    window.install_dataset(Dataset(Path("lanes.ft2"), spectra, peaks, (), True, 0.01))
    w = window.workspace
    assert w is not None
    w.fitting.configuration = config.model_copy(deep=True)
    qapp.processEvents()
    publish(w, result)
    c = w.canvas
    w.set_lanes(True)
    w.set_mode(2)  # Residual hides the lanes, lowering the view's minimum
    floor = w.canvas.minimum()[0] + sum(p.minimumWidth() + 1 for p in (w.navigator, w.panel))
    _resize(qapp, window, floor + 4, 720)  # narrower than Overlay with lanes needs
    assert not w.navigator.isHidden()
    assert not w.panel.isHidden()
    w.set_mode(1)
    for _ in range(3):  # let Qt apply any new minimum to the window
        qapp.processEvents()
        layout = window.layout()
        assert layout is not None
        layout.activate()
    assert c.comparison_mode == 1
    assert window.size() == QtCore.QSize(floor + 4, 720)
    assert c.arrangement.main[2] >= MAIN_MIN[0]
    assert c.arrangement.main[3] >= MAIN_MIN[1]
    assert c.sizes.lanes  # the preference is kept
    assert not c.row_lane.isVisible()  # the lanes gave way to the window
    assert "Residual lanes are hidden to fit the window" in w.error.text()
    # Modes always switch; once the window is wide enough the lanes return.
    w.set_mode(2)
    w.set_mode(0)
    assert c.comparison_mode == 0
    _resize(qapp, window, 1280, 800)
    w.set_mode(1)
    qapp.processEvents()
    assert c.row_lane.isVisible()
    window.close()
    window.deleteLater()


@pytest.mark.usefixtures("settings")
def test_a_saved_layout_restores_the_inspector_before_the_navigator(qapp, scientific, monkeypatch):
    spectra, peaks, _, _, _ = scientific
    window = EditorWindow()
    monkeypatch.setattr(window, "protect_edits", lambda **_: True)
    monkeypatch.setattr(workspace_module, "dataset_header", lambda _dataset: {})
    window.resize(1280, 760)
    window.show()
    window.install_dataset(Dataset(Path("layout.ft2"), spectra, peaks, (), True, 0.01))
    w = window.workspace
    assert w is not None
    qapp.processEvents()
    saved = w.capture()
    # Same window: the exact panel widths come back.
    sizes = w.splitter.sizes()
    w.restore(saved)
    qapp.processEvents()
    assert w.splitter.sizes() == sizes
    # A narrower window keeps the spectral area and the inspector; the navigator gives way.
    w.peaks_button.click()  # hidden, so the window may narrow
    _resize(qapp, window, 1000, 760)
    w.restore(saved)
    qapp.processEvents()
    assert not w.panel.isHidden()
    assert w.navigator.isHidden()
    assert window.width() == 1000
    window.close()
    window.deleteLater()


def test_profile_keeps_amplitudes_when_uncertainty_is_unavailable(qapp, scientific):
    result = scientific[-1]
    cluster = next(c for c in result.outcome.clusters if c.usable and len(c.peak_names) > 1)
    evaluation = cluster.analytical_evaluation
    assert evaluation is not None
    # A usable but poorly identifiable cluster can report NaN amplitude errors.
    errors = evaluation.amplitude_standard_errors.copy()
    errors[0] = np.nan
    weak = replace(
        cluster,
        analytical_evaluation=replace(evaluation, amplitude_standard_errors=errors),
    )
    clusters = tuple(weak if c is cluster else c for c in result.outcome.clusters)
    outcome = replace(
        result.outcome,
        clusters=clusters,
    )
    run = replace(result, completed_fit=replace(result.completed_fit, outcome=outcome))
    peak = next(i.working_id for i in run.inputs.identities if i.engine_name == weak.peak_names[0])
    panel = ProfilePlot()
    panel.resize(320, 260)
    panel.show()
    shown = peak_profile(run, peak, None)
    assert isinstance(shown, PeakProfile)
    panel.show_profile(shown, 0)
    qapp.processEvents()
    assert not panel.plot.grab().isNull()
    assert panel._profile is not None
    np.testing.assert_array_equal(panel._profile.series[0].amplitudes, evaluation.amplitudes[0])
    assert np.isnan(panel._profile.series[0].error)
    bars = [i for i in panel.plot.getPlotItem().items if isinstance(i, pg.ErrorBarItem)]
    assert len(bars) == len(weak.peak_names) - 1
    assert np.all(np.isfinite(panel.plot.getViewBox().viewRange()))
    assert "σ not available" in panel.legend.text()
    # The value is determined, so it is drawn; only its error bar is missing.
    assert not panel.withheld.isVisible()
    assert "not established (uncertainty unavailable)" in panel.diagnostics.text()
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
    assert row.text(3) == f"{evaluation.statistics.reduced_chi_squared:.2f}"
    # The first member's name in full, a member count, and every name in the tooltip;
    # a name too long for the column elides in the middle.
    names = {p.id: p.name for p in w.peaks}
    members = [names[identities[n]] for n in cluster.peak_names]
    assert row.text(1) == members[0]
    assert row.text(2) == str(len(members))
    assert row.toolTip(1) == " · ".join(members)
    assert tree.textElideMode() == QtCore.Qt.TextElideMode.ElideMiddle
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
    fitted = w.panel.fitted
    cells = [
        [fitted.itemAtPosition(row, column).widget().text() for column in (1, 2)]
        for row in range(1, fitted.rowCount())
        if fitted.itemAtPosition(row, 2) is not None
    ]
    assert [value, sigma] in cells
    assert w.edits.revision == 0


def test_inspector_shows_the_marginal_sigma_of_the_absolute_amplitude(workspace, scientific):
    w = workspace
    result = scientific[-1]
    publish(w, result)
    cluster = next(c for c in result.outcome.clusters if c.usable and len(c.peak_names) > 1)
    evaluation = cluster.analytical_evaluation
    marginal = marginal_standard_errors(evaluation.covariance_factors, evaluation.amplitudes)
    identities = {i.engine_name: i.working_id for i in result.inputs.identities}
    w.select_peak(identities[cluster.peak_names[1]])
    fitted = w.panel.fitted

    def amplitude_row():
        rows = [
            [fitted.itemAtPosition(row, column).widget().text() for column in (0, 1, 2)]
            for row in range(1, fitted.rowCount())
            if fitted.itemAtPosition(row, 2) is not None
        ]
        return next(r for r in rows if r[0].startswith("Amplitude"))

    for plane in (1, 3):
        w.set_plane(plane)
        value, sigma = formatting.with_sigma(
            evaluation.amplitudes[1, plane - 1], marginal[1, plane - 1]
        )
        assert amplitude_row() == [f"Amplitude plane {plane}", value, sigma]
    # Labelled as the uncertainty of the absolute amplitude, not of a profile.
    captions = [
        fitted.itemAt(i).widget().text()
        for i in range(fitted.count())
        if isinstance(fitted.itemAt(i).widget(), QtWidgets.QLabel)
    ]
    assert any("σ of the absolute amplitude" in text for text in captions)


def test_dense_profile_click_opens_the_nearest_plane(qapp, scientific):
    result = scientific[-1]
    peak = result.inputs.identities[0].working_id
    shown = peak_profile(result, peak, None)
    assert isinstance(shown, PeakProfile)
    panel = ProfilePlot()
    panel.resize(320, 260)
    panel.show()
    panel.show_profile(shown, 0)
    qapp.processEvents()
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
    ranges, mode = c.box.viewRange(), c.comparison_mode
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
    assert c.box.viewRange() == ranges
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
    wait_until(lambda: fitting.prepared is not None)
    submitted = []
    monkeypatch.setattr(fitting, "start", submitted.append)
    fitting.fit()
    # The second Fit submits exactly the preparation the setup describes.
    assert submitted == [fitting.prepared]


def test_fit_is_disabled_with_the_preparation_reason(workspace):
    w, fitting = workspace, workspace.fitting
    wait_until(lambda: fitting.prepared is not None)
    invalid = fitting.configuration.model_copy(deep=True)
    invalid.exclude_planes = [1]
    fitting.configure(invalid)
    assert fitting.prepared is None
    wait_until(lambda: bool(fitting.blocked))
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
    wait_until(lambda: fitting.prepared is not None)
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
    wait_until(lambda: fitting.prepared is not None)
    assert '"refine_iterations":3' in fitting.prepared.snapshot.configuration
    setup.lineshape.setCurrentIndex(setup.lineshape.findData("gaussian"))
    assert fitting.configuration.fitting.lineshape == "gaussian"
    assert "Gaussian · explicit" in setup.shapes.text()


def test_fitting_regions_are_exact_hidden_by_default_and_labelled(workspace, scientific):
    w, c, fitting = workspace, workspace.canvas, workspace.fitting
    wait_until(lambda: fitting.prepared is not None)
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
    view, mode, jobs = c.box.viewRange(), c.comparison_mode, c.contour_jobs
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
    assert view == c.box.viewRange()
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
    wait_until(lambda: fitting.prepared is not None)
    w._update_inspector()
    assert c.region_source == "preview for current settings"
    w.panel.setup.regions.setChecked(False)
    assert not c.region_image.isVisible()
    assert "fitting regions" not in c.legend.text()


def test_the_loaded_screen_says_each_thing_once(workspace):
    w, fitting, setup = workspace, workspace.fitting, workspace.panel.setup
    wait_until(lambda: fitting.prepared is not None)
    w.select_peak(None)
    w._update_inspector()
    # The toolbar states the run; the counts live in the tab, plane field and page.
    assert w.summary.text() == "Not fitted"
    assert not w.panel.review_block.isVisibleTo(w)  # no run section before a fit
    assert w.panel.status.text().startswith(f"{len(w.peaks)} peaks · ")
    # The total is in the Peaks tab; the footer counts only a filter.
    nav = w.navigator
    assert nav.count.isHidden()
    nav.search.setText("remote")  # matches one peak's ID in this fixture
    assert not nav.count.isHidden()
    assert nav.count.text().endswith(f"of {len(w.peaks)} peaks")
    nav.search.setText("")
    # Setup values stay; their explanations are tooltips, numbers never use e+.
    assert "\n" not in setup.shapes.text()
    assert setup.shapes.toolTip()
    assert "e+" not in setup.noise.text() + setup.level.text()
    # Only an explicit noise level is worth a caption; the estimate is the default.
    explicit = bool(fitting.configuration.noise_level)
    assert setup.noise_source.isHidden() is not explicit
    assert setup.noise.toolTip() == (
        "Explicit value from the fit settings" if explicit else "Estimated from all planes"
    )
    assert setup.passes.toolTip()
    assert setup.regions.toolTip()


def test_the_peak_page_says_each_thing_once(workspace, scientific):
    w = workspace
    panel, nav = w.panel, w.navigator
    peak = w.peaks[0]
    w.select_peak(peak.id)
    # Before a fit the toolbar already says Not fitted; the page does not repeat it.
    assert panel.status.isHidden()
    assert "Unfolded ppm" in panel.editor.x_field.toolTip()
    result = scientific[-1]
    publish(w, result)
    w.select_peak(peak.id)
    # The toolbar names the current run; the cluster list says only when it is outdated.
    assert nav.run_label.isHidden()
    warned = sum(bool(r.warnings) for r in nav._reviews)
    assert nav.cluster_count.isHidden() is (warned == 0)
    # The review is label rows sharing the inspector's label column.
    if not warned:
        assert _review_row(w.panel.review, "Warnings") == "none"
    assert _review_row(w.panel.review, "χ²ᵣ")
    # Off-scale observations never use e+ notation.
    assert "e+" not in panel.profile.show_all.toolTip()
    # One left edge for every line, one value column for the form and the review.
    QtWidgets.QApplication.processEvents()

    def left(widget) -> int:
        return widget.mapTo(panel, QtCore.QPoint(0, 0)).x()

    edge = left(panel.kind)
    for toggle in (panel.starts_toggle, panel.profile.about_toggle, panel.provenance_toggle):
        assert abs(left(toggle) - edge) <= 1
    assert left(_review_cell(panel.review, "χ²ᵣ")) == left(panel.editor.x_field)


def test_default_lineshape_fallback_is_a_caution_not_a_block(workspace):
    w, fitting, setup = workspace, workspace.fitting, workspace.panel.setup
    wait_until(lambda: fitting.prepared is not None)
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
    # Linewidth outlier: fitted LW over the run's median for that axis, usable peaks only.
    usable = [c for c in result.outcome.clusters if c.usable]
    widths = [
        p.value for c in usable for p in c.final_nonlinear_parameters if p.name.endswith(".F2.lw")
    ]
    median = float(np.median(widths))
    width, name, cluster = max(
        (
            (p.value, p.name.split(".")[0], c)
            for c in usable
            for p in c.final_nonlinear_parameters
            if p.name.endswith(".F2.lw")
        ),
        key=lambda item: abs(item[0] / median - 1),
    )
    identity = next(i for i in result.inputs.identities if i.engine_name == name)
    ratio = width / median
    nothing = review_module.HighlightRules(1.0, 0.0, 1e9)
    everything = review_module.HighlightRules(0.0, 1.0, 1.0)
    assert not any(r.highlights for r in _review(result, spectra, nothing))
    flagged = next(
        r for r in _review(result, spectra, everything) if r.cluster_id == cluster.cluster_id
    )
    outlier = next(
        h
        for h in flagged.highlights
        if h.kind == "Linewidth outlier" and h.peak == identity.working_id and "axis 0" in h.rule
    )
    assert outlier.rule == f"LW {ratio:.2f} × median (axis 0)"
    # Movement from the initial position is normal optimization, never a highlight.
    kinds = {h.kind for r in _review(result, spectra, everything) for h in r.highlights}
    assert kinds <= {"Amplitude correlation", "Linewidth outlier"}


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
    w.set_rules(review_module.HighlightRules(0.0, 1.0, 1.0))
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
    # The Clusters tab gives the total; the footer counts only a filter.
    if highlighted < total:
        assert navigator.cluster_count.text().startswith(f"{highlighted} of {total} clusters")
    else:
        assert "of" not in navigator.cluster_count.text()
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
    assert "Warnings" in text
    assert review.highlights[0].rule in text


def test_starting_values_show_the_next_fit_inputs(workspace):
    w, fitting = workspace, workspace.fitting
    wait_until(lambda: fitting.prepared is not None)
    peak = w.peaks[0]
    w.select_peak(peak.id)
    w._update_inspector()
    prepared = fitting.prepared
    engine = next(i.engine_name for i in prepared.identities if i.working_id == peak.id)
    expected = [
        p for p in prepared.parameters.values() if p.param_id and p.param_id.peak_name == engine
    ]
    grid = w.panel.start_grid

    def text(row: int, column: int) -> str:
        return grid.itemAtPosition(row, column).widget().text()

    # One line per parameter under Start, Lower and Upper; positions before widths.
    assert [text(0, column) for column in (1, 2, 3)] == ["Start", "Lower", "Upper"]
    rows = range(1, len(expected) + 1)
    assert [text(row, 0).split()[0] for row in rows] == ["δ", "δ", "LW", "LW"]

    def number(row: int, column: int) -> float:
        return float(text(row, column).replace("−", "-"))

    shown = sorted((number(r, 1), number(r, 2), number(r, 3)) for r in rows)
    recorded = sorted((p.value, p.min, p.max) for p in expected)
    np.testing.assert_allclose(shown, recorded, rtol=1e-3, atol=1e-2)
    assert "next fit" in w.panel.starts_note.text()
    # Without a current preparation the section says why instead of showing stale values.
    invalid = fitting.configuration.model_copy(deep=True)
    invalid.exclude_planes = [1]
    fitting.configure(invalid)
    wait_until(lambda: bool(fitting.blocked))
    w._update_inspector()
    assert grid.count() == 0
    assert fitting.blocked in w.panel.starts_note.text()


def test_warning_navigation_never_selects_a_hidden_cluster(workspace, scientific):
    w = workspace
    publish(w, _with_bound(scientific[-1])[0])
    navigator = w.navigator
    # Make every cluster highlighted, then warn about none that the filter shows.
    w.set_rules(review_module.HighlightRules(0.0, 1.0, 1.0))
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


def test_normalized_profile_draws_ratios_and_keeps_recorded_amplitudes(qapp, scientific):
    result = scientific[-1]
    peak = result.inputs.identities[0].working_id
    shown = peak_profile(result, peak, None)
    assert isinstance(shown, PeakProfile)
    recorded = shown.series[0].amplitudes.copy()
    panel = ProfilePlot()
    panel.resize(320, 260)
    panel.show()
    panel.show_profile(shown, 0)
    qapp.processEvents()
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
    assert "±1σ of the ratio, conditional on lineshapes" in panel.note.text()
    assert "full-gradient" in panel.note.toolTip()
    # The reference set persists across selections; recorded amplitudes never change.
    panel.show_profile(shown, 0)
    assert panel.normalized
    np.testing.assert_array_equal(shown.series[0].amplitudes, recorded)
    panel.set_normalized(False)
    _x, y = _selected_points(panel).getData()
    np.testing.assert_array_equal(y, recorded)


def test_unavailable_normalization_keeps_the_raw_profile_with_the_reason(qapp, scientific):
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
    qapp.processEvents()
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


def _withheld(series, status, cause, **extra):
    """The series with a synthetic classification, as the engine would record it."""
    member = MemberAmplitude(status, cause, 0.0, 0.0, 2, 2, 1e-4, **extra)
    return replace(series, identifiability=member)


def test_a_withheld_selected_peak_says_normalization_was_not_applied(qapp, scientific):
    result = scientific[-1]
    cluster = next(c for c in result.outcome.clusters if c.usable and len(c.peak_names) > 1)
    peak = next(
        i.working_id for i in result.inputs.identities if i.engine_name == cluster.peak_names[0]
    )
    shown = peak_profile(result, peak, None)
    assert isinstance(shown, PeakProfile)
    selected, partner = shown.series[0], shown.series[1]
    drawable = _withheld(partner, AmplitudeStatus.IDENTIFIABLE, Cause.NONE)
    unreliable = _withheld(
        selected, AmplitudeStatus.NOT_ESTABLISHED, Cause.NEAR_COLLINEAR, partners=(1,)
    )
    reference = (int(shown.source_planes[0]),)
    panel = ProfilePlot()
    panel.show()
    panel.show_profile(replace(shown, series=(unreliable, drawable)), 0)
    panel.set_reference(reference)
    qapp.processEvents()
    # The partner is drawn raw; the requested ratio is named as not applied, and why.
    assert panel._normalized is None
    assert panel.reference == reference
    assert panel.mode.text() == "÷ plane 1 · not applied"
    assert panel.fallback.isVisible()
    assert "withheld as unreliable" in panel.fallback.text()
    assert f"individual amplitudes of {selected.assignment} are not established" in (
        panel.fallback.text()
    )
    _x, y = _selected_points(panel).getData()
    np.testing.assert_array_equal(y, drawable.amplitudes)
    # A selected amplitude in the null space makes the ratio undefined, not unreliable.
    undetermined = _withheld(selected, AmplitudeStatus.UNIDENTIFIABLE, Cause.NULL_SPACE)
    panel.show_profile(replace(shown, series=(undetermined, drawable)), 0)
    assert "is undefined" in panel.fallback.text()
    assert "are not identifiable" in panel.fallback.text()
    assert "data do not determine the individual amplitude decomposition" in (panel.withheld.text())
    # With nothing drawn, no raw values are claimed.
    panel.show_profile(replace(shown, series=(unreliable,)), 0)
    assert not panel.fallback.isVisible()
    # The reference is kept: a determined selected peak is normalized again.
    panel.show_profile(replace(shown, series=(selected, drawable)), 0)
    assert panel._normalized is not None
    assert not panel.fallback.isVisible()
    panel.close()


def test_unestablished_members_are_withheld_from_the_profile_and_its_axes(qapp, scientific):
    result = scientific[-1]
    cluster = next(c for c in result.outcome.clusters if c.usable and len(c.peak_names) > 1)
    peak = next(
        i.working_id for i in result.inputs.identities if i.engine_name == cluster.peak_names[0]
    )
    shown = peak_profile(result, peak, None)
    assert isinstance(shown, PeakProfile)
    selected, member = shown.series[0], shown.series[1]
    # The member's recorded amplitudes are inflated, as in a degenerate split.
    inflated = _withheld(
        replace(member, amplitudes=member.amplitudes * 1e3, partners=(selected.assignment,)),
        AmplitudeStatus.NOT_ESTABLISHED,
        Cause.NEAR_COLLINEAR,
        partners=(0,),
        dissimilarity=3.3e-8,
    )
    profile = replace(shown, series=(selected, inflated, *shown.series[2:]))
    panel = ProfilePlot()
    panel.show()
    panel.show_profile(profile, 0)
    qapp.processEvents()
    drawn = [i for i in panel.plot.getPlotItem().items if isinstance(i, pg.ScatterPlotItem)]
    assert len(drawn) == len(profile.series) - 1
    _low, high = panel.plot.getViewBox().viewRange()[1]
    assert high < np.max(np.abs(inflated.amplitudes)) / 10
    text = panel.withheld.text()
    assert panel.withheld.isVisible()
    assert f"{inflated.assignment}: reliability not established: near-collinear" in text
    assert "not identifiable" not in text
    assert "total fitted model remains available for comparison" in text
    assert "decomposition is unreliable" in text
    assert panel.legend.text().count("●") == len(profile.series) - 1
    # The recorded values stay reachable, unchanged.
    assert panel.diagnostics_toggle.isVisible()
    assert not panel.diagnostics.isVisible()
    panel.diagnostics_toggle.setChecked(True)
    assert panel.diagnostics.isVisible()
    assert "recorded amplitudes" in panel.diagnostics.text()
    mantissa = f"{float(np.max(inflated.amplitudes)):.2e}".split("e")[0].replace("-", "−")
    assert f"to {mantissa}×10" in panel.diagnostics.text()
    # A withheld member is never normalized; the selected peak still is.
    panel.set_reference((int(profile.source_planes[0]),))
    assert panel._normalized is not None
    # A mathematically unidentifiable member is named as such.
    unidentifiable = _withheld(member, AmplitudeStatus.UNIDENTIFIABLE, Cause.NULL_SPACE)
    panel.show_profile(replace(shown, series=(selected, unidentifiable)), 0)
    assert f"{member.assignment}: not identifiable: the fitted shapes are linearly dependent" in (
        panel.withheld.text()
    )
    # A withheld selected peak with no other member leaves only the explanation.
    alone = _withheld(selected, AmplitudeStatus.NOT_ESTABLISHED, Cause.ILL_CONDITIONED)
    panel.show_profile(replace(shown, series=(alone,)), 0)
    assert not panel.plot.isVisible()
    assert not panel.note.isVisible()
    assert not panel.mode.isVisible()
    assert not panel.show_all.isVisible()
    assert "ill-conditioned fitted shapes" in panel.withheld.text()
    panel.close()
    assert panel.mode.text() == f"÷ plane {int(shown.source_planes[0]) + 1}"


def _page_labels(page) -> list[str]:
    return [label.text() for label in page.findChildren(QtWidgets.QLabel) if label.isVisible()]


def _assert_correlations(grid, matrix, order: list[int]) -> None:
    """The page shows the recorded correlations in member order."""
    shown = [
        [formatting.number(round(float(matrix[a, b]), 2) + 0.0, 2) for b in order] for a in order
    ]
    if len(order) == 2:  # one value: a ρ row with its qualifier underneath
        assert _review_cell(grid, "ρ").text() == shown[0][1]
        assert grid.itemAtPosition(1, 1).widget().text() == "conditional on lineshapes"
        return
    for i in range(len(order)):
        for j in range(len(order)):
            text = grid.itemAtPosition(i + 1, j + 1).widget().text()
            assert text == ("1" if i == j else shown[i][j])


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
    _assert_correlations(panel.cluster_page.matrix, matrix, order)
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
    view = c.box.viewRange()
    # A member link opens that peak's page; the peak page links back to its cluster.
    target = review.ids[-1]
    links[-1].click()  # assignments may repeat; links follow member order
    assert w.selected_id == target
    assert not panel.cluster_view
    assert panel.peak_page.isVisible()
    assert panel.peak_page.isAncestorOf(panel.profile_block)
    assert panel.cluster_link.isVisible()
    assert panel.cluster_link.text() == "Review cluster ›"
    assert panel.kind.text() == f"PEAK · CLUSTER OF {len(members)}"  # the size, said once
    panel.cluster_link.click()
    assert panel.cluster_view
    assert w.selected_id == target
    assert c.box.viewRange() == view
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
        # Nothing to correlate: the section is not shown at all.
        assert w.panel.cluster_page.correlation_title.isHidden()
        assert w.panel.cluster_page.matrix_note.isHidden()


def test_the_cluster_page_lists_each_member_once(workspace, scientific):
    w = workspace
    result = scientific[-1]
    publish(w, result)
    cluster = next(cl for cl in result.outcome.clusters if cl.usable and len(cl.peak_names) > 1)
    identities = {i.engine_name: i.working_id for i in result.inputs.identities}
    w.select_peak(identities[cluster.peak_names[0]], cluster=True)
    page = w.panel.cluster_page
    review = next(r for r in w.reviews() if r.cluster_id == cluster.cluster_id)
    grid = page.members
    # One line per member: its number, a link to its page, then its recorded values.
    for row, name in enumerate(review.names, 1):
        assert grid.itemAtPosition(row, 0).widget().text() == str(row)
        link = grid.itemAtPosition(row, 1).layout().itemAt(0).widget()
        assert link.text() == name
        assert grid.itemAtPosition(row, 2).widget().toolTip().count("±") == 1
    sections = sorted(t for t in _page_labels(page) if t.isupper())
    assert sections == ["AMPLITUDE CORRELATION", "MEMBERS", "REVIEW"]  # no Parameters
    assert page.members_note.text() == "δ in ppm · LW in Hz"
    # Provenance opens like every other disclosure, from the same left edge.
    assert page.provenance_toggle.text() == "▸  Provenance"
    left = page.about_toggle.mapTo(page, QtCore.QPoint(0, 0)).x()
    assert page.provenance_toggle.mapTo(page, QtCore.QPoint(0, 0)).x() == left
    page.provenance_toggle.click()
    assert page.provenance.isVisible()
    assert page.provenance_toggle.text() == "▾  Provenance"
    # The table never pushes the inspector wider than its viewport.
    assert w.panel.widget().width() <= w.panel.viewport().width()


def test_cluster_page_names_numerical_warnings_at_the_top(workspace, scientific):
    w = workspace
    edited, bounded, parameter = _with_bound(scientific[-1])
    publish(w, edited)
    identities = {i.engine_name: i.working_id for i in edited.inputs.identities}
    page = w.panel.cluster_page
    w.select_peak(identities[bounded.peak_names[0]], cluster=True)
    # A compact line names the warning; its explanation stays under Review.
    assert page.warnings.isVisible()
    assert page.warnings.text() == "▲ At bound · see Review"
    assert parameter.name not in page.warnings.text()
    assert parameter.name in _review_text(w)  # the detail stays under Review
    other = next(c for c in edited.outcome.clusters if c.cluster_id != bounded.cluster_id)
    review = next(r for r in w.reviews() if r.cluster_id == other.cluster_id)
    if not review.warnings:
        w.select_peak(identities[other.peak_names[0]], cluster=True)
        assert not page.warnings.isVisible()


def test_the_active_run_reads_as_label_rows(workspace, scientific):
    w = workspace
    result = scientific[-1]
    publish(w, result)
    w.select_peak(None)
    panel = w.panel
    total = len(result.outcome.clusters)
    usable = sum(c.usable for c in result.outcome.clusters)
    assert _review_row(panel.run_rows, "Run") == f"{result.run_id[:8]} · Current"
    assert _review_cell(panel.run_rows, "Run").toolTip() == result.run_id
    assert _review_row(panel.run_rows, "Usable") == f"{usable} / {total}"
    fitted = _review_row(panel.run_rows, "Fitted")
    assert fitted.startswith(result.completed_at[:10] + "\u00a0")  # a date and time, not ISO
    assert fitted.count(" ") == 2  # ordinary spaces only around the dot, where it wraps
    # The run section and the fit setup put their values in one column.
    QtWidgets.QApplication.processEvents()  # lay out the new rows
    x = lambda widget: widget.mapTo(panel.analysis_page, QtCore.QPoint(0, 0)).x()  # noqa: E731
    assert x(_review_cell(panel.run_rows, "Run")) == x(panel.setup.passes)
    # No parent run: no changes section.
    assert panel.changes_title.isHidden()


def test_analysis_page_lists_changes_from_the_parent_run(workspace, scientific):
    w, fitting = workspace, workspace.fitting
    result = scientific[-1]
    publish(w, result)
    w.select_peak(None)
    w.render_fitting()
    first = [label.text() for label in w.panel.analysis_page.findChildren(QtWidgets.QLabel)]
    assert "First run: there is no earlier run to compare with." in first
    cluster = next(c for c in result.outcome.clusters if c.usable and c.analytical_evaluation)
    warned = replace(
        cluster,
        analytical_evaluation=replace(cluster.analytical_evaluation, uncertainty_warning="x"),
    )
    outcome = replace(
        result.outcome,
        clusters=tuple(warned if c is cluster else c for c in result.outcome.clusters),
    )
    parent = replace(
        result,
        run_id="parent0000000000",
        completed_fit=replace(result.completed_fit, outcome=outcome),
    )
    completion = replace(fitting.completion, result=parent)

    started, released = Event(), Event()

    class Stored(StoredRun):
        def load(self):
            # Hold the background read until the transient state has been checked.
            started.set()
            released.wait()
            return completion

    fitting.history = [Stored(Path(), {"run_id": parent.run_id}), *fitting.history]
    fitting.journal = [
        {**e, "parent_run_id": parent.run_id} if e.get("run_id") == result.run_id else e
        for e in fitting.journal
    ]
    panel = w.panel
    texts = lambda: [label.text() for label in panel.analysis_page.findChildren(QtWidgets.QLabel)]  # noqa: E731
    try:
        w.select_peak(None)
        w.render_fitting()
        wait_until(started.is_set)
        # The stored parent is read off Qt; the page says so meanwhile.
        assert "Reading run parent00…" in texts()
    finally:
        released.set()
    wait_until(lambda: fitting._parents.get(parent.run_id) is parent)
    w.render_fitting()
    assert panel.changes_title.text() == "CHANGES FROM PARENT RUN"
    assert "parent00" in texts()  # the run ID keeps its case, as a row
    assert any("▲ Uncertainty unreliable no longer reported" in t for t in texts())
    assert not any("resolved" in t for t in texts())
    links = [
        b
        for b in panel.analysis_page.findChildren(QtWidgets.QPushButton)
        if b.objectName() == "link"
    ]
    assert links
    links[0].click()
    identities = {i.engine_name: i.working_id for i in result.inputs.identities}
    assert w.selected_id in {identities[n] for n in cluster.peak_names}
    assert panel.cluster_view


def test_peak_page_shows_where_the_run_started(workspace, scientific):
    w = workspace
    result = scientific[-1]
    publish(w, result)
    peak = result.inputs.snapshot.peaks[0]
    w.select_peak(peak.id)
    dx, dy = w.panel.editor.decimals
    x_title, y_title = (w.panel._axes[label] for label in w.panel._axis_order)
    expected = (
        f"Started from {x_title} {peak.positions[1]:.{dx}f}, "
        f"{y_title} {peak.positions[0]:.{dy}f} ppm"
    )
    labels = [label.text() for label in w.panel.peak_page.findChildren(QtWidgets.QLabel)]
    assert expected in labels


def test_cluster_focus_frames_compares_and_returns_exactly(workspace, scientific):  # noqa: PLR0915
    w, c = workspace, workspace.canvas
    result = scientific[-1]
    publish(w, result)
    cluster = next(cl for cl in result.outcome.clusters if cl.usable and len(cl.peak_names) > 1)
    identities = {i.engine_name: i.working_id for i in result.inputs.identities}
    member = identities[cluster.peak_names[0]]
    nav = w.navigator
    nav.tree.setCurrentItem(nav._clusters[member])
    c.scale_intensity(120)  # an overview gain, kept aside while in focus
    view, mode, sizes = c.box.viewRange(), c.comparison_mode, w.splitter.sizes()
    shown, sample = c.slice_state(), (c.ix, c.iy, c.pinned, c.pin_peak_id)
    nav.tree.setFocus()
    QtTest.QTest.keyClick(nav.tree, QtCore.Qt.Key.Key_Return)
    assert w.focus_open
    assert w.focus_bar.isVisibleTo(w)
    assert w.focus_bar.title.text().startswith("Focus · ")
    assert w.panel.cluster_view
    assert not w.panel.cluster_page.focus_button.isVisibleTo(w)
    # The one spectral view, reframed in place: every mode stays available.
    assert all(button.isEnabled() for button in w.mode_buttons)
    assert c.sizes.lanes  # focus shows the residual lanes
    assert c.gain == 1  # a new cluster starts unmagnified
    # Default framing: displayed members ± max(3 fitted FWHM, 8 samples), not the masks.
    members = [p for p in w.peaks if p.id in w._focus_ids]
    assert {p.id for p in members} == {identities[n] for n in cluster.peak_names}
    widths = w._fitted_widths()
    assert all(np.isfinite(widths[p.id]).all() for p in members)
    x_range, y_range = slices.focus_ranges(
        [p.positions for p in members], [widths[p.id] for p in members], (c.y_axis, c.x_axis)
    )
    (x_low, x_high), (y_low, y_high) = c.box.viewRange()
    assert (x_low, x_high) == pytest.approx(sorted(x_range))
    assert (y_low, y_high) == pytest.approx(sorted(y_range))
    for p in members:
        x, y = c.display(p)
        assert x_low < x < x_high
        assert y_low < y < y_high
    # The fitting region is a separate toggle, shared with the setup's.
    w.focus_bar.regions.click()
    assert c.regions_visible
    assert w.panel.setup.regions.isChecked()
    # The residual is the whole-spectrum residual of the overview: (D − M) / σ_stat.
    w.set_mode(2)
    wait_until(lambda: not c.contour_pending)
    axes = [axis.model_copy(deep=True) for axis in c.spectra.spectral_params]
    model = dataset_model_plane(result, axes, c.plane, cancel=Event())
    expected = (c.spectra.data[c.plane] - model.values) / result.outcome.noise
    np.testing.assert_allclose(c.curve_x.getData()[1], expected[c.iy, :])
    # Full spectrum (Ctrl+0) reframes the cluster while in focus.
    framed = c.box.viewRange()
    c.set_ranges((x_low, (x_low + x_high) / 2), (y_low, y_high))
    w.full_view()
    np.testing.assert_allclose(c.box.viewRange(), framed)
    # Focus follows the plane.
    w.set_plane(2)
    wait_until(lambda: not c.contour_pending)
    assert c.plane == 1
    # A committed edit makes the run Outdated: its model stays, labelled historical.
    peak = next(p for p in w.peaks if p.id == member)
    edited = (peak.positions[0], peak.positions[1] + 0.001)
    w.move_peak(peak.id, edited)
    wait_until(lambda: not c.contour_pending)
    tag = f"Run {result.run_id[:8]} · historical"
    assert "Outdated · model and residual are historical" in w.focus_bar.note.text()
    assert tag in c.legend.text()
    assert w.focus_open
    # Escape returns to the overview exactly: viewport, mode, slices, scale, sample,
    # regions and panel widths; the edit made in focus stays.
    c.view.setFocus()
    QtTest.QTest.keyClick(c.view, QtCore.Qt.Key.Key_Escape)
    assert not w.focus_open
    assert not w.focus_bar.isVisibleTo(w)
    assert w.panel.cluster_page.focus_button.isVisibleTo(w)
    assert c.box.viewRange() == view
    assert c.comparison_mode == mode
    assert c.slice_state() == shown
    assert (c.ix, c.iy, c.pinned, c.pin_peak_id) == sample
    assert not c.regions_visible
    assert w.splitter.sizes() == sizes
    assert next(p for p in w.peaks if p.id == member).positions == edited
    assert not w.fitting.current


def _open_focus(w, result):
    cluster = next(cl for cl in result.outcome.clusters if cl.usable and len(cl.peak_names) > 1)
    identities = {i.engine_name: i.working_id for i in result.inputs.identities}
    w.open_focus(identities[cluster.peak_names[0]])
    wait_until(lambda: not w.canvas.contour_pending)
    return cluster


def test_display_settings_apply_in_focus_without_touching_results(
    workspace, scientific, monkeypatch
):
    w, c = workspace, workspace.canvas
    result = scientific[-1]
    publish(w, result)
    view = c.box.viewRange()
    _open_focus(w, result)
    w.set_mode(2)
    wait_until(lambda: not c.contour_pending)
    chosen = {"First contour": c.contour_threshold * 2, "First residual contour": 5.0}

    def accept(dialog):
        for field in dialog.findChildren(QtWidgets.QDoubleSpinBox):
            if field.accessibleName() in chosen:
                field.setValue(chosen[field.accessibleName()])
        return QtWidgets.QDialog.DialogCode.Accepted

    jobs = c.contour_jobs
    monkeypatch.setattr(QtWidgets.QDialog, "exec", accept)
    w.display_dialog()
    # Both contour sets apply at once, whatever the mode; the focused view redraws.
    assert c.contour_threshold == chosen["First contour"]
    assert c.residual_threshold == chosen["First residual contour"]
    wait_until(lambda: not c.contour_pending)
    assert c.contour_jobs > jobs
    # Display only: same run, still Current; focus and its frame stay.
    assert w.fitting.result is result
    assert w.fitting.current
    assert w.focus_open
    w.close_focus()
    assert c.box.viewRange() == view


def test_escape_leaves_focus_from_any_column_but_discards_a_draft_first(workspace, scientific):
    w = workspace
    result = scientific[-1]
    publish(w, result)
    _open_focus(w, result)
    # From the cluster list.
    w.navigator.tree.setFocus()
    QtTest.QTest.keyClick(w.navigator.tree, QtCore.Qt.Key.Key_Escape)
    assert not w.focus_open
    # From the inspector: a draft is discarded first, then Escape leaves focus.
    _open_focus(w, result)
    w.select_peak(w.selected_id)  # a member link: the peak page, still in focus
    assert w.focus_open
    assert w.panel.peak_page.isVisible()
    field = w.inspector.x_field
    assert field.isVisible()
    field.setFocus()
    QtTest.QTest.keyClicks(field, "1")
    assert w.inspector.pending
    QtTest.QTest.keyClick(field, QtCore.Qt.Key.Key_Escape)
    assert not w.inspector.pending
    assert w.focus_open
    QtTest.QTest.keyClick(field, QtCore.Qt.Key.Key_Escape)
    assert not w.focus_open


def _unexplained_links(w):
    layout = w.panel.unexplained
    return [
        widget
        for index in range(layout.count())
        if (row := layout.itemAt(index).layout()) is not None
        and isinstance(widget := row.itemAt(0).widget(), QtWidgets.QPushButton)
    ]


def test_views_compute_each_model_plane_once(workspace, scientific, monkeypatch):
    w, c = workspace, workspace.canvas
    result = scientific[-1]
    computed = []

    def counted(run, axes, plane, *, cancel):
        computed.append(plane)
        return dataset_model_plane(run, axes, plane, cancel=cancel)

    for module in (canvas_module, workspace_module):
        monkeypatch.setattr(module, "dataset_model_plane", counted)
    publish(w, result)  # the overview's Overlay computes plane 1
    wait_until(lambda: not c.contour_pending)
    assert computed == [0]
    # The residual search computes plane 3 while the overview shows only the experiment.
    w.set_mode(0)
    w.set_plane(3)
    wait_until(lambda: not c.contour_pending)
    w.select_peak(None)
    w.find_unexplained()
    wait_until(lambda: w._unexplained[1] != "pending")
    assert computed == [0, 2]
    # The view reuses it, in focus too: the same immutable plane, not a recomputation.
    w.set_mode(2)
    wait_until(lambda: not c.contour_pending)
    assert c._model is w.model_planes.get(result.run_id, 2)
    _open_focus(w, result)
    assert computed == [0, 2]


def test_unexplained_signal_is_listed_on_request_and_frames_without_editing(  # noqa: PLR0915
    workspace, scientific, monkeypatch
):
    w, c = workspace, workspace.canvas
    result = scientific[-1]
    publish(w, result)
    w.select_peak(None)
    panel = w.panel
    assert panel.unexplained_button.text() == "Find on plane 1"
    assert panel.unexplained_button.isEnabled()
    assert _unexplained_links(w) == []  # nothing is computed until asked
    assert "beyond ±5 σ_stat" in panel.unexplained_note.text()

    # A low Display threshold, so the synthetic noise crosses it.
    def accept(dialog):
        for field in dialog.findChildren(QtWidgets.QDoubleSpinBox):
            if field.accessibleName() == "Unexplained signal beyond":
                field.setValue(2.0)
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(QtWidgets.QDialog, "exec", accept)
    w.display_dialog()
    assert w.unexplained_threshold == 2.0
    assert "beyond ±2 σ_stat" in panel.unexplained_note.text()
    revision, peaks, mode = w.edits.revision, w.peaks, c.comparison_mode
    panel.unexplained_button.click()
    wait_until(lambda: w._unexplained[1] != "pending")
    assert isinstance(w._unexplained[1], list), w._unexplained
    regions = w._unexplained[1]
    # Exactly the regions of the whole-spectrum residual (D − M) / σ_stat.
    axes = [axis.model_copy(deep=True) for axis in c.spectra.spectral_params]
    model = dataset_model_plane(result, axes, 0, cancel=Event())
    residual = (c.spectra.data[0] - model.values) / result.outcome.noise
    positions = [
        (
            p.id,
            float(c.y_axis.ppm2pts(display_position(c.y_axis, p.positions[0]))),
            float(c.x_axis.ppm2pts(display_position(c.x_axis, p.positions[1]))),
        )
        for p in w.peaks
    ]
    assert regions == unexplained_regions(residual, 2.0, positions)
    assert regions
    count = len(regions)
    assert panel.unexplained_title.text() == (
        f"Plane 1 · {count} region{'s' if count != 1 else ''} beyond ±2 σ_stat"
    )
    assert not panel.unexplained_button.isEnabled()
    # The new list scrolls into view below the fit setup.
    section = panel.unexplained_section
    wait_until(lambda: section.mapTo(panel.viewport(), QtCore.QPoint(0, 0)).y() >= 0)
    assert section.mapTo(panel.viewport(), QtCore.QPoint(0, 0)).y() < panel.viewport().height()
    links = _unexplained_links(w)
    assert len(links) == min(count, 8)
    first = regions[0]
    name = next(p.name for p in w.peaks if p.id == first.nearest)
    detail = links[0].parent().findChildren(QtWidgets.QLabel)
    assert any(
        label.text().startswith(f"{first.extreme:+.1f} σ · {first.size} sample")
        and label.text().endswith(f"{first.distance:.1f} pts from {name}")
        for label in detail
    )
    # Choosing a region frames it and samples through its extreme; nothing is edited.
    links[0].click()
    (x_low, x_high), (y_low, y_high) = c.box.viewRange()
    for low, high, ppm, (start, stop) in (
        (x_low, x_high, c.ppm_x, first.columns),
        (y_low, y_high, c.ppm_y, first.rows),
    ):
        assert low <= min(ppm[start], ppm[stop]) <= max(ppm[start], ppm[stop]) <= high
    assert (c.iy, c.ix) == (first.row, first.column)
    assert (w.edits.revision, w.peaks, c.comparison_mode) == (revision, peaks, mode)
    assert w.selected_id is None
    assert w.fitting.current
    # Another plane needs its own request; the old list is not shown for it.
    w.set_plane(2)
    assert panel.unexplained_button.text() == "Find on plane 2"
    assert panel.unexplained_button.isEnabled()
    assert _unexplained_links(w) == []
    # An Outdated run's list stays available, labelled with its run.
    peak = w.peaks[0]
    w.move_peak(peak.id, (peak.positions[0], peak.positions[1] + 0.001))
    w.select_peak(None)
    assert panel.unexplained_button.isEnabled()
    assert panel.unexplained_note.text().startswith(
        f"Residual of Run {result.run_id[:8]} · historical. Distances are to the current"
    )


def _view_point(c, x: float, y: float) -> QtCore.QPoint:
    """Viewport pixel of a ppm position in the spectral view's 2D plot."""
    return c.view.mapFromScene(c.box.mapViewToScene(QtCore.QPointF(x, y)))


def _refit(w):
    """A real whole-dataset fit of the current working inputs, published as Fit would."""
    prepared = prepare_dataset_fit(
        w.dataset.spectra, w.peaks, w.fitting.configuration.model_copy(deep=True)
    )
    prepared = replace(
        prepared, snapshot=replace(prepared.snapshot, source_files=w.dataset.sources)
    )
    result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    # Publishing a real-data run writes its recovery copy in a spawned process;
    # allow for a contended CI runner, as the project tests do.
    publish(w, result, timeout=60)
    return result


def test_correct_and_refit_in_focus(workspace, scientific):  # noqa: PLR0915
    w, c = workspace, workspace.canvas
    result = scientific[-1]
    publish(w, result)
    w.open_focus("1")
    wait_until(lambda: not c.contour_pending)
    pair = w._focus_ids
    assert set(pair) == {"0", "1"}
    before = {p.id: p.positions for p in w.peaks}
    revision = w.edits.revision

    # A drag is a draft: ghost at the proposal, accepted model and markers kept.
    c._move_peak(0.01, 0.02, False)
    assert w.pending
    x, y = c.ghost.getData()
    assert (x[0], y[0]) == pytest.approx((before["1"][1] + 0.01, before["1"][0] + 0.02))
    assert c.markers.getData()[0].tolist() == [
        display_position(c.x_axis, p.positions[1]) for p in w.peaks
    ]
    assert w.fitting.current
    assert c._model is not None
    assert w.edits.revision == revision
    c._move_peak(0.01, 0.02, True)
    moved = next(p for p in w.peaks if p.id == "1").positions
    assert moved == (before["1"][0] + 0.02, before["1"][1] + 0.01)  # exact, unsnapped
    assert w.edits.revision == revision + 1
    assert not w.fitting.current
    assert c.ghost.getData()[0].size == 0
    assert w.focus_open
    assert w._focus_ids == pair

    # The folded peak moves in unfolded ppm; its marker stays at the aliased position.
    w.select_peak("0", cluster=True)
    c._move_peak(-0.005, 0.01, True)
    folded = next(p for p in w.peaks if p.id == "0").positions
    assert folded == (before["0"][0] + 0.01, before["0"][1] - 0.005)
    marker_y = c.markers.getData()[1][[p.id for p in w.peaks].index("0")]
    assert marker_y == display_position(c.y_axis, folded[0])
    assert marker_y != folded[0]

    # Undo and Redo are the workspace's; focus stays on its cluster.
    w.history(False)
    assert next(p for p in w.peaks if p.id == "0").positions == before["0"]
    w.history(True)
    assert next(p for p in w.peaks if p.id == "0").positions == folded
    assert w.focus_open
    assert w._focus_ids == pair

    # Placement: A arms it, Escape cancels it without leaving focus.
    c.view.setFocus()
    QtTest.QTest.keyClick(c.view, QtCore.Qt.Key.Key_A)
    assert c.placing
    QtTest.QTest.keyClick(c.view, QtCore.Qt.Key.Key_Escape)
    assert not c.placing
    assert w.focus_open
    # A click in the focused plot adds a peak at its exact position.
    count = len(w.peaks)
    w.start_placing()
    (x0, x1), (y0, y1) = c.box.viewRange()
    point = _view_point(c, (x0 + x1) / 2, (y0 + y1) / 2)
    QtTest.QTest.mouseClick(c.view.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=point)
    assert len(w.peaks) == count + 1
    added = w.peaks[-1]
    assert w.selected_id == added.id
    clicked = c.box.mapSceneToView(c.view.mapToScene(point))
    assert added.positions == (clicked.y(), clicked.x())
    # The new peak is not in the run, yet focus keeps its cluster and says so.
    assert w.focus_open
    assert w._focus_ids == pair
    assert f"{added.name} is not in this cluster of the run" in w.focus_bar.note.text()
    assert not c.placing
    # Undoing the addition keeps focus too.
    w.history(False)
    assert len(w.peaks) == count
    assert w.focus_open
    assert w._focus_ids == pair

    # A failed or canceled fit keeps the edits and focus.
    edited = w.peaks
    w.fitting._finished(0, RuntimeError("worker died"))
    assert w.fitting.failure == "worker died"
    w.fitting.cancel()
    assert w.peaks == edited
    assert w.focus_open
    assert w._focus_ids == pair

    # Move peak 1 next to the remote peak: the refit splits the pair and merges 1.
    w.select_peak("1", cluster=True)
    remote = next(p for p in w.peaks if p.id == "remote").positions
    current = next(p for p in w.peaks if p.id == "1").positions
    c._move_peak(remote[1] - current[1] + 0.02, remote[0] - current[0] + 0.03, True)
    refit = _refit(w)
    wait_until(lambda: not c.contour_pending)
    identities = {i.engine_name: i.working_id for i in refit.inputs.identities}
    now = next(
        tuple(identities[n] for n in cl.peak_names)
        for cl in refit.outcome.clusters
        if "1" in {identities[n] for n in cl.peak_names}
    )
    assert set(now) == {"1", "remote"}
    assert w.focus_open
    assert w._focus_ids == now
    assert w.fitting.current
    bar = w.focus_bar
    text = bar.membership_plain
    assert text.startswith(f"Regrouped by run {refit.run_id[:8]} · was Duplicate +1")
    assert "joined by Duplicate" in text
    assert text.endswith("now fitted in other clusters: Duplicate ›")
    assert '<a href="0">Duplicate ›</a>' in bar.membership.text()
    # The link opens the cluster the previous member is now fitted in.
    bar.membership.linkActivated.emit("0")
    assert w.selected_id == "0"
    assert "0" in w._focus_ids
    assert bar.membership.isVisibleTo(w)
    # Leaving focus reverts nothing: the refit and its edits stay.
    w.close_focus()
    assert w.fitting.result is refit
    assert next(p for p in w.peaks if p.id == "1").positions != before["1"]


def test_an_added_peak_keeps_focus_context_and_is_followed_after_refit(workspace, scientific):
    w, c = workspace, workspace.canvas
    publish(w, scientific[-1])
    w.open_focus("remote")
    wait_until(lambda: not c.contour_pending)
    assert w._focus_ids == ("remote",)
    # Placed inside the cluster's frame: focus keeps the cluster it shows.
    remote = next(p for p in w.peaks if p.id == "remote").positions
    w.start_placing()
    w.place_peak((remote[0] + 0.02, remote[1] + 0.02))
    added = w.peaks[-1]
    assert w.selected_id == added.id
    assert w._focus_ids == ("remote",)
    # Outside it, an unfitted peak shows the cluster with its nearest member.
    one = next(p for p in w.peaks if p.id == "1")
    far = w.edits.add((one.positions[0] + 0.001, one.positions[1] + 0.001))
    w._refresh()
    w.select_peak(far.id, cluster=True)
    assert set(w._focus_ids) == {"0", "1"}
    assert w.focus_open
    # Remove it again; refit with the first added peak selected.
    w.history(False)
    w.select_peak(added.id, cluster=True)
    assert w._focus_ids == ("remote",)
    refit = _refit(w)
    wait_until(lambda: not c.contour_pending)
    # It joined the remote peak's cluster: a merge, said without a link back.
    assert set(w._focus_ids) == {"remote", added.id}
    assert (
        w.focus_bar.membership_plain == f"Run {refit.run_id[:8]} fits {added.name} with Duplicate"
    )


def test_outdated_run_stays_a_labelled_historical_comparison(workspace, scientific):  # noqa: PLR0915
    w, c = workspace, workspace.canvas
    result = scientific[-1]
    publish(w, result)
    w.set_mode(2)
    wait_until(lambda: not c.contour_pending)
    tag = f"Run {result.run_id[:8]} · historical"
    axes = [axis.model_copy(deep=True) for axis in c.spectra.spectral_params]
    model = dataset_model_plane(result, axes, c.plane, cancel=Event())
    residual = (c.spectra.data[c.plane] - model.values) / result.outcome.noise
    recorded = next(p for p in w.fitting.completion.working_peaks if p.id == "1")

    # Two corrections in a row without refitting: a move and an added peak.
    w.move_peak("1", (recorded.positions[0] + 0.01, recorded.positions[1] + 0.01))
    added = w.edits.add((recorded.positions[0] + 0.05, recorded.positions[1] - 0.05))
    w._refresh()
    w.select_peak(added.id)
    assert not w.fitting.current
    comparison = w.fitting.comparison
    assert (comparison.result, comparison.historical, comparison.label) == (result, True, tag)
    assert comparison.added == {added.id}
    assert comparison.departed == (recorded,)
    # The overview keeps the run's residual, labelled in the legend and sample text.
    assert all(button.isEnabled() for button in w.mode_buttons)
    assert w.mode_buttons[2].toolTip() == f"Compares with {tag}"
    assert c.comparison_mode == 2
    wait_until(lambda: not c.contour_pending)
    np.testing.assert_array_equal(c._model.values, model.values)
    c.sample(float(c.ppm_x[12]), float(c.ppm_y[13]))
    np.testing.assert_array_equal(c.curve_x.getData()[1], residual[13, :])
    assert f"σ_stat residual · residual of {tag}" in c.view.accessibleDescription()
    legend = c.legend.text()
    assert tag in legend
    assert "not in this model" in legend
    assert "position in this model" in legend
    # ◇ marks the added peak; × the run's position of the moved one, linked to it.
    assert list(zip(*c.added_marks.getData(), strict=True)) == [c.display(added)]
    assert list(zip(*c.recorded_marks.getData(), strict=True)) == [c.display(recorded)]
    moved = next(p for p in w.peaks if p.id == "1")
    assert list(zip(*c.recorded_links.getData(), strict=True)) == [
        c.display(recorded),
        c.display(moved),
    ]
    x, y = c.display(added)
    assert c._marker_tip(x=x, y=y, data=added.id).endswith(f"Not in the model of {tag}")
    # The Experimental view shows no model, so no model-relative marks either.
    w.set_mode(0)
    assert c.added_marks.getData()[0].size == 0
    assert tag not in c.legend.text()
    w.set_mode(1)
    assert "model · " + tag in c.legend.text()

    # Focus compares with the same historical model and marks the same peaks.
    w.set_mode(2)
    w.open_focus("1")
    wait_until(lambda: not c.contour_pending)
    np.testing.assert_array_equal(c._model.values, model.values)
    assert "model and residual are historical" in w.focus_bar.note.text()
    assert c.added_marks.getData()[0].size == 1
    assert c.recorded_marks.getData()[0].size == 1
    assert tag in c.legend.text()
    assert "not in this model" in c.legend.text()
    assert "position in this model" in c.legend.text()
    w.close_focus()

    # The unexplained list keeps the run's residual; distances follow working peaks.
    w.select_peak(None)
    w.unexplained_threshold = 2.0
    w.panel.unexplained_button.click()
    wait_until(lambda: w._unexplained[1] != "pending")
    regions = w._unexplained[1]
    assert isinstance(regions, list)
    assert regions
    first = regions[0]
    assert w.panel.unexplained_title.text().endswith(f"beyond ±2 σ_stat · {tag}")
    nearest, distance = nearest_peak(first.row, first.column, w._native_peaks())
    names = {p.id: p.name for p in w.peaks}
    detail = _unexplained_links(w)[0].parent().findChildren(QtWidgets.QLabel)
    assert any(f"{distance:.1f} pts from {names[nearest]}" in label.text() for label in detail)

    # Undo restores the run's peaks: no marks, yet the run stays Outdated.
    w.history(False)
    w.history(False)
    assert w.peaks == w.fitting.completion.working_peaks
    assert not w.fitting.current
    assert w.fitting.comparison.historical
    assert c.added_marks.getData()[0].size == 0
    assert c.recorded_marks.getData()[0].size == 0

    # New plane values: the run's planes no longer match, so the comparison withdraws.
    values = w.dataset.spectra.z_values + 1.0
    spectra = w.dataset.spectra.model_copy(update={"z_values": values})
    w.dataset = w.canvas.dataset = replace(w.dataset, spectra=spectra)
    w.canvas.spectra = spectra
    w.edits.revision += 1
    w.render_fitting()
    reason = f"Run {result.run_id[:8]} was recorded with different plane values"
    assert w.fitting.comparison.reason == reason
    assert c.comparison_mode == 0
    assert c.comparison_result is None
    assert not w.mode_buttons[2].isEnabled()
    assert w.mode_buttons[2].toolTip() == f"Unavailable: {reason}"
    assert w.panel.unexplained_title.text() == f"Unavailable: {reason}"
    # Focus shows the experiment only, and the modes still say why.
    w.open_focus("1")
    assert w.focus_open
    assert c.comparison_mode == 0
    assert c.comparison_result is None
    assert w.mode_buttons[2].toolTip() == f"Unavailable: {reason}"


def test_run_summary_opens_the_review_which_leads_the_analysis_page_after_a_fit(
    workspace, scientific
):
    w = workspace
    panel = w.panel
    layout = panel._analysis_layout
    w.select_peak(None)
    # Before the first fit the setup leads and there is no unexplained-signal section.
    assert layout.indexOf(panel.setup_block) == 0
    assert not panel.unexplained_button.isVisible()
    publish(w, scientific[-1])
    w.select_peak(w.peaks[0].id)
    assert not panel.analysis_page.isVisible()
    # A click on the toolbar run summary clears the selection and opens the review.
    QtTest.QTest.mouseClick(w.summary, QtCore.Qt.MouseButton.LeftButton)
    wait_until(lambda: panel.analysis_page.isVisible())
    assert w.selected_id is None
    assert layout.indexOf(panel.review_block) == 0
    assert panel.unexplained_button.isVisible()
    assert "Click to open the analysis page" in w.summary.toolTip()
    review = panel.review_block.mapTo(panel.viewport(), QtCore.QPoint(0, 0)).y()
    assert 0 <= review < panel.viewport().height()
    # The settings chip still reaches the setup, now below the review.
    QtTest.QTest.mouseClick(w.settings_button, QtCore.Qt.MouseButton.LeftButton)
    wait_until(
        lambda: 0
        <= panel.setup_block.mapTo(panel.viewport(), QtCore.QPoint(0, 0)).y()
        < panel.viewport().height()
    )


def test_enter_on_a_name_returns_focus_so_a_places_the_next_peak(workspace, scientific):
    w = workspace
    publish(w, scientific[-1])
    w.open_focus("1")
    wait_until(lambda: not w.canvas.contour_pending)
    w.start_placing()
    remote = next(p for p in w.peaks if p.id == "remote").positions
    w.place_peak((remote[0] + 0.02, remote[1] + 0.02))
    name = w.inspector.name
    name.setFocus()
    name.selectAll()
    QtTest.QTest.keyClicks(name, "U1")
    QtTest.QTest.keyClick(name, QtCore.Qt.Key.Key_Return)
    assert w.peaks[-1].name == "U1"
    focus = QtWidgets.QApplication.focusWidget()
    assert focus is not None
    assert focus is w.canvas.view or w.canvas.view.isAncestorOf(focus)
    QtTest.QTest.keyClick(focus, QtCore.Qt.Key.Key_A)
    assert w.canvas.placing
    # A coordinate field keeps focus after Enter, showing the exact committed value.
    w.cancel_gesture()
    field = w.inspector.x_field
    field.setFocus()
    field.setText(repr(w.peaks[-1].positions[1] + 0.001))
    QtTest.QTest.keyClick(field, QtCore.Qt.Key.Key_Return)
    assert QtWidgets.QApplication.focusWidget() is field


def test_preview_is_prepared_again_when_only_the_continuation_source_changes(workspace, scientific):
    w = workspace
    result = scientific[-1]
    publish(w, result)
    fitting = w.fitting
    wait_until(lambda: fitting.prepared is not None)
    first = fitting.prepared
    assert first.continuation.run_id == result.run_id
    assert first.continuation.continued  # unchanged peaks continue from the run
    # Same working peaks and settings, but a different active run.
    other = replace(fitting.completion, result=replace(fitting.completion.result, run_id="other"))
    peaks, settings = w.peaks, fitting.configuration.model_dump_json()
    fitting.completion = other
    w._refresh()
    assert (w.peaks, fitting.configuration.model_dump_json()) == (peaks, settings)
    assert fitting.prepared is None  # the previous preparation is never reused
    wait_until(lambda: fitting.prepared is not None)
    assert fitting.prepared.continuation.run_id == "other"


@pytest.fixture
def bundled_workspace(qapp):
    """The editor on the bundled CEST data, whose near-collinear pairs reproduce on
    every platform tested (an exact synthetic duplicate does not). They collapse only
    from broad linewidth starts, so the configuration starts every width at 20 Hz."""
    data = Path(__file__).resolve().parents[1] / "examples/02-advanced-fitting/data"
    dataset = read_dataset(data / "pseudo3d.ft2", data / "pseudo3d.list", data / "b1_offsets.txt")
    widget = EditorWorkspace(dataset)
    widget.fitting.configuration = PeakFitConfig(
        fitting=FitConfig(refine_iterations=2),
        parameters=ParameterConfig.model_validate({"defaults": {"*.*.lw": {"value": 20.0}}}),
    )
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


@pytest.mark.slow
def test_near_collinear_peaks_keep_working_positions_and_say_why(bundled_workspace):
    w = bundled_workspace
    ids = {p.name: p.id for p in w.peaks}
    before = {p.id: p.positions for p in w.peaks}
    _refit(w)
    completion = w.fitting.completion
    retained = dict(completion.retained)
    assert retained[ids["53N-H"]] == "near-collinear fitted shapes with 147N-H"
    assert retained[ids["147N-H"]] == "near-collinear fitted shapes with 53N-H"
    # Current, with the working markers deliberately not following the recorded fit.
    assert w.fitting.current
    after = {p.id: p.positions for p in w.peaks}
    for name in ("53N-H", "147N-H"):
        assert after[ids[name]] == before[ids[name]]
    assert len(w.canvas.unadopted.data) == len(retained)
    assert "fitted position, not adopted" in w.canvas.legend.text()
    # A small model-coloured □, distinct from the ◇ of a peak absent from a historical model.
    assert w.canvas.unadopted.opts["symbol"] == "s"
    assert w.canvas.added_marks.opts["symbol"] == "d"
    # The peak page states it, and the review lists the pair as a measure.
    w.select_peak(ids["53N-H"])
    labels = [label.text() for label in w.panel.peak_page.findChildren(QtWidgets.QLabel)]
    kept = "Fitted position not adopted: near-collinear fitted shapes with 147N-H"
    assert any(text.startswith(kept) for text in labels)
    assert _review_row(w.panel.review, "Collinear") == "53N-H / 147N-H"
    assert "□ Fitted position not adopted · working position kept" in w.panel.status.text()
    withheld = w.panel.profile.withheld.text()
    assert "53N-H and 147N-H: reliability not established: near-collinear" in withheld
    assert "not identifiable" not in withheld
    # Moving one restores ordinary initialization; its partner stays retained.
    moved = next(p for p in w.peaks if p.id == ids["147N-H"])
    w.move_peak(moved.id, (moved.positions[0] + 0.002, moved.positions[1]))
    wait_until(lambda: w.fitting.prepared is not None, timeout=60)
    prepared = w.fitting.prepared
    assert prepared is not None
    assert prepared.continuation is not None
    reasons = dict(prepared.continuation.reasons)
    run = completion.result.run_id[:8]
    assert reasons[moved.id] == f"moved since run {run}"
    assert reasons[ids["53N-H"]].startswith(f"not adopted by run {run}: near-collinear")
    # Focus on the pair keeps its □ marks and their key.
    w.open_focus(ids["53N-H"])
    wait_until(lambda: not w.canvas.contour_pending, timeout=60)
    assert w.focus_open
    assert len(w.canvas.unadopted.data) == len(retained)
    assert "fitted position, not adopted" in w.canvas.legend.text()


def test_qualifiers_stay_beside_values_and_explanations_open_in_place(workspace, scientific):
    w = workspace
    result = scientific[-1]
    publish(w, result)
    cluster = next(c for c in result.outcome.clusters if c.usable and len(c.peak_names) > 1)
    identities = {i.engine_name: i.working_id for i in result.inputs.identities}
    member = identities[cluster.peak_names[0]]
    w.select_peak(member)
    panel = w.panel
    profile = panel.profile
    note = profile.note.text()
    # Conditional σ and the noise treatment stay visible as short qualifiers; χ²ᵣ is
    # one of the cluster's measures.
    assert "±1σ conditional on lineshapes" in note
    assert "noise correlation not modelled" in note
    assert "χ²ᵣ" not in note
    # The full explanation opens in place, by keyboard, not only as a tooltip.
    toggle = profile.about_toggle
    assert toggle.isVisible()
    assert not profile.about.isVisible()
    toggle.setFocus()
    QtTest.QTest.keyClick(toggle, QtCore.Qt.Key.Key_Space)
    assert profile.about.isVisible()
    assert "never rescaled by χ²ᵣ" in profile.about.text()
    assert "held fixed" in profile.about.text()
    # Measures: the reference maximum says what it assumes and is not a threshold.
    residual = _review_cell(panel.review, "Max residual")
    assert "(guide " in residual.text()
    assert "independent Gaussian samples" in residual.toolTip()
    # The correlation keeps its qualifier on screen, just under the value.
    row = next(
        r
        for r in range(panel.review.rowCount())
        if (item := panel.review.itemAtPosition(r, 0)) and item.widget().text().startswith("ρ ")
    )
    assert panel.review.itemAtPosition(row + 1, 1).widget().text() == "conditional on lineshapes"
    assert "not a significance threshold" in panel.about.text()
    assert "correlated" in panel.about.text()
    # The measures' explanation keeps its state across selections and pages.
    panel.about_toggle.click()
    assert panel.about.isVisible()
    w.select_peak(identities[cluster.peak_names[1]])
    assert panel.about.isVisible()
    w.select_peak(member, cluster=True)
    assert panel.cluster_page.about.isVisible()
    # The profile's explanation stays open too (the profile fills once on show).
    assert profile.about_toggle.isChecked()
    assert not profile.about.isHidden()
    assert "see About these measures" in panel.cluster_page.matrix_note.toolTip()
    # Editing instructions are in the fields; the unfolded qualifier is shown beside a
    # displayed alias, where it changes what the numbers mean, and is otherwise a tooltip.
    w.select_peak(member)
    assert panel.editor.details.text().startswith("Unfolded ppm · displayed alias")
    assert "Unfolded ppm" in panel.editor.x_field.toolTip()
    assert "Escape discards" in panel.editor.x_field.toolTip()
    # The unexplained-signal list is framed as a diagnostic, never peak picking.
    w.select_peak(None)
    assert panel.unexplained_note.text().startswith("Diagnostic only")
    assert "never proposes peaks" in panel.unexplained_note.toolTip()


def test_the_error_bar_explanation_describes_the_quantity_on_show(qapp, scientific):
    result = scientific[-1]
    shown = peak_profile(result, result.inputs.identities[0].working_id, None)
    assert isinstance(shown, PeakProfile)
    panel = ProfilePlot()
    panel.resize(320, 260)
    panel.show()
    panel.show_profile(shown, 0)
    qapp.processEvents()
    common = ("held fixed", "never rescaled by χ²ᵣ", shown.noise_model.description)

    def explanation():
        text = panel.about.text()
        assert all(part in text for part in common)
        return text

    # Raw: the recorded σ, one per peak, the same on every plane.
    assert "the same on every plane" in explanation()
    assert "ratio" not in explanation()
    # One reference plane: the ratio's propagated σ, varying by plane; reference exact.
    first, second = (int(p) for p in shown.source_planes[:2])
    panel.set_reference((first,))
    text = explanation()
    assert "ratio A_k / R" in text
    assert "differ from plane to plane" in text
    assert "single reference plane its ratio is exactly 1 and has no error bar" in text
    assert "the same on every plane" not in text
    # Several reference planes: each has a propagated error of its own.
    panel.set_reference((first, second))
    assert "its own ratio has a propagated error" in explanation()
    assert "exactly 1" not in explanation()
    # Back to raw amplitudes.
    panel.set_normalized(False)
    assert "the same on every plane" in explanation()
    panel.close()
