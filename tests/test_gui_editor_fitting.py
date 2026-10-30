"""Protect the fit → optimized working state → native comparison → refit loop."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import MappingProxyType

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets
from test_gui_dataset import _dataset
from test_gui_editor import wait_for

import peakfit.gui.editor.canvas as canvas_module
from peakfit.engine.algorithms.evaluation import FitOutcomeClassification
from peakfit.fit.storage import read_scientific
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.editor import fitting as fitting_module
from peakfit.gui.editor.window import EditorWindow
from peakfit.gui.editor.workspace import EditorWorkspace
from peakfit.gui.fit_display import dataset_model_plane
from peakfit.gui.fit_export import export_fit_result
from peakfit.gui.fit_process import FitProcessResult
from peakfit.gui.spectrum import Dataset

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
