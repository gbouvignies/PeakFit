"""Plane exclusion in the desktop editor (issue #216): an explicit, undoable model edit.

An excluded plane stays in navigation and in the profiles, measured but not
fitted; the run records the exclusion and a reopened project is checked against it.
"""

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from builders import two_cluster_dataset
from gui_helpers import wait_until
from PySide6 import QtCore

from peakfit.gui import formatting, project
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.editor.axis_table import DIFFERS
from peakfit.gui.editor.model_page import Setting
from peakfit.gui.editor.profile import not_fitted_text, peak_profile
from peakfit.gui.editor.window import EditorWindow
from peakfit.gui.editor.workspace import EditorWorkspace
from peakfit.gui.fit_display import dataset_model_plane
from peakfit.gui.fit_positions import dataset_completion
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.peaks import WorkingEdits
from peakfit.gui.project import Analysis, dataset_header, open_analysis, save_analysis
from peakfit.gui.spectrum import Dataset

pytestmark = pytest.mark.gui

EXCLUDED = 1  # zero-based: plane 2 of 3


@pytest.fixture(scope="module")
def scientific():
    """The three-plane dataset and a run that excluded plane 2."""
    spectra, peaks, config, _prepared = two_cluster_dataset()
    config = config.model_copy(update={"exclude_planes": [EXCLUDED]})
    prepared = prepare_dataset_fit(spectra, peaks, config)
    result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    return spectra, peaks, config, result


@pytest.fixture
def workspace(qapp, scientific):
    spectra, peaks, config, _result = scientific
    widget = EditorWorkspace(Dataset(Path("synthetic.ft2"), spectra, peaks, (), True, 0.01))
    widget.edits = WorkingEdits(widget.peaks, config.model_copy(update={"exclude_planes": []}))
    widget.fitting.model_seen = True
    widget.resize(1280, 800)
    widget.show()
    qapp.processEvents()
    yield widget
    widget.stop()
    widget.close()
    widget.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    qapp.processEvents()


def _publish(workspace, result):
    fitting = workspace.fitting
    fitting._submitted_revision = workspace.edits.revision
    fitting.process.state = "completed"
    fitting._finished(0, result)
    wait_until(lambda: not fitting.busy)
    assert fitting.result is result, fitting.failure


def test_excluding_the_current_plane_is_one_undoable_edit_kept_in_navigation(workspace):
    w = workspace
    action = w.exclude_plane_action
    w.set_plane(2)
    assert action.text() == "Exclude plane 2 from the fit"
    assert not w.plane_excluded.isVisible()
    action.trigger()
    assert w.fitting.configuration.exclude_planes == [EXCLUDED]
    assert w.undo_action.text() == "Undo settings change"
    # Still navigated and shown as measured, marked not fitted.
    assert w.canvas.plane == EXCLUDED
    assert w.plane_excluded.isVisible()
    assert action.text() == "Include plane 2 in the fit"
    model = w.panel.model_page
    assert model.excluded.text() == "2"
    assert model.planes.text() == "2 of 3 fitted"
    w.set_plane(1)
    assert not w.plane_excluded.isVisible()
    w.history(False)
    assert w.fitting.configuration.exclude_planes == []
    assert model.excluded.text() == ""
    w.set_plane(2)
    assert not w.plane_excluded.isVisible()


def test_the_last_fitted_plane_cannot_be_excluded(workspace):
    w = workspace
    settings = w.fitting.configuration.model_copy(update={"exclude_planes": [0, 2]})
    w.configure(settings)
    w.set_plane(2)
    assert not w.exclude_plane_action.isEnabled()
    w.set_plane(1)
    assert w.exclude_plane_action.isEnabled()  # including one again is always possible


def test_each_new_model_setting_is_one_undoable_edit(workspace):
    w = workspace
    model = w.panel.model_page
    wait_until(lambda: w.fitting.prepared is not None)
    start = w.fitting.configuration

    def commit(edit, text):
        edit.setText(text)
        edit.editingFinished.emit()

    model.noise_mode.setCurrentIndex(model.noise_mode.findData("manual"))
    commit(model.noise_value, "0.003")
    model.threshold_mode.setCurrentIndex(model.threshold_mode.findData("level"))
    commit(model.level_value, "0.2")
    commit(model.excluded, "3")
    model.constraints_toggle.setChecked(True)
    commit(model.constraints.windows["F2"], "0.5")
    config = w.fitting.configuration
    assert config.noise_level == 0.003
    assert config.clustering.contour_level == 0.2
    assert config.exclude_planes == [2]
    assert config.parameters.position_windows.F2 == 0.5
    # Undo steps back one setting at a time; the mode switches were their own edits.
    w.history(False)
    assert w.fitting.configuration.parameters.position_windows.F2 is None
    assert model.constraints.windows["F2"].text() == ""
    w.history(False)
    assert w.fitting.configuration.exclude_planes == []
    w.history(False)
    assert w.fitting.configuration.clustering.contour_level != 0.2
    while w.edits.can_undo:
        w.history(False)
    assert w.fitting.configuration == start
    # The start's own sources: a manual noise and an explicit level.
    assert model.noise_mode.currentData() == "manual"
    assert model.noise_value.text() == formatting.setting(0.002) == "2 × 10⁻³"
    assert model.threshold_mode.currentData() == "level"


def test_a_run_that_excluded_a_plane_shows_it_measured_not_fitted(workspace, scientific):
    w = workspace
    spectra, _peaks, config, result = scientific
    w.configure(config)  # the run's own settings: it is Current once published
    _publish(w, result)
    assert w.fitting.current
    # The spectrum: the excluded plane is navigated; the comparison says why it has no model.
    w.set_plane(EXCLUDED + 1)
    w.set_mode(1)
    wait_until(lambda: not w.canvas.contour_pending)
    label = w.canvas.pending_label
    assert label.isVisible()
    assert "plane 2 is measured but not fitted" in label.toPlainText()
    with pytest.raises(ValueError, match="Plane 2 is measured but not fitted"):
        dataset_model_plane(result, spectra.spectral_params, EXCLUDED, cancel=Event())
    # Fitted planes map to their own columns.
    third = dataset_model_plane(result, spectra.spectral_params, 2, cancel=Event())
    assert third.values.shape == (32, 32)
    # The profiles keep the plane, marked.
    profile = peak_profile(result, w.peaks[0].id, tuple(float(z) for z in spectra.z_values))
    assert not isinstance(profile, str)
    np.testing.assert_array_equal(profile.source_planes, [0, 2])
    np.testing.assert_array_equal(profile.excluded, [EXCLUDED])
    assert profile.excluded_x.tolist() == [float(spectra.z_values[EXCLUDED])]
    assert profile.plane_values  # still the recorded plane values, not plane numbers
    assert not_fitted_text(profile) == "plane 2 measured, not fitted"
    w.select_peak(w.peaks[0].id)
    w._update_inspector()
    assert "plane 2 measured, not fitted" in w.panel.profile.note.text()
    # The Run page shows the run's settings as submitted.
    w.select_peak(None)
    w.show_run_page()
    w._update_inspector()
    rows = {row.label: row.value for row in w.panel.run_page._key[0]}
    assert rows["Planes"] == "2 of 3 fitted · 2 excluded"
    assert rows["Noise"].endswith("· manual")
    assert rows["Threshold"] == "level 0.1"


def test_changing_the_exclusion_marks_the_run_outdated_and_the_planes_row(workspace, scientific):
    w = workspace
    _spectra, _peaks, config, result = scientific
    w.configure(config)
    _publish(w, result)
    w.set_plane(1)
    w.exclude_plane_action.trigger()
    assert not w.fitting.current
    model = w.panel.model_page
    w._update_inspector()
    assert model.titles[Setting.PLANES].text() == f"Planes {DIFFERS}"
    assert model.titles[Setting.PLANES].toolTip() == (
        f"Run {result.run_id[:8]} used planes 2 excluded"
    )
    # The Outdated run stays comparable on the planes it fitted.
    assert w.fitting.comparison.result is result
    assert w.fitting.comparison.historical


def test_a_saved_project_records_the_exclusion_and_reopens_current(
    qapp, scientific, tmp_path, monkeypatch
):
    spectra, _peaks, config, result = scientific
    source = tmp_path / "source.ft2"
    source.write_bytes(b"External spectrum stand-in; scientific reader mocked")
    stat = source.stat()
    record = json.dumps(
        {
            "path": str(source),
            "checksum_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "size": str(stat.st_size),
            "mtime_ns": str(stat.st_mtime_ns),
            "ctime_ns": str(stat.st_ctime_ns),
            "inode": str(stat.st_ino),
            "device": str(stat.st_dev),
        }
    )
    sources = (("spectrum", record),)
    snapshot = replace(result.inputs.snapshot, source_files=sources)
    result = replace(result, inputs=replace(result.inputs, snapshot=snapshot))
    completion = dataset_completion(result, snapshot, submitted_revision=0, manual_revision=0)
    dataset = Dataset(source, spectra, completion.working_peaks, sources, True, 0.01)
    analysis = Analysis(
        dataset,
        dataset_header(dataset),
        completion.working_peaks,
        config.model_dump_json(),
        0,
        (completion,),
        {"plane": 1},
    )
    assert analysis.current
    path = save_analysis(analysis, tmp_path / "excluded.peakfit")
    monkeypatch.setattr(project, "read_dataset", lambda _, **_kw: dataset)
    loaded = open_analysis(path)
    assert loaded.dataset is not None, loaded.reason
    assert loaded.current
    reopened = loaded.completion.result
    assert reopened.outcome.excluded_planes == (EXCLUDED,)
    assert reopened.inputs.snapshot.source_plane_indices == (0, 2)
    assert json.loads(loaded.configuration)["exclude_planes"] == [EXCLUDED]
    window = EditorWindow()
    window.install_analysis(loaded)
    w = window.workspace
    assert w is not None
    assert w.fitting.current
    assert w.fitting.configuration.exclude_planes == [EXCLUDED]
    assert w.exclude_plane_action in window.analysis_menu.actions()
    window._saved_state = window._analysis_state()
    window.close()
    qapp.processEvents()
