"""Scientific round trips and transactional GUI export protection."""

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from test_gui_fit_execution import _example, _wait

from peakfit.engine.algorithms.evaluation import FitOutcomeClassification
from peakfit.gui.app import PeakFitWindow
from peakfit.gui.fit_export import export_local_fit
from peakfit.gui.fit_process import FitProcessResult
from peakfit.gui.local_fit import execute_local_fit
from peakfit.gui.spectrum import verified_source_files
from peakfit.gui.viewer import SpectrumViewer
from peakfit.io.readers.results import ResultsLoader

pytestmark = pytest.mark.gui


@pytest.mark.parametrize("classification", ["converged", "usable_non_converged", "unusable"])
def test_export_authoritative_values_and_original_identities(tmp_path, monkeypatch, classification):  # noqa: PLR0915
    _spectra, working, prepared, config = _example()
    if classification == "usable_non_converged":
        config.fitting.max_iterations = 1
        prepared = replace(
            prepared, snapshot=replace(prepared.snapshot, configuration=config.model_dump_json())
        )
    if classification == "unusable":
        from peakfit.gui import local_fit  # noqa: PLC0415

        real_fit = local_fit.fit_single_cluster_task

        def unusable_attempt(task):
            result = real_fit(task)
            result.cost = np.nan
            return result

        monkeypatch.setattr(local_fit, "fit_single_cluster_task", unusable_attempt)
    source = tmp_path / "original.ft2"
    source.write_bytes(b"verified original input")
    with verified_source_files({"spectrum": source}) as records:
        prepared = replace(prepared, snapshot=replace(prepared.snapshot, source_files=records))
    completed = execute_local_fit(prepared)
    cluster = completed.outcome.clusters[0]
    assert cluster.classification == FitOutcomeClassification(classification)
    path = export_local_fit(completed, tmp_path / "export")
    loader = ResultsLoader(path)
    summary = loader.load_summary()
    raw = json.loads((path / "summary/fit.json").read_text())
    companion = json.loads((path / "summary/local_fit.json").read_text())
    assert raw["schema_version"] == "4.0.0"
    assert summary.clusters[0].classification == classification
    assert summary.metadata.timestamp.isoformat() == completed.completed_at
    assert summary.metadata.software_version == completed.software_version
    assert summary.metadata.configuration == json.loads(prepared.snapshot.configuration)
    assert summary.metadata.input_files["spectrum"]["path"] == str(source)
    assert "checksum_sha256" in summary.metadata.input_files["spectrum"]
    assert companion["run_id"] == completed.run_id
    assert companion["cross_cluster_corrections"] is False
    assert companion["planes"] == [
        {"index": 0, "value": 2.0},
        {"index": 1, "value": 0.0},
        {"index": 2, "value": 2.0},
    ]
    assert "spectrum_id" not in companion
    assert companion["noise"] == prepared.noise
    assert companion["contour_threshold"] == prepared.contour_level
    assert companion["shape_names"] == list(prepared.shape_names)
    assert companion["peak_identities"] == [
        {"engine_name": f"p_{i}", "working_id": peak.id, "assignment": peak.name}
        for i, peak in enumerate(working)
    ]
    assert companion["grid_indices"] == [index.tolist() for index in prepared.scope.grid_indices]
    assert (
        summary.clusters[0].optimizer_provenance.termination_message
        == cluster.optimizer_provenance.termination_message
    )
    for actual, expected in zip(
        summary.final_nonlinear_parameters,
        completed.outcome.final_nonlinear_parameters,
        strict=True,
    ):
        assert actual.value == expected.value
        assert actual.vary == expected.vary
        assert actual.min_bound == (expected.min if np.isfinite(expected.min) else None)
        assert actual.max_bound == (expected.max if np.isfinite(expected.max) else None)
        assert actual.standard_error == (
            expected.standard_error if np.isfinite(expected.standard_error) else None
        )
    evaluation = cluster.analytical_evaluation
    with (path / "tables/intensities.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if evaluation is None:
        assert rows == []
        assert summary.clusters[0].analytical_evaluation is None
        assert summary.clusters[0].unusable_reason == cluster.unusable_reason
        assert summary.final_nonlinear_parameters == []
    else:
        loaded = summary.clusters[0].analytical_evaluation
        assert loaded is not None
        np.testing.assert_array_equal(loaded.amplitudes, evaluation.amplitudes)
        np.testing.assert_array_equal(
            loaded.amplitude_standard_errors, evaluation.amplitude_standard_errors
        )
        np.testing.assert_array_equal(
            loaded.scaled_amplitude_standard_errors, evaluation.scaled_amplitude_standard_errors
        )
        for row in rows:
            peak_index = cluster.peak_names.index(row["peak_name"])
            plane = int(row["plane_index"])
            assert float(row["intensity"]) == pytest.approx(
                evaluation.amplitudes[peak_index, plane], rel=1e-15
            )
            assert float(row["intensity_err"]) == pytest.approx(
                evaluation.scaled_amplitude_standard_errors[peak_index], rel=1e-15
            )
        state = loader.load_fitting_state()
        assert set(state.scalar_params) == {p.name for p in cluster.final_nonlinear_parameters}
    before = (path / "summary/fit.json").read_bytes()
    with pytest.raises(FileExistsError):
        export_local_fit(completed, path)
    assert (path / "summary/fit.json").read_bytes() == before


def test_partial_failure_has_no_published_directory(tmp_path, monkeypatch):
    _spectra, _working, prepared, _config = _example()
    completed = execute_local_fit(prepared)
    from peakfit.gui import fit_export  # noqa: PLC0415

    def fail(_outcome, output, *_args, **_kwargs):
        output.mkdir()
        (output / "partial").write_text("partial output")
        raise OSError("disk full")

    monkeypatch.setattr(fit_export, "write_fit_outputs", fail)
    with pytest.raises(OSError, match="disk full"):
        export_local_fit(completed, tmp_path / "failed")
    assert list(tmp_path.iterdir()) == []
    assert completed.inputs is not None


@pytest.mark.parametrize(
    "publication", ["success", "rename_failure", "late_empty", "late_populated", "locked"]
)
def test_directory_publication_and_cleanup(tmp_path, monkeypatch, publication):
    from peakfit.gui import fit_export  # noqa: PLC0415

    _spectra, _working, prepared, _config = _example()
    completed = execute_local_fit(prepared)
    destination = tmp_path / "export"
    lock = tmp_path / ".export.export-lock"
    write = fit_export.write_fit_outputs
    rename = Path.rename
    published = []

    def write_then_compete(*args, **kwargs):
        files = write(*args, **kwargs)
        if publication.startswith("late_"):
            destination.mkdir()
            if publication == "late_populated":
                (destination / "existing-result").write_text("keep me")
        return files

    def windows_directory_move(staged, target):
        # Windows refuses replacing even an empty directory. Enforce this on macOS/Linux too.
        assert not target.exists()
        assert staged.parent.parent == target.parent  # Staging stays on the same filesystem.
        ResultsLoader(staged).load_summary()
        assert (staged / "summary/local_fit.json").is_file()
        assert (staged / "tables/intensities.csv").is_file()
        assert lock.is_dir()
        if publication == "rename_failure":
            raise OSError("publication denied")
        published.append(target)
        return rename(staged, target)

    monkeypatch.setattr(fit_export, "write_fit_outputs", write_then_compete)
    monkeypatch.setattr(Path, "rename", windows_directory_move)
    monkeypatch.setattr(Path, "replace", windows_directory_move)
    if publication == "locked":
        lock.mkdir()
    if publication == "success":
        assert export_local_fit(completed, destination) == destination
        assert published == [destination]
        ResultsLoader(destination).load_summary()
    else:
        with pytest.raises(OSError, match=r"publication denied|[Ee]xists"):
            export_local_fit(completed, destination)
        assert not published
    expected = [lock] if publication == "locked" else []
    if publication in {"success", "late_empty", "late_populated"}:
        expected.append(destination)
    assert sorted(tmp_path.iterdir()) == sorted(expected)
    if publication == "late_populated":
        assert (destination / "existing-result").read_text() == "keep me"


def test_long_export_path_does_not_change_footer_layout(tmp_path, monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    spectra, working, prepared, _config = _example()
    viewer = SpectrumViewer(spectra)
    viewer.set_peaks(working)
    viewer.fit_result = FitProcessResult(execute_local_fit(prepared), 0, 0, 0, 0)
    viewer.resize(1000, 720)
    viewer.show()
    app.processEvents()
    # Layout starts with the same compact wording: only the destination length changes.
    viewer.result_notice.setText("Fit results exported")
    app.processEvents()
    before = (
        viewer.size(),
        viewer.minimumSizeHint(),
        viewer.spectrum_plot.size(),
        viewer.trace_x.size(),
    )
    destination = tmp_path.joinpath(*(["long-export-directory"] * 30))
    messages = []
    monkeypatch.setattr("peakfit.gui.viewer.choose_and_export", lambda *_: destination)
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information", lambda *args: messages.append(args[-1])
    )
    try:
        assert viewer.export_fit_results()
        app.processEvents()
        assert viewer.result_notice.text() == "Fit results exported"
        assert viewer.result_notice.toolTip() == str(destination)
        assert str(destination) in messages[0]
        assert (
            viewer.size(),
            viewer.minimumSizeHint(),
            viewer.spectrum_plot.size(),
            viewer.trace_x.size(),
        ) == before
    finally:
        _wait(lambda: not viewer._computation._busy)
        viewer.stop_preparation()
        viewer.close()
        viewer.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        app.processEvents()


def test_export_and_close_keep_result_and_peak_states_separate(tmp_path, monkeypatch):  # noqa: PLR0915
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    spectra, working, prepared, _config = _example()
    scientific = execute_local_fit(prepared)
    window = PeakFitWindow()
    viewer = SpectrumViewer(spectra)
    window.setCentralWidget(viewer)
    window.viewer = viewer
    viewer.set_peaks(working)
    viewer.edits.update(
        working[0].id, working[0].name, (working[0].positions[0] + 0.001, working[0].positions[1])
    )
    retained = FitProcessResult(scientific, 0, 0, 0, 0)
    viewer.fit_result = retained
    viewer._result_revision = -1  # Outdated inputs are still exportable.
    before = viewer.peaks
    viewer.show_fit_result()
    try:
        monkeypatch.setattr("peakfit.gui.viewer.choose_and_export", lambda *_: None)
        assert not viewer.export_fit_results()
        assert not viewer.result_exported
        monkeypatch.setattr(
            viewer, "_result_loss_choice", lambda _: QtWidgets.QMessageBox.StandardButton.Save
        )
        assert not viewer.confirm_result_loss("discard")
        assert viewer.fit_result is retained
        # Exercise the real off-thread disk path and responsive Qt event loop.
        from peakfit.gui.export_dialog import choose_and_export  # noqa: PLC0415

        monkeypatch.setattr("peakfit.gui.viewer.choose_and_export", choose_and_export)
        monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", lambda *_: str(tmp_path))
        monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *_: None)
        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_: None)
        beats = []
        timer = QtCore.QTimer(viewer)
        timer.setInterval(1)
        timer.timeout.connect(lambda: beats.append(1))
        timer.start()
        assert viewer.export_fit_results()
        timer.stop()
        assert beats
        assert viewer.result_exported
        assert viewer.fit_result is retained
        assert viewer.peaks == before
        assert viewer.edits.dirty
        assert viewer.confirm_result_loss("replace")  # No result prompt after exporting.
        # Existing directory fails; an earlier success remains a success for that exact result.
        assert not viewer.export_fit_results()
        assert viewer.result_exported
        # New result identity has independent export state, even with identical fitted values.
        viewer.fit_result = replace(
            retained, scientific=replace(scientific, run_id="another-result")
        )
        assert not viewer.result_exported
        # A failed export of this new result cannot inherit the previous saved state.
        (tmp_path / "local-fit-another-result").mkdir()
        assert not viewer.export_fit_results()
        assert not viewer.result_exported
        (tmp_path / "local-fit-another-result").rmdir()
        monkeypatch.setattr(
            viewer, "_result_loss_choice", lambda _: QtWidgets.QMessageBox.StandardButton.Cancel
        )
        close = QtGui.QCloseEvent()
        monkeypatch.setattr(window, "_protect_edits", lambda: True)
        window.closeEvent(close)
        assert not close.isAccepted()
        assert not viewer._closing
        assert viewer.fit_result is not None
        monkeypatch.setattr(
            viewer, "_result_loss_choice", lambda _: QtWidgets.QMessageBox.StandardButton.Save
        )
        assert viewer.confirm_result_loss("discard")
        assert viewer.result_exported
        window.closeEvent(close)
        assert close.isAccepted()
    finally:
        _wait(lambda: not viewer._computation._busy)
        viewer.stop_preparation()
        window.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        app.processEvents()
