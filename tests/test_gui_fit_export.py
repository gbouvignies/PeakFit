"""Scientific round trips and transactional export of completed dataset fits."""

import csv
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets
from test_gui_dataset import _example

from peakfit.engine.algorithms.evaluation import FitOutcomeClassification
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.fit_export import export_fit_result
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.spectrum import verified_source_files
from peakfit.io.readers.results import ResultsLoader
from peakfit.io.scientific import read_scientific

pytestmark = pytest.mark.gui


def _prepare(configure=lambda _config: None):
    spectra, working, config = _example()
    config.clustering.contour_level = 0.01
    configure(config)
    prepared = prepare_dataset_fit(spectra, working, config)
    assert len(prepared.clusters) == 1
    return working, prepared


def _fit(prepared):
    return finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))


@pytest.mark.parametrize("classification", ["converged", "usable_non_converged", "unusable"])
def test_export_authoritative_values_and_original_identities(tmp_path, monkeypatch, classification):  # noqa: PLR0915
    def configure(config):
        config.output.formats = ["json", "csv"]
        config.output.detailed_evidence = True
        if classification == "usable_non_converged":
            config.fitting.max_iterations = 1

    working, prepared = _prepare(configure)
    if classification == "unusable":
        from peakfit.fit import pipeline  # noqa: PLC0415

        real_fit = pipeline.fit_single_cluster_task

        def unusable_attempt(task):
            result = real_fit(task)
            result.cost = np.nan
            return result

        monkeypatch.setattr(pipeline, "fit_single_cluster_task", unusable_attempt)
    source = tmp_path / "original.ft2"
    source.write_bytes(b"verified original input")
    with verified_source_files({"spectrum": source}) as records:
        prepared = replace(prepared, snapshot=replace(prepared.snapshot, source_files=records))
    completed = _fit(prepared)
    cluster = completed.outcome.clusters[0]
    assert cluster.classification == FitOutcomeClassification(classification)
    path = export_fit_result(completed, tmp_path / "export")
    loader = ResultsLoader(path)
    summary = loader.load_summary()
    raw = json.loads((path / "summary/fit.json").read_text())
    _outcome, samples, companion = read_scientific(path / "scientific.npz")
    assert raw["schema_version"] == "4.1.0"
    assert summary.clusters[0].classification == classification
    assert summary.metadata.timestamp.isoformat() == completed.completed_at
    assert summary.metadata.software_version == completed.software_version
    assert summary.metadata.configuration == json.loads(prepared.snapshot.configuration)
    assert summary.metadata.input_files["spectrum"]["path"] == str(source)
    assert "checksum_sha256" in summary.metadata.input_files["spectrum"]
    assert companion["plane_values"].tolist() == [2.0, 0.0, 2.0]
    assert companion["peak_identities"] == [
        {"engine_name": f"p_{i}", "peak_id": peak.id, "assignment": peak.name}
        for i, peak in enumerate(working)
    ]
    for old, new in zip(prepared.clusters[0].grid_indices, samples[0].grid_indices, strict=True):
        np.testing.assert_array_equal(old, new)
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
        export_fit_result(completed, path)
    assert (path / "summary/fit.json").read_bytes() == before


def test_partial_failure_has_no_published_directory(tmp_path, monkeypatch):
    _working, prepared = _prepare()
    completed = _fit(prepared)
    from peakfit.gui import fit_export  # noqa: PLC0415

    def fail(_outcome, output, *_args, **_kwargs):
        output.mkdir()
        (output / "partial").write_text("partial output")
        raise OSError("disk full")

    monkeypatch.setattr(fit_export, "write_completed_outputs", fail)
    with pytest.raises(OSError, match="disk full"):
        export_fit_result(completed, tmp_path / "failed")
    assert list(tmp_path.iterdir()) == []
    assert completed.inputs is not None


@pytest.mark.parametrize(
    "publication", ["success", "rename_failure", "late_empty", "late_populated", "locked"]
)
def test_directory_publication_and_cleanup(tmp_path, monkeypatch, publication):
    from peakfit.gui import fit_export  # noqa: PLC0415

    _working, prepared = _prepare()
    completed = _fit(prepared)
    destination = tmp_path / "export"
    lock = tmp_path / ".export.export-lock"
    write = fit_export.write_completed_outputs
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
        assert (staged / "scientific.npz").is_file()
        assert (staged / "tables/intensities.csv").is_file()
        assert lock.is_dir()
        if publication == "rename_failure":
            raise OSError("publication denied")
        published.append(target)
        return rename(staged, target)

    monkeypatch.setattr(fit_export, "write_completed_outputs", write_then_compete)
    monkeypatch.setattr(Path, "rename", windows_directory_move)
    monkeypatch.setattr(Path, "replace", windows_directory_move)
    if publication == "locked":
        lock.mkdir()
    if publication == "success":
        assert export_fit_result(completed, destination) == destination
        assert published == [destination]
        ResultsLoader(destination).load_summary()
    else:
        with pytest.raises(OSError, match=r"publication denied|[Ee]xists"):
            export_fit_result(completed, destination)
        assert not published
    expected = [lock] if publication == "locked" else []
    if publication in {"success", "late_empty", "late_populated"}:
        expected.append(destination)
    assert sorted(tmp_path.iterdir()) == sorted(expected)
    if publication == "late_populated":
        assert (destination / "existing-result").read_text() == "keep me"


def test_export_shows_progress_before_starting_writer(tmp_path, monkeypatch):
    from threading import Event  # noqa: PLC0415

    from peakfit.gui import export_dialog  # noqa: PLC0415

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    parent = QtWidgets.QWidget()
    parent.show()
    app.processEvents()
    shown = Event()
    observed = []

    class Progress(export_dialog._ExportProgress):
        def showEvent(self, event):  # noqa: N802
            super().showEvent(event)
            shown.set()

    _working, prepared = _prepare()
    completed = _fit(prepared)

    def write(*_args, **_kwargs):
        observed.append(shown.is_set())
        return tmp_path

    monkeypatch.setattr(export_dialog, "_ExportProgress", Progress)
    monkeypatch.setattr(export_dialog, "export_fit_result", write)
    monkeypatch.setattr(
        export_dialog, "_choose_export_directory", lambda *_a, **_k: (str(tmp_path), False)
    )
    assert export_dialog.choose_and_export(parent, completed) == tmp_path
    assert observed == [True]
    parent.close()
    parent.deleteLater()


def test_export_picker_has_one_evidence_option_and_honors_configuration(tmp_path):
    from peakfit.gui.export_dialog import _choose_export_directory  # noqa: PLC0415

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert isinstance(app, QtWidgets.QApplication)
    observed = []

    def accept():
        dialog = next(
            w
            for w in app.topLevelWidgets()
            if isinstance(w, QtWidgets.QFileDialog) and w.isVisible()
        )
        checks = [
            w for w in dialog.findChildren(QtWidgets.QCheckBox) if "detailed evidence" in w.text()
        ]
        assert len(checks) == 1
        observed.append(checks[0].isChecked())
        dialog.selectFile(str(tmp_path))
        dialog.accept()

    QtCore.QTimer.singleShot(0, accept)
    directory, detailed = _choose_export_directory(None, True)
    assert Path(directory) == tmp_path
    assert detailed
    assert observed == [True]
