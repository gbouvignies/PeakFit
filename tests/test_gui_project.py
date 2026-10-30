"""Lossless evidence, linked source identity and transactional analysis saves."""

import hashlib
import json
import os
import shutil
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from PySide6 import QtWidgets
from test_gui_dataset import _dataset

from peakfit.engine.domain.config import PeakFitConfig
from peakfit.fit import pipeline
from peakfit.gui import fitting, project, spectrum
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.editor.numerical import NumericalReview
from peakfit.gui.editor.window import EditorWindow
from peakfit.gui.fit_positions import dataset_completion
from peakfit.gui.project import Analysis, dataset_header, open_analysis, save_analysis
from peakfit.gui.project_process import project_operation
from peakfit.gui.project_results import journal_entry, verify_model
from peakfit.gui.spectrum import Dataset, read_dataset

DATA = Path(__file__).resolve().parents[1] / "examples/02-advanced-fitting/data"
pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def evidence():
    spectra, _peaks, config, prepared = _dataset()
    result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    completion = dataset_completion(
        result, prepared.snapshot, submitted_revision=0, manual_revision=0
    )
    return spectra, config, completion


@pytest.fixture
def analysis(evidence, tmp_path):
    spectra, config, completion = evidence
    source = tmp_path / "source.ft2"
    source.write_bytes(
        b"External spectrum stand-in; scientific reader mocked only in synthetic tests"
    )
    record = json.dumps(
        {"path": str(source), "checksum_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    )
    sources = (("spectrum", record),)
    snapshot = replace(completion.result.inputs.snapshot, source_files=sources)
    result = replace(completion.result, inputs=replace(completion.result.inputs, snapshot=snapshot))
    completion = replace(completion, result=result)
    dataset = Dataset(source, spectra, completion.working_peaks, sources, True, 5.0, 0.01)
    return Analysis(
        dataset,
        dataset_header(dataset),
        completion.working_peaks,
        config.model_dump_json(),
        0,
        (completion,),
        {"plane": 1},
    )


def test_lossless_numerical_round_trip_no_optimization(analysis, tmp_path, monkeypatch):
    path = save_analysis(analysis, tmp_path / "test.peakfit")
    monkeypatch.setattr(
        project, "read_dataset", lambda _path, **_kw: (_ for _ in ()).throw(OSError("missing"))
    )
    loaded = open_analysis(path)
    assert loaded.dataset is None
    assert "missing" in loaded.reason
    assert loaded.peaks == analysis.peaks
    old, new = analysis.completion.result, loaded.completion.result
    assert new.inputs.snapshot.peaks == old.inputs.snapshot.peaks
    assert new.inputs.identities == old.inputs.identities
    assert [
        (p.name, p.value, p.min, p.max, p.vary, p.unit) for p in new.inputs.parameters.values()
    ] == [(p.name, p.value, p.min, p.max, p.vary, p.unit) for p in old.inputs.parameters.values()]
    for a, b in zip(old.completed_fit.samples, new.completed_fit.samples, strict=True):
        np.testing.assert_array_equal(a.data, b.data)
        np.testing.assert_array_equal(a.neighbor_corrections, b.neighbor_corrections)
        assert not b.data.flags.writeable
    for a, b in zip(old.outcome.clusters, new.outcome.clusters, strict=True):
        assert a.final_nonlinear_parameters == b.final_nonlinear_parameters
        np.testing.assert_array_equal(
            a.analytical_evaluation.amplitudes, b.analytical_evaluation.amplitudes
        )
        np.testing.assert_array_equal(
            a.analytical_evaluation.raw_residuals, b.analytical_evaluation.raw_residuals
        )
    verify_model(new)
    assert loaded.completion.working_peaks != new.inputs.snapshot.peaks
    assert all(
        p.stat().st_size < analysis.dataset.spectra.data.nbytes * 20
        for p in path.rglob("*")
        if p.is_file()
    )


def test_routine_save_zero_source_reads_zero_evidence_writes(analysis, tmp_path, monkeypatch):
    path = save_analysis(analysis, tmp_path / "test.peakfit")
    files = {
        p: (p.stat().st_mtime_ns, p.read_bytes()) for p in (path / "runs").rglob("*") if p.is_file()
    }

    def forbidden(*_args, **_kwargs):
        pytest.fail("Routine Save must not hash sources or rewrite unchanged evidence")

    monkeypatch.setattr(project, "_digest", forbidden)
    monkeypatch.setattr(project, "_write_run", forbidden)
    save_analysis(replace(analysis, path=path), path)
    assert all((p.stat().st_mtime_ns, p.read_bytes()) == state for p, state in files.items())


def test_atomic_save_failure_and_cancel_preserve_last_generation(analysis, tmp_path, monkeypatch):
    path = save_analysis(analysis, tmp_path / "test.peakfit")
    manifest = (path / "manifest.json").read_bytes()
    original = project._json

    def disk_full(target, value):
        if target.name.startswith(".manifest-"):
            raise OSError("disk full")
        original(target, value)

    monkeypatch.setattr(project, "_json", disk_full)
    with pytest.raises(OSError, match="disk full"):
        save_analysis(replace(analysis, path=path), path)
    assert (path / "manifest.json").read_bytes() == manifest
    monkeypatch.setattr(project, "_json", original)
    with pytest.raises(project.CancelledError):
        save_analysis(replace(analysis, path=path), path, canceled=lambda: True)
    assert (path / "manifest.json").read_bytes() == manifest
    assert open_analysis(path).completion is not None


def test_corrupt_and_unsupported_project_rejected(analysis, tmp_path):
    path = save_analysis(analysis, tmp_path / "test.peakfit")
    payload = next(path.rglob("samples.npz"))
    payload.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt project evidence"):
        open_analysis(path)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["schema_version"] = "999"
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsupported analysis version"):
        open_analysis(path)


def test_verified_reopen_rebinds_currentness_and_manual_edit_outdates(
    analysis, tmp_path, monkeypatch
):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    path = save_analysis(analysis, tmp_path / "test.peakfit")
    monkeypatch.setattr(project, "read_dataset", lambda _, **_kw: analysis.dataset)
    loaded = open_analysis(path)
    assert loaded.dataset is not None, loaded.reason
    window = EditorWindow()
    window.install_analysis(loaded)
    w = window.workspace
    assert w is not None
    assert w.fitting.current
    peak = w.peaks[0]
    w.move_peak(peak.id, (peak.positions[0], peak.positions[1] + 0.0001))
    assert not w.fitting.current
    window._saved_key = window._analysis_key()
    window.close()
    app.processEvents()


def test_missing_source_numerical_workspace(analysis, tmp_path, monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    path = save_analysis(analysis, tmp_path / "test.peakfit")
    monkeypatch.setattr(
        project, "read_dataset", lambda _, **_kw: (_ for _ in ()).throw(OSError("missing"))
    )
    loaded = open_analysis(path)
    window = EditorWindow()
    window.install_analysis(loaded)
    assert window.workspace is None
    review = window.centralWidget()
    assert isinstance(review, NumericalReview)
    assert review.parameters.rowCount() > 0
    assert not window.fit_action.isEnabled()
    assert window.analysis_save.isEnabled()
    assert window.relink_action.isEnabled()
    window.close()
    app.processEvents()


def test_real_source_relocation_missing_changed_and_explicit_relink(tmp_path):
    directory = tmp_path / "original"
    directory.mkdir()
    source = directory / "spectrum.ft2"
    shutil.copyfile(DATA / "pseudo3d.ft2", source)
    dataset = read_dataset(source, DATA / "pseudo3d.list", DATA / "b1_offsets.txt")
    analysis = Analysis(
        dataset, dataset_header(dataset), dataset.peaks, PeakFitConfig().model_dump_json(), 0
    )
    path = save_analysis(analysis, directory / "review.peakfit")
    assert sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) < 100_000
    directory.rename(tmp_path / "relocated")
    path = tmp_path / "relocated/review.peakfit"
    loaded = open_analysis(path)
    assert loaded.dataset is not None, loaded.reason
    assert loaded.peaks == dataset.peaks
    np.testing.assert_array_equal(loaded.dataset.spectra.z_values, dataset.spectra.z_values)
    moved = tmp_path / "moved.ft2"
    (path.parent / "spectrum.ft2").rename(moved)
    assert open_analysis(path).dataset is None
    assert open_analysis(path, relink=moved).dataset is not None
    content = moved.read_bytes()
    moved.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    assert open_analysis(path, relink=moved).dataset is None


def test_repeated_fits_bounded_and_save_as(analysis, tmp_path):
    runs = tuple(
        replace(analysis.completion, result=replace(analysis.completion.result, run_id=uuid4().hex))
        for _ in range(12)
    )
    path = save_analysis(replace(analysis, runs=runs), tmp_path / "test.peakfit")
    assert len(json.loads((path / "manifest.json").read_text())["runs"]) == 10
    assert len(list((path / "runs").iterdir())) == 10
    copied = save_analysis(replace(analysis, path=path), tmp_path / "copy.peakfit")
    assert open_analysis(copied).peaks == analysis.peaks


def _rewrite_review(path, mutate):
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["runs"][0]
    review_path = path / "runs" / entry["run_id"] / "review.json"
    review = json.loads(review_path.read_text())
    mutate(review)
    review_path.write_text(json.dumps(review))
    entry["files"]["review.json"] = {
        "sha256": project._digest(review_path),
        "bytes": review_path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest))


def test_unsupported_model_retains_numerical_evidence(analysis, tmp_path, monkeypatch):
    path = save_analysis(analysis, tmp_path / "unknown-model.peakfit")

    def change_model(review):
        review["models"][0]["peaks"][0]["shapes"][0]["name"] = "future-model"

    _rewrite_review(path, change_model)
    monkeypatch.setattr(project, "read_dataset", lambda _, **_kw: analysis.dataset)
    loaded = open_analysis(path)
    assert loaded.dataset is None
    assert "Unsupported saved model descriptor" in loaded.reason
    assert loaded.completion.result.outcome.clusters[0].analytical_evaluation is not None


def test_inconsistent_working_transition_and_sample_mapping_rejected(analysis, tmp_path):
    path = save_analysis(analysis, tmp_path / "invalid-transition.peakfit")

    def change_position(review):
        review["post_fit_peaks"][0]["positions"][1] += 0.01

    _rewrite_review(path, change_position)
    with pytest.raises(ValueError, match="working transition"):
        open_analysis(path)


def test_changed_same_size_same_timestamp_does_not_verify(tmp_path):
    source = tmp_path / "source.ft2"
    shutil.copyfile(DATA / "pseudo3d.ft2", source)
    dataset = read_dataset(source)
    analysis = Analysis(dataset, dataset_header(dataset), (), PeakFitConfig().model_dump_json(), 0)
    path = save_analysis(analysis, tmp_path / "test.peakfit")

    before = source.stat()
    with source.open("r+b") as stream:
        stream.seek(-4, 2)
        stream.write(np.array([12345.0], dtype="float32").tobytes())
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert source.stat().st_size == before.st_size
    assert source.stat().st_mtime_ns == before.st_mtime_ns
    assert open_analysis(path).dataset is None


def test_verification_cancel_and_unreadable_sources_keep_numerical(analysis, tmp_path, monkeypatch):
    path = save_analysis(analysis, tmp_path / "test.peakfit")
    loaded = open_analysis(path, canceled=lambda: True)
    assert loaded.dataset is None
    assert "canceled" in loaded.reason
    assert loaded.completion is not None

    def unreadable(*_args, **_kwargs):
        raise PermissionError("Permission denied")

    monkeypatch.setattr(project, "read_dataset", unreadable)
    loaded = open_analysis(path)
    assert loaded.dataset is None
    assert "Permission denied" in loaded.reason


def test_protected_history_and_summary_only_journal(analysis, tmp_path):
    runs = tuple(
        replace(analysis.completion, result=replace(analysis.completion.result, run_id=uuid4().hex))
        for _ in range(12)
    )
    path = save_analysis(
        replace(
            analysis,
            runs=runs,
            journal=tuple(journal_entry(c) for c in runs),
            protected=frozenset({runs[0].result.run_id}),
        ),
        tmp_path / "history.peakfit",
    )
    manifest = json.loads((path / "manifest.json").read_text())
    assert len(manifest["runs"]) == 11
    assert manifest["journal"][0]["evidence"] == "Full"
    assert manifest["journal"][0]["parameters"] is None
    assert manifest["journal"][1]["evidence"] == "Summary only"
    assert manifest["journal"][1]["parameters"]
    assert manifest["journal"][-1]["parameters"] is None


def test_source_mutation_during_read_cannot_publish_identity(tmp_path, evidence, monkeypatch):
    source = tmp_path / "changing.ft2"
    source.write_bytes(b"first content")

    def load(*_args):
        source.write_bytes(b"other content")
        return evidence[0]

    monkeypatch.setattr(spectrum, "load_spectrum", load)
    with pytest.raises(ValueError, match="changed while loading"):
        spectrum.read_dataset(source)


def test_new_source_model_and_amplitude_solves_are_never_used_on_open(
    analysis, tmp_path, monkeypatch
):
    path = save_analysis(analysis, tmp_path / "test.peakfit")
    monkeypatch.setattr(project, "read_dataset", lambda _, **_kw: analysis.dataset)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Reopen must not prepare defaults, segment data, optimize or solve amplitudes")

    monkeypatch.setattr(pipeline, "run_pipeline_iter", forbidden)
    monkeypatch.setattr(pipeline, "prepare_cluster_parameters", forbidden)
    monkeypatch.setattr(fitting, "prepare_dataset_fit", forbidden)
    loaded = open_analysis(path)
    assert loaded.dataset is not None
    verify_model(loaded.completion.result)


def test_disposable_project_process_uses_only_trusted_ipc(analysis, tmp_path):
    path = tmp_path / "process.peakfit"
    options = {"progress": lambda _: None, "canceled": lambda: False}
    saved = project_operation("save", (analysis, path), **options)
    assert saved == path
    loaded = project_operation("open", path, **options)
    assert loaded.dataset is None  # The synthetic source deliberately has no NMRPipe data.
    assert loaded.peaks == analysis.peaks
    assert loaded.current
    verify_model(loaded.completion.result)


def test_source_unavailable_is_not_currentness_and_settings_outdate(analysis, tmp_path):
    path = save_analysis(analysis, tmp_path / "test.peakfit")
    loaded = open_analysis(path, canceled=lambda: True)
    assert loaded.dataset is None
    assert loaded.current
    changed = PeakFitConfig.model_validate_json(analysis.configuration)
    changed.noise_level = 0.25
    save_analysis(replace(analysis, configuration=changed.model_dump_json(), path=path), path)
    loaded = open_analysis(path, canceled=lambda: True)
    assert not loaded.current


def test_interrupted_first_save_never_publishes_and_dead_pid_lock_recovers(
    analysis, tmp_path, monkeypatch
):
    path = tmp_path / "new.peakfit"

    def interrupted(*_args):
        raise OSError("interrupted payload write")

    original = project._write_run
    monkeypatch.setattr(project, "_write_run", interrupted)
    with pytest.raises(OSError, match="interrupted payload"):
        save_analysis(analysis, path)
    assert not path.exists()
    monkeypatch.setattr(project, "_write_run", original)
    lock = path.with_name("." + path.name + ".save-lock")
    lock.mkdir()
    (lock / "pid").write_text("123456789")
    if os.name == "nt":
        with pytest.raises(OSError, match="Save lock exists"):
            save_analysis(analysis, path)
        shutil.rmtree(lock)
    save_analysis(analysis, path)
    assert open_analysis(path, canceled=lambda: True).completion is not None
    assert not lock.exists()
