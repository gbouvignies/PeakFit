"""Lossless evidence, linked source identity and transactional analysis saves."""

import hashlib
import json
import os
import shutil
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import Event
from uuid import uuid4

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets
from test_gui_dataset import _dataset
from test_gui_editor import wait_for

from peakfit.engine.domain.config import PeakFitConfig
from peakfit.fit import pipeline
from peakfit.gui import fitting, project, spectrum
from peakfit.gui.background import ViewerComputation
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.editor.numerical import NumericalReview
from peakfit.gui.editor.window import EditorWindow
from peakfit.gui.fit_positions import dataset_completion
from peakfit.gui.fit_process import FitProcessResult
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.project import Analysis, dataset_header, open_analysis, save_analysis
from peakfit.gui.project_process import project_operation
from peakfit.gui.project_results import journal_entry, reconstruct_result, verify_model
from peakfit.gui.session_evidence import SessionEvidence, interrupted_sessions
from peakfit.gui.spectrum import Dataset, read_dataset
from peakfit.io.scientific import read_scientific

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
    stat = source.stat()
    record = json.dumps(
        dict(
            json.loads(record),
            size=str(stat.st_size),
            mtime_ns=str(stat.st_mtime_ns),
            ctime_ns=str(stat.st_ctime_ns),
            inode=str(stat.st_ino),
            device=str(stat.st_dev),
        )
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
    assert loaded.current
    assert not (path / "spectra").exists()
    assert loaded.peaks == analysis.peaks
    old, new = analysis.completion.result, loaded.completion.result
    assert new.inputs.snapshot.peaks == old.inputs.snapshot.peaks
    assert new.inputs.identities == old.inputs.identities
    assert [
        (p.name, p.value, p.min, p.max, p.vary, p.unit) for p in new.inputs.parameters.values()
    ] == [(p.name, p.value, p.min, p.max, p.vary, p.unit) for p in old.inputs.parameters.values()]
    for a, b in zip(old.completed_fit.samples, new.completed_fit.samples, strict=True):
        assert not hasattr(b, "data")
        np.testing.assert_array_equal(a.neighbor_corrections, b.neighbor_corrections)
    for a, b in zip(old.outcome.clusters, new.outcome.clusters, strict=True):
        assert a.final_nonlinear_parameters == b.final_nonlinear_parameters
        np.testing.assert_array_equal(
            a.analytical_evaluation.amplitudes, b.analytical_evaluation.amplitudes
        )
        np.testing.assert_array_equal(
            a.analytical_evaluation.amplitude_covariance,
            b.analytical_evaluation.amplitude_covariance,
        )
        assert b.sample_evaluation is None
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
    payload = next(path.rglob("result.npz"))
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


def test_canceled_verification_numerical_workspace(analysis, tmp_path, monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    path = save_analysis(analysis, tmp_path / "test.peakfit")
    monkeypatch.setattr(
        project, "read_dataset", lambda _, **_kw: (_ for _ in ()).throw(OSError("missing"))
    )
    loaded = open_analysis(path, canceled=lambda: True)
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
    assert (
        sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) < dataset.spectra.data.nbytes
    )
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
    assert open_analysis(path, relink=source).dataset is None


def test_verification_cancel_and_unreadable_sources_keep_numerical(analysis, tmp_path, monkeypatch):
    path = save_analysis(analysis, tmp_path / "test.peakfit")
    loaded = open_analysis(path, canceled=lambda: True)
    assert loaded.dataset is None
    assert "canceled" in loaded.reason
    assert loaded.completion is not None

    def unreadable(*_args, **_kwargs):
        raise PermissionError("Permission denied")

    monkeypatch.setattr(project, "read_dataset", unreadable)
    loaded = open_analysis(path, relink=analysis.dataset.path)
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
    assert loaded.dataset is None
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


def test_save_ten_reopen_fit_save_reopen_preserves_evicted_scientific_summary(
    analysis, tmp_path, monkeypatch
):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    runs = tuple(
        replace(analysis.completion, result=replace(analysis.completion.result, run_id=uuid4().hex))
        for _ in range(10)
    )
    expected = json.loads(json.dumps(journal_entry(runs[0])))
    path = save_analysis(replace(analysis, runs=runs), tmp_path / "History.peakfit")
    assert all(
        e["parameters"] is None for e in json.loads((path / "manifest.json").read_text())["journal"]
    )
    monkeypatch.setattr(project, "read_dataset", lambda _, **_kw: analysis.dataset)
    loaded = open_analysis(path)
    assert len(loaded.runs) == 10
    assert all(isinstance(r, project.StoredRun) for r in loaded.runs)
    window = EditorWindow()
    window.install_analysis(loaded)
    del loaded, runs
    w = window.workspace
    assert w is not None
    w.fitting.fit()  # The first Fit of a session reveals the setup instead of running.
    assert (w.fitting.setup_seen, w.fitting.busy, w.selected_id) == (True, False, None)
    w.fitting.fit()  # Fresh preparation and a real optimizer invocation after reopening.
    wait_for(lambda: not w.fitting.busy, timeout=20)
    assert w.fitting.current, w.fitting.failure
    assert all(isinstance(r, project.StoredRun) for r in w.fitting.history)
    recovery = w.fitting._recovery_path
    assert recovery is not None
    recoverable = open_analysis(recovery, canceled=lambda: True)
    assert w.fitting.result is not None
    assert recoverable.completion.result.run_id == w.fitting.result.run_id
    del recoverable
    assert window.save_analysis_dialog()
    wait_for(lambda: not window._saving, timeout=20)
    wait_for(lambda: not recovery.exists(), timeout=10)
    window.close()
    app.processEvents()
    reopened = open_analysis(path)
    evicted = next(e for e in reopened.journal if e["run_id"] == expected["run_id"])
    assert evicted["evidence"] == "Summary only"
    for field, value in expected.items():
        assert evicted[field] == value, field
    assert evicted["parameters"]
    assert len(reopened.runs) == 10
    review = NumericalReview(replace(reopened, dataset=None))
    index = next(
        i
        for i in range(review.run.count())
        if isinstance(review.run.itemData(i), dict)
        and review.run.itemData(i)["run_id"] == expected["run_id"]
    )
    review.run.setCurrentIndex(index)
    assert review.parameters.rowCount() == len(expected["parameters"])
    assert review.initial.rowCount() == len(expected["initial_peaks"])
    assert review.starting_parameters.rowCount() == len(expected["initial_parameters"])
    assert not review.export.isEnabled()
    # A second metadata-only save/reopen must preserve the already demoted summary too.
    save_analysis(reopened, path)
    again = open_analysis(path, canceled=lambda: True)
    assert next(e for e in again.journal if e["run_id"] == expected["run_id"]) == evicted
    review.close()
    review.deleteLater()


def test_open_and_history_list_materialize_only_active_then_selected_run(
    analysis, tmp_path, monkeypatch
):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    runs = tuple(
        replace(analysis.completion, result=replace(analysis.completion.result, run_id=uuid4().hex))
        for _ in range(10)
    )
    path = save_analysis(replace(analysis, runs=runs), tmp_path / "Lazy.peakfit")
    original = project._load_run
    materialized = []

    def counted(root, entry):
        materialized.append(entry["run_id"])
        return original(root, entry)

    monkeypatch.setattr(project, "_load_run", counted)
    loaded = open_analysis(path, canceled=lambda: True)
    assert materialized == [runs[-1].result.run_id]
    review = NumericalReview(loaded)
    assert review._loaded_completion is None
    assert materialized == [runs[-1].result.run_id]
    review.run.setCurrentIndex(1)
    wait_for(lambda: review._loaded_completion is not None, timeout=10)
    assert review.selected_run().result.run_id == runs[-2].result.run_id
    review.run.setCurrentIndex(0)
    assert review._loaded_completion is None
    assert review.selected_run() is loaded.completion
    review.close()
    review.deleteLater()
    app.processEvents()


@pytest.mark.skipif(
    os.name != "posix", reason="Safe automatic recovery ownership uses POSIX PID checks"
)
def test_session_evidence_cleanup_and_explicit_recovery(analysis, tmp_path, monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    owner = SessionEvidence()
    saved = owner.preserve(analysis)
    recovery = owner.path
    assert recovery is not None
    root = recovery.parent
    assert open_analysis(recovery, canceled=lambda: True).peaks == analysis.peaks
    assert all(isinstance(r, project.StoredRun) for r in saved.runs)
    # A crash bypasses normal cleanup. Only the owner PID marker changes in this simulation.
    (root / "pid").write_text("123456789")
    assert recovery in interrupted_sessions()
    monkeypatch.setattr(project, "read_dataset", lambda _, **_kw: analysis.dataset)
    window = EditorWindow()
    window._recovering_path = recovery
    window.install_analysis(open_analysis(recovery))
    assert window.analysis_path is None  # Recovery requires an explicit permanent Save as.
    assert window.isWindowModified()
    destination = tmp_path / "Recovered.peakfit"
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", lambda *_a, **_k: (str(destination), "")
    )
    assert window.save_analysis_dialog()
    wait_for(lambda: not window._saving, timeout=15)
    wait_for(lambda: not root.exists())
    window.close()
    app.processEvents()
    assert open_analysis(destination, canceled=lambda: True).peaks == analysis.peaks
    owner.root = None  # Ownership transferred to the recovered workspace and cleaned there.


def test_typed_scientific_archive_has_no_json_arrays(analysis, tmp_path):
    from zipfile import ZipFile  # noqa: PLC0415

    path = save_analysis(analysis, tmp_path / "Typed.peakfit")
    archive_path = next(path.rglob("result.npz"))
    with ZipFile(archive_path) as archive:
        document = json.loads(archive.read("metadata.json"))
    assert document["outcome"]["clusters"][0]["analytical_evaluation"]["shapes"]["array"]
    assert not list(path.rglob("fit.json*"))
    outcome, _samples, _context = read_scientific(archive_path)
    assert (
        outcome.final_nonlinear_parameters
        == analysis.completion.result.outcome.final_nonlinear_parameters
    )


def test_repeated_fits_beyond_history_limit_have_one_resident_result_and_cleanup_on_close(
    analysis, tmp_path, monkeypatch
):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = EditorWindow()
    window.install_dataset(analysis.dataset)
    w = window.workspace
    assert w is not None
    f = w.fitting
    f.configuration = PeakFitConfig.model_validate_json(analysis.configuration)
    for index in range(12):
        prepared = prepare_dataset_fit(w.dataset.spectra, w.peaks, f.configuration)
        prepared = replace(
            prepared, snapshot=replace(prepared.snapshot, source_files=w.dataset.sources)
        )
        result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
        f._submitted_revision = w.edits.revision
        f.process.state = "completed"
        f._finished(0, FitProcessResult(result, 0, 0, 0, 0))
        del result, prepared
        wait_for(lambda: not f.busy, timeout=15)
        assert f.current, f.failure
        assert len(f.history) == min(index + 1, 10)
        assert all(isinstance(r, project.StoredRun) for r in f.history)
        assert all(not hasattr(r, "result") for r in f.history)
        wait_for(lambda: len(list((f._session.path / "runs").iterdir())) == len(f.history))
    recovery = f._recovery_path
    assert recovery is not None
    root = recovery.parent
    snapshot = tmp_path / "Preserved.peakfit"
    shutil.copytree(recovery, snapshot)
    assert f.result is not None
    last_id = f.result.run_id
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda _dialog: 0)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "clickedButton",
        lambda dialog: next(b for b in dialog.buttons() if b.text() == "Discard changes"),
    )
    window.close()
    wait_for(lambda: not root.exists())
    app.processEvents()
    reopened = open_analysis(snapshot, canceled=lambda: True)
    assert reopened.completion.result.run_id == last_id
    assert len(reopened.runs) == 10
    assert all(isinstance(r, project.StoredRun) for r in reopened.runs)
    assert len([e for e in reopened.journal if e["evidence"] == "Summary only"]) == 2
    assert all(e["parameters"] for e in reopened.journal if e["evidence"] == "Summary only")


def test_numerical_capability_transition_transfers_temporary_evidence_owner(analysis):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    owner = SessionEvidence()
    saved = owner.preserve(analysis)
    recovery = owner.path
    assert recovery is not None
    root = recovery.parent
    window = EditorWindow()
    window.install_dataset(analysis.dataset)
    w = window.workspace
    assert w is not None
    w.fitting._session = owner
    window.install_analysis(replace(saved, dataset=None, reason="Source unavailable"))
    assert w.fitting._session is not owner
    assert window._recovered_session is owner
    app.processEvents()
    assert recovery.exists()  # Stopping spectral widgets must not delete numerical history.
    loaded = open_analysis(recovery, canceled=lambda: True)
    assert loaded.completion.result.run_id == analysis.completion.result.run_id
    window.install_dataset(
        analysis.dataset
    )  # Replacing the numerical workspace also ends ownership.
    wait_for(lambda: not root.exists())
    window.close()


def test_budget_eviction_materializes_summary_before_removing_payload(
    analysis, tmp_path, monkeypatch
):
    runs = tuple(
        replace(analysis.completion, result=replace(analysis.completion.result, run_id=uuid4().hex))
        for _ in range(3)
    )
    expected = json.loads(json.dumps(journal_entry(runs[0])))
    path = save_analysis(replace(analysis, runs=runs), tmp_path / "Budget.peakfit")
    loaded = open_analysis(path, canceled=lambda: True)
    monkeypatch.setattr(project, "MAX_EVIDENCE_BYTES", 1)
    save_analysis(loaded, path)
    again = open_analysis(path, canceled=lambda: True)
    assert len(again.runs) == 1
    evicted = next(e for e in again.journal if e["run_id"] == expected["run_id"])
    assert evicted["parameters"] == expected["parameters"]
    assert evicted["initial_parameters"] == expected["initial_parameters"]
    assert evicted["diagnostics"] == expected["diagnostics"]
    assert not (path / "runs" / runs[0].result.run_id).exists()


def test_normal_close_waits_for_inflight_preservation_before_cleanup(analysis, tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    owner = SessionEvidence()
    owner.preserve(analysis)
    root = owner.root
    assert root is not None
    started, release = Event(), Event()
    parent = QtCore.QObject()
    worker = ViewerComputation(parent)
    second = SessionEvidence()
    second.root = tmp_path / "second-owned-store"
    second.root.mkdir()
    second_root = second.root

    def preserving():
        started.set()
        assert release.wait(10)
        owner.preserve(analysis)

    worker.submit("preserve", 0, preserving)
    assert started.wait(5)
    try:
        worker.cleanup_after_work(second.close)
        worker.close(finalize=owner.close)
        assert second_root.exists()
        assert root.exists()  # Cleanup must queue behind the writer, not race it.
    finally:
        release.set()
    wait_for(lambda: not root.exists(), timeout=10)
    assert not second_root.exists()
    app.processEvents()


def test_checkpoint_before_ui_adoption_does_not_delete_live_history_references(analysis):
    owner = SessionEvidence()
    runs = tuple(
        replace(analysis.completion, result=replace(analysis.completion.result, run_id=uuid4().hex))
        for _ in range(10)
    )
    live = owner.preserve(replace(analysis, runs=runs))
    oldest = live.runs[0]
    next_run = replace(
        analysis.completion, result=replace(analysis.completion.result, run_id=uuid4().hex)
    )
    try:
        checkpoint = owner.preserve(
            replace(
                live,
                runs=(*live.runs, next_run),
                active=next_run,
                journal=(*live.journal, journal_entry(next_run)),
            )
        )
        # Simulate cancellation after disk publication but before Qt accepts the new run.
        assert len(checkpoint.runs) == 10
        assert oldest.run_id not in {r.run_id for r in checkpoint.runs}
        assert oldest.load().result.run_id == oldest.run_id
        # Once Qt has adopted that checkpoint, the next generation can retire its old lease.
        newest = replace(next_run, result=replace(next_run.result, run_id=uuid4().hex))
        owner.preserve(
            replace(
                checkpoint,
                runs=(*checkpoint.runs, newest),
                active=newest,
                journal=(*checkpoint.journal, journal_entry(newest)),
            )
        )
        assert not (oldest.root / "runs" / oldest.run_id).exists()
    finally:
        owner.close()


def test_two_linked_analyses_share_source_and_relink_is_persisted(tmp_path):
    source = tmp_path / "spectrum.ft2"
    shutil.copyfile(DATA / "pseudo3d.ft2", source)
    dataset = read_dataset(source, DATA / "pseudo3d.list", DATA / "b1_offsets.txt")
    analysis = Analysis(
        dataset, dataset_header(dataset), dataset.peaks, PeakFitConfig().model_dump_json(), 0
    )
    first = save_analysis(analysis, tmp_path / "First.peakfit")
    second = save_analysis(analysis, tmp_path / "Second.peakfit")
    for path in (first, second):
        assert not (path / "spectra").exists()
        opened = open_analysis(path)
        assert opened.dataset is not None, opened.reason
        assert (
            opened.header["source"]["checksum_sha256"]
            == analysis.header["source"]["checksum_sha256"]
        )
    moved = tmp_path / "moved.ft2"
    source.rename(moved)
    assert open_analysis(first).dataset is None
    linked = open_analysis(first, relink=moved)
    assert linked.dataset is not None, linked.reason
    save_analysis(linked, first)
    assert open_analysis(first).dataset is not None
    assert open_analysis(second).dataset is None  # No implicit filesystem search/relink.
    moved.write_bytes(b"not a spectrum")
    invalid = open_analysis(first)
    assert invalid.dataset is None
    assert invalid.reason


def test_linked_verify_source_plane_identity(analysis, tmp_path, monkeypatch):
    spectra = analysis.dataset.spectra.model_copy(deep=True)
    spectra.exclude_planes([1])  # Repeated experimental values cannot identify these planes.
    config = PeakFitConfig.model_validate_json(analysis.configuration)
    prepared = prepare_dataset_fit(spectra, analysis.peaks, config)
    prepared = replace(
        prepared, snapshot=replace(prepared.snapshot, source_files=analysis.dataset.sources)
    )
    result = finish_dataset_fit(prepared, compute_dataset_fit(prepared))
    completion = dataset_completion(
        result, prepared.snapshot, submitted_revision=0, manual_revision=0
    )
    path = save_analysis(
        replace(analysis, runs=(completion,), peaks=completion.working_peaks),
        tmp_path / "Mapped.peakfit",
    )
    monkeypatch.setattr(project, "read_dataset", lambda _, **_kw: analysis.dataset)
    loaded = open_analysis(path)
    assert loaded.dataset is not None, loaded.reason
    assert loaded.completion.result.inputs.snapshot.source_plane_indices == (0, 2)
    for old, new in zip(
        result.completed_fit.samples,
        reconstruct_result(loaded.completion.result, loaded.dataset).completed_fit.samples,
        strict=True,
    ):
        np.testing.assert_array_equal(old.data, new.data)
        np.testing.assert_array_equal(old.neighbor_corrections, new.neighbor_corrections)


def test_project_has_no_observations_predictions_or_residual_arrays(analysis, tmp_path):
    from zipfile import ZipFile  # noqa: PLC0415

    path = save_analysis(analysis, tmp_path / "External.peakfit")
    assert {p.name for p in path.iterdir()} == {"manifest.json", "runs"}
    with ZipFile(next(path.rglob("result.npz"))) as archive:
        assert all(
            not name.endswith(
                (
                    "/data.npy",
                    "/model_values.npy",
                    "/raw_residuals.npy",
                    "/normalized_residuals.npy",
                )
            )
            for name in archive.namelist()
        )
        assert json.loads(archive.read("metadata.json"))["evidence"] == "results"
    loaded = open_analysis(path, canceled=lambda: True)
    copied = save_analysis(loaded, tmp_path / "Copied.peakfit")
    assert (
        next(path.rglob("result.npz")).read_bytes() == next(copied.rglob("result.npz")).read_bytes()
    )


def test_reconstruction_exact_parent_inputs_no_solve_and_reference_failure(
    analysis, tmp_path, monkeypatch
):
    path = save_analysis(analysis, tmp_path / "External.peakfit")
    monkeypatch.setattr(project, "read_dataset", lambda _, **_kw: analysis.dataset)
    loaded = open_analysis(path)
    assert loaded.dataset is not None, loaded.reason

    def forbidden(*_args, **_kw):
        pytest.fail("Reconstruction must not refit or resolve amplitudes")

    monkeypatch.setattr(pipeline, "run_pipeline_iter", forbidden)
    from peakfit.engine.algorithms import evaluation  # noqa: PLC0415
    from peakfit.io.readers.continuation import continuation_state  # noqa: PLC0415

    monkeypatch.setattr(evaluation, "evaluate_analytical_model", forbidden)
    restored = reconstruct_result(loaded.completion.result, loaded.dataset)
    for before, after in zip(
        analysis.completion.result.completed_fit.samples,
        restored.completed_fit.samples,
        strict=True,
    ):
        assert before.data.tobytes() == after.data.tobytes()
        assert before.neighbor_corrections.tobytes() == after.neighbor_corrections.tobytes()
    for before, after in zip(
        analysis.completion.result.outcome.clusters, restored.outcome.clusters, strict=True
    ):
        if before.sample_evaluation:
            np.testing.assert_allclose(
                before.sample_evaluation.model_values,
                after.sample_evaluation.model_values,
                rtol=1e-10,
                atol=1e-9,
            )
            np.testing.assert_allclose(
                before.sample_evaluation.raw_residuals,
                after.sample_evaluation.raw_residuals,
                rtol=1e-10,
                atol=1e-9,
            )
    state = continuation_state(restored.completed_fit)
    for cluster, sample in zip(state.clusters, restored.completed_fit.samples, strict=True):
        np.testing.assert_array_equal(
            cluster.corrected_data, sample.data - sample.neighbor_corrections
        )
    broken = loaded.dataset.spectra.model_copy(deep=True)
    broken.data.flat[0] += 123
    # Modify an actual fitted observation, not a pixel outside the masks.
    sample = restored.completed_fit.samples[0]
    broken.data[(0, *[i[0] for i in sample.grid_indices])] += 123
    with pytest.raises(ValueError, match="recorded identity"):
        reconstruct_result(loaded.completion.result, replace(loaded.dataset, spectra=broken))


def test_missing_source_disables_samples_preserves_numerical_inspection(analysis, tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    path = save_analysis(analysis, tmp_path / "External.peakfit")
    missing = open_analysis(path, canceled=lambda: True)
    review = NumericalReview(missing)
    assert review.parameters.rowCount() > 0
    assert review.peak.count() > 0
    assert not review.plane.isEnabled()
    assert not review.cluster.isEnabled()
    assert review.samples.rowCount() == 0
    assert review.export.isEnabled()
    review.close()
    app.processEvents()


def test_verified_numerical_review_materializes_only_requested_samples(
    analysis, tmp_path, monkeypatch
):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    path = save_analysis(analysis, tmp_path / "External.peakfit")
    monkeypatch.setattr(project, "read_dataset", lambda _, **_kw: analysis.dataset)
    loaded = open_analysis(path)
    assert all(c.sample_evaluation is None for c in loaded.completion.result.outcome.clusters)
    review = NumericalReview(loaded)
    wait_for(lambda: review._loaded_completion is not None)
    assert review.parameters.rowCount() > 0
    assert review.samples.rowCount() > 0
    assert review.plane.isEnabled()
    assert all(c.sample_evaluation is None for c in loaded.completion.result.outcome.clusters)
    review.close()
    app.processEvents()


def test_reconstruction_checks_evaluator_tolerance_and_rejects_drift(
    analysis, tmp_path, monkeypatch
):
    from peakfit.engine.domain.cluster import Cluster  # noqa: PLC0415

    path = save_analysis(analysis, tmp_path / "External.peakfit")
    loaded = open_analysis(path, canceled=lambda: True)
    original = Cluster.evaluate_peaks
    monkeypatch.setattr(
        Cluster, "evaluate_peaks", lambda *a, **kw: original(*a, **kw) * (1 + 1e-12)
    )
    reconstruct_result(loaded.completion.result, analysis.dataset)
    monkeypatch.setattr(Cluster, "evaluate_peaks", lambda *a, **kw: original(*a, **kw) * (1 + 1e-4))
    with pytest.raises(AssertionError, match="reference"):
        reconstruct_result(loaded.completion.result, analysis.dataset)


@pytest.mark.parametrize("detailed", [False, True])
def test_reopened_project_exports_preserve_completed_science(analysis, tmp_path, detailed):
    from zipfile import ZipFile  # noqa: PLC0415

    from peakfit.gui.fit_export import export_fit_result  # noqa: PLC0415

    path = save_analysis(analysis, tmp_path / "External.peakfit")
    loaded = open_analysis(path, canceled=lambda: True)
    before = export_fit_result(
        analysis.completion.result,
        tmp_path / "Before",
        completion=analysis.completion,
        current=True,
        detailed_evidence=detailed,
    )
    after = export_fit_result(
        loaded.completion.result,
        tmp_path / "After",
        completion=loaded.completion,
        current=True,
        detailed_evidence=detailed,
        dataset=analysis.dataset,
    )
    with ZipFile(before / "scientific.npz") as a, ZipFile(after / "scientific.npz") as b:
        assert set(a.namelist()) == set(b.namelist())
        for name in a.namelist():
            if name.endswith(".npy") and name.startswith(("outcome/", "samples/")):
                np.testing.assert_allclose(
                    np.load(a.open(name)), np.load(b.open(name)), rtol=1e-10, atol=1e-9
                )
        assert json.loads(a.read("metadata.json")) == json.loads(b.read("metadata.json"))
    for name in ("parameters", "intensities", "clusters", "shifts"):
        assert (before / f"tables/{name}.csv").read_bytes() == (
            after / f"tables/{name}.csv"
        ).read_bytes()
    if detailed:
        with pytest.raises(ValueError, match="Detailed evidence requires"):
            export_fit_result(
                loaded.completion.result, tmp_path / "Unavailable", detailed_evidence=True
            )
    else:
        export_fit_result(
            loaded.completion.result, tmp_path / "NumericalOnly", detailed_evidence=False
        )
