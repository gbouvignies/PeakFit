"""Binary authority, untrusted array limits, and cross-interface scientific parity."""

import json
from dataclasses import replace
from zipfile import ZipFile

import numpy as np
import pytest
from test_gui_dataset import _dataset

from peakfit.fit import fitting, storage
from peakfit.fit.output_metadata import RunMetadata
from peakfit.fit.pipeline import run_pipeline
from peakfit.fit.run_models import FitRun
from peakfit.gui.dataset_fit import DatasetFitAttempt, finish_dataset_fit
from peakfit.gui.fit_export import export_fit_result


def completed_result(formats, save_simulated):
    spectra, _peaks, config, prepared = _dataset()
    config.output.formats = formats
    config.output.save_simulated = save_simulated
    prepared = replace(
        prepared, snapshot=replace(prepared.snapshot, configuration=config.model_dump_json())
    )
    pipeline = run_pipeline(
        config,
        prepared.clusters,
        prepared.noise,
        prepared.parameters,
        [p for c in prepared.clusters for p in c.peaks],
    )
    gui = finish_dataset_fit(prepared, DatasetFitAttempt(pipeline.completed_fit, (), 1.0))
    return spectra, config, pipeline, gui


@pytest.mark.parametrize("formats", [[], ["csv"], ["json", "csv", "txt"]])
def test_cli_gui_export_scientific_parity(tmp_path, monkeypatch, formats):
    spectra, config, pipeline, gui = completed_result(formats, True)
    metadata = RunMetadata(
        run_id=gui.run_id,
        timestamp=gui.completed_at,
        software_version=gui.software_version,
        configuration=json.loads(gui.inputs.snapshot.configuration),
        run_duration_seconds=gui.execution_seconds,
    )
    monkeypatch.setattr(fitting, "capture_output_metadata", lambda *_args: metadata)
    cli_dir = tmp_path / "cli"
    run = FitRun(
        gui.outcome,
        pipeline.continuation_state,
        cli_dir,
        spectra,
        gui.simulation_snapshot,
        gui.completed_fit,
        tuple(
            {"engine_name": i.engine_name, "peak_id": i.working_id, "assignment": i.assignment}
            for i in gui.inputs.identities
        ),
    )
    fitting.write_fit_run_outputs(run, spectra, config, {})
    gui_dir = export_fit_result(gui, tmp_path / "gui")
    # Compare all numerical members and metadata byte-for-byte; ZIP timestamps
    # and directory structure are container details, not scientific differences.
    with ZipFile(cli_dir / "scientific.npz") as cli, ZipFile(gui_dir / "scientific.npz") as editor:
        assert cli.namelist() == editor.namelist()
        for name in cli.namelist():
            assert cli.read(name) == editor.read(name), name
    for path in gui_dir.rglob("*"):
        if not path.is_file() or path.name == "scientific.npz":
            continue
        expected = cli_dir / path.relative_to(gui_dir)
        if path.suffix == ".md":
            assert [
                s for s in path.read_text().splitlines() if not s.startswith("- Generated:")
            ] == [s for s in expected.read_text().splitlines() if not s.startswith("- Generated:")]
        else:
            assert path.read_bytes() == expected.read_bytes()
    outcome, samples, context = storage.read_scientific(gui_dir / "scientific.npz")
    assert context["plane_values"].tolist() == [2.0, 0.0, 2.0]
    assert outcome.terminal_correction_revision == gui.outcome.terminal_correction_revision
    for old, new in zip(gui.completed_fit.samples, samples, strict=True):
        assert old.data.tobytes() == new.data.tobytes()
        assert old.neighbor_corrections.tobytes() == new.neighbor_corrections.tobytes()
        for a, b in zip(old.grid_indices, new.grid_indices, strict=True):
            assert a.dtype == b.dtype
            assert a.tobytes() == b.tobytes()


def test_nonfinite_uncertainties_and_array_bits_are_lossless(tmp_path):
    _spectra, _config, _pipeline, gui = completed_result([], False)
    c = gui.outcome.clusters[0]
    e = c.analytical_evaluation
    errors = np.array(e.amplitude_standard_errors, copy=True)
    errors[:] = np.nan
    cov = np.full_like(e.amplitude_covariance, np.nan)
    changed = replace(
        c,
        analytical_evaluation=replace(
            e, amplitude_standard_errors=errors, amplitude_covariance=cov
        ),
    )
    clusters = (changed, *gui.outcome.clusters[1:])
    outcome = replace(gui.outcome, clusters=clusters)
    path = storage.write_scientific(tmp_path / "science.npz", outcome, gui.completed_fit.samples)
    restored, samples, _context = storage.read_scientific(path)
    assert (
        restored.clusters[0].analytical_evaluation.amplitude_standard_errors.tobytes()
        == errors.tobytes()
    )
    assert (
        restored.clusters[0].analytical_evaluation.amplitude_covariance.tobytes() == cov.tobytes()
    )
    assert all(not s.data.flags.writeable for s in samples)


@pytest.mark.parametrize(
    "corruption", ["extra", "shape", "dtype", "identity", "residual", "revision"]
)
def test_malformed_scientific_evidence_rejected(tmp_path, corruption):
    _spectra, _config, _pipeline, gui = completed_result([], False)
    path = storage.write_scientific(tmp_path / "source.npz", gui.outcome, gui.completed_fit.samples)
    with ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    document = json.loads(members["metadata.json"])
    c = document["outcome"]["clusters"][0]
    if corruption == "extra":
        members["unexpected.npy"] = b"unreferenced"
    elif corruption in {"shape", "dtype"}:
        descriptor = c["analytical_evaluation"]["shapes"]
        descriptor[corruption] = [999999999999] if corruption == "shape" else "O"
    elif corruption == "identity":
        document["samples"][0]["cluster_id"] = -500
    elif corruption == "revision":
        c["correction_revision"] += 1
    else:
        # Replace a residual with a different, structurally valid array member.
        c["analytical_evaluation"]["raw_residuals"], c["analytical_evaluation"]["model_values"] = (
            c["analytical_evaluation"]["model_values"],
            c["analytical_evaluation"]["raw_residuals"],
        )
    members["metadata.json"] = json.dumps(document).encode()
    broken = tmp_path / "broken.npz"
    with ZipFile(broken, "w") as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    with pytest.raises((ValueError, AssertionError)):
        storage.read_scientific(broken)


def test_reader_limits_before_allocation(tmp_path, monkeypatch):
    _spectra, _config, _pipeline, gui = completed_result([], False)
    path = storage.write_scientific(
        tmp_path / "science.npz", gui.outcome, gui.completed_fit.samples
    )
    monkeypatch.setattr(storage, "MAX_METADATA_BYTES", 10)
    with pytest.raises(ValueError, match="metadata exceeds"):
        storage.read_scientific(path)
    monkeypatch.setattr(storage, "MAX_METADATA_BYTES", 16 * 1024**2)
    monkeypatch.setattr(storage, "MAX_ARRAY_BYTES", 10)
    with pytest.raises(ValueError, match="Expanded scientific arrays"):
        storage.read_scientific(path)


def test_repeated_plane_values_retain_source_columns_after_exclusion(tmp_path):
    from peakfit.gui.fitting import prepare_dataset_fit  # noqa: PLC0415

    spectra, peaks, config, _prepared = _dataset()
    spectra.exclude_planes([1])
    assert spectra.source_plane_indices == (0, 2)
    prepared = prepare_dataset_fit(spectra, peaks, config)
    pipeline = run_pipeline(
        config,
        prepared.clusters,
        prepared.noise,
        prepared.parameters,
        [p for c in prepared.clusters for p in c.peaks],
    )
    gui = finish_dataset_fit(prepared, DatasetFitAttempt(pipeline.completed_fit, (), 1.0))
    path = export_fit_result(gui, tmp_path / "selected")
    _, samples, context = storage.read_scientific(path / "scientific.npz")
    assert context["plane_values"].tolist() == [2.0, 2.0]
    assert context["source_plane_indices"].tolist() == [0, 2]
    assert all(s.data.shape[1] == 2 for s in samples)
    spectra.exclude_planes([0])
    assert spectra.source_plane_indices == (2,)


def test_arbitrary_provenance_tags_and_endian_array_bits(tmp_path):
    _spectra, _config, _pipeline, gui = completed_result([], False)
    c = gui.outcome.clusters[0]
    payload = np.array([0x7FF8000000000001, 0x8000000000000000], dtype=">u8").view(">f8")
    metadata = {
        "array": "ordinary metadata",
        "float": "ordinary metadata",
        "mapping": {"mapping": "also metadata"},
        "payload": payload,
        "scalar_nan": float(payload[0]),
    }
    changed = replace(c, optimizer_provenance=replace(c.optimizer_provenance, metadata=metadata))
    outcome = replace(gui.outcome, clusters=(changed, *gui.outcome.clusters[1:]))
    path = storage.write_scientific(tmp_path / "science.npz", outcome, gui.completed_fit.samples)
    restored, _samples, _context = storage.read_scientific(path)
    actual = restored.clusters[0].optimizer_provenance.metadata
    assert actual["array"] == metadata["array"]
    assert actual["float"] == metadata["float"]
    assert actual["mapping"] == metadata["mapping"]
    assert actual["payload"].dtype == payload.dtype
    assert actual["payload"].tobytes() == payload.tobytes()
    assert np.float64(actual["scalar_nan"]).tobytes() == np.float64(float(payload[0])).tobytes()


def test_every_authoritative_array_round_trips_exactly(tmp_path):
    from collections.abc import Mapping  # noqa: PLC0415
    from dataclasses import fields, is_dataclass  # noqa: PLC0415

    _spectra, _config, _pipeline, gui = completed_result([], False)
    path = storage.write_scientific(
        tmp_path / "science.npz", gui.outcome, gui.completed_fit.samples
    )
    outcome, samples, _context = storage.read_scientific(path)

    def compare(old, new):
        if isinstance(old, np.ndarray):
            assert old.dtype == new.dtype
            assert old.shape == new.shape
            assert old.tobytes() == new.tobytes()
            assert not new.flags.writeable
        elif is_dataclass(old):
            for field in fields(old):
                compare(getattr(old, field.name), getattr(new, field.name))
        elif isinstance(old, Mapping):
            assert old.keys() == new.keys()
            for key in old:
                compare(old[key], new[key])
        elif isinstance(old, tuple):
            assert isinstance(new, tuple)
            assert len(old) == len(new)
            for a, b in zip(old, new, strict=True):
                compare(a, b)
        elif isinstance(old, float):
            assert np.float64(old).tobytes() == np.float64(new).tobytes()
        else:
            assert old == new

    compare(gui.outcome, outcome)
    compare(gui.completed_fit.samples, samples)
