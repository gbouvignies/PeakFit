"""The native archive reconstructs the same terminal problem through both interfaces."""

import json
from zipfile import ZipFile

import numpy as np
import pytest
from builders import completed_result, two_cluster_dataset

from peakfit.engine.algorithms.noise_correlation import not_modelled
from peakfit.engine.scientific.sampling import sampling_problem
from peakfit.gui.fit_export import export_fit_result
from peakfit.io.readers.continuation import continuation_state, load_completed, restore_completed
from peakfit.io.readers.results import ResultsLoader
from peakfit.io.scientific import read_scientific
from peakfit.shared.exceptions import ScientificIntegrityError


@pytest.mark.parametrize("detailed", [False, True])
def test_native_terminal_problem_and_parent_likelihood(tmp_path, detailed):
    spectra, _, pipeline, result = completed_result(["csv", "txt"], False, detailed)
    directory = export_fit_result(result, tmp_path / "results")
    assert not (directory / "summary/fit.json").exists()
    outcome, samples, context = read_scientific(directory / "scientific.npz")
    reconstructed = restore_completed(outcome, samples, context, source_data=spectra.data)
    state = continuation_state(reconstructed)
    for old, new in zip(pipeline.continuation_state.clusters, state.clusters, strict=True):
        assert old.cluster_id == new.cluster_id
        np.testing.assert_array_equal(old.data, new.data)
        np.testing.assert_array_equal(old.corrections, new.corrections)
        np.testing.assert_array_equal(old.corrected_data, new.corrected_data)
        assert [p.name for p in old.peaks] == [p.name for p in new.peaks]
        # The parent's sampled posterior and the reopened one are the same problem.
        parent = sampling_problem(pipeline.completed_fit, new.cluster_id)
        native = sampling_problem(reconstructed, new.cluster_id)
        assert native.names == parent.names
        np.testing.assert_array_equal(native.start, parent.start)
        expected = parent.conditional(parent.start)
        actual = native.conditional(native.start)
        assert expected is not None
        assert actual is not None
        assert np.isfinite(actual.log_likelihood)
        assert actual.log_likelihood == expected.log_likelihood
        np.testing.assert_array_equal(actual.mean, expected.mean)
    assert [(p.name, p.value, p.min, p.max, p.vary) for p in state.scalar_params.values()] == [
        (p.name, p.value, p.min, p.max, p.vary) for p in result.outcome.final_nonlinear_parameters
    ]
    if detailed:
        native = load_completed(directory / "scientific.npz")
        loaded = continuation_state(ResultsLoader(directory).load_completed())
        assert (
            native.outcome.terminal_correction_revision
            == result.outcome.terminal_correction_revision
        )
        assert loaded.noise == state.noise
        for a, b in zip(state.clusters, loaded.clusters, strict=True):
            np.testing.assert_array_equal(a.corrected_data, b.corrected_data)
    else:
        with pytest.raises(FileNotFoundError, match="verified spectrum"):
            load_completed(directory / "scientific.npz")
        with pytest.raises(ValueError, match="verified spectrum"):
            restore_completed(outcome, samples, context)


# Each earlier archive version recorded a configuration key that no longer exists, and
# which configuration validation forbids: version rejection must come first.
@pytest.mark.parametrize(
    ("version", "section", "key"),
    [("4.0.0", "parameters", "from_file"), ("5.0.0", "fitting", "optimizer_seed")],
)
def test_archive_recording_a_removed_config_key_is_rejected_by_version(
    tmp_path, version, section, key
):
    _, _, _, result = completed_result([], False, True)
    directory = export_fit_result(result, tmp_path / "results")
    archive = directory / "scientific.npz"
    state = continuation_state(ResultsLoader(directory).load_completed())
    assert state.noise is not None
    assert [p.name for p in state.scalar_params.values()] == [
        p.name for p in result.outcome.final_nonlinear_parameters
    ]

    with ZipFile(archive) as source:
        members = {name: source.read(name) for name in source.namelist()}
    document = json.loads(members["metadata.json"])
    configuration = document["context"]["mapping"]["metadata"]["mapping"]["configuration"]
    configuration["mapping"][section]["mapping"][key] = None
    document["schema_version"] = version
    members["metadata.json"] = json.dumps(document).encode()
    with ZipFile(archive, "w") as target:
        for name, payload in members.items():
            target.writestr(name, payload)
    with pytest.raises(ValueError, match="Unsupported scientific result version"):
        ResultsLoader(directory).load_completed()


def test_reconstruction_refuses_changed_evaluator_reference(tmp_path):
    spectra, _, _, result = completed_result([], False)
    directory = export_fit_result(result, tmp_path / "results")
    outcome, samples, context = read_scientific(directory / "scientific.npz")
    changed_source = spectra.data.copy()
    changed_source[(0, *(i[0] for i in samples[0].grid_indices))] += 1.0
    with pytest.raises(ValueError, match="exact recorded identity"):
        restore_completed(outcome, samples, context, source_data=changed_source)
    e = outcome.clusters[0].analytical_evaluation
    e.shapes.flags.writeable = True
    e.shapes[0, 0] += 0.1
    with pytest.raises(ScientificIntegrityError, match="evaluator no longer reproduces"):
        restore_completed(outcome, samples, context, source_data=spectra.data)


def test_compact_continuation_verifies_source_and_exact_excluded_plane_identity(
    tmp_path, monkeypatch
):
    import hashlib  # noqa: PLC0415
    import json  # noqa: PLC0415
    from dataclasses import replace  # noqa: PLC0415

    from peakfit.fit.pipeline import run_pipeline  # noqa: PLC0415
    from peakfit.gui.dataset_fit import DatasetFitAttempt, finish_dataset_fit  # noqa: PLC0415
    from peakfit.gui.fitting import prepare_dataset_fit  # noqa: PLC0415

    spectra, peaks, config, _ = two_cluster_dataset()
    original = spectra.model_copy(deep=True)
    spectra.exclude_planes([1])
    source = tmp_path / "original.ft2"
    source.write_bytes(b"test input; scientific reader supplied below")
    record = json.dumps(
        {"path": str(source), "checksum_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    )
    prepared = prepare_dataset_fit(spectra, peaks, config)
    prepared = replace(
        prepared, snapshot=replace(prepared.snapshot, source_files=(("spectrum", record),))
    )
    pipeline = run_pipeline(
        config,
        prepared.clusters,
        prepared.noise,
        prepared.parameters,
        noise_model=not_modelled("synthetic spectrum"),
    )
    result = finish_dataset_fit(prepared, DatasetFitAttempt(pipeline.completed_fit, (), 1.0))
    directory = export_fit_result(result, tmp_path / "results")
    monkeypatch.setattr("peakfit.io.readers.continuation.read_spectra", lambda *_: original)
    restored = load_completed(directory / "scientific.npz")
    for a, b in zip(restored.samples, result.completed_fit.samples, strict=True):
        np.testing.assert_array_equal(a.data, b.data)
        np.testing.assert_array_equal(a.neighbor_corrections, b.neighbor_corrections)
    _, _, context = read_scientific(directory / "scientific.npz")
    assert context["plane_values"].tolist() == [2.0, 2.0]
    assert context["source_plane_indices"].tolist() == [0, 2]
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        load_completed(directory / "scientific.npz")
