"""MCMC records: docs/mcmc-formulation.md §8 and validation V11."""

import hashlib
import json
import shutil
from dataclasses import replace
from zipfile import ZipFile

import numpy as np
import pytest
from builders import completed_result

from peakfit.engine.scientific.sampler import FORMULATION, SamplerSettings, regenerate_amplitudes
from peakfit.engine.scientific.sampling import sampling_problem
from peakfit.gui.fit_export import export_fit_result
from peakfit.io.mcmc_record import latest_record, read_record, write_record
from peakfit.io.readers.results import ResultsLoader
from peakfit.io.scientific import encode_value, read_scientific
from peakfit.mcmc.analysis import run_mcmc_analysis
from peakfit.plot.outputs import generate_mcmc_diagnostics
from peakfit.shared.exceptions import ScientificIntegrityError


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """Both clusters of the two-cluster fit, sampled from detailed evidence."""
    result = completed_result([], False, True)[3]
    directory = export_fit_result(result, tmp_path_factory.mktemp("mcmc") / "results")
    analysis = run_mcmc_analysis(directory, SamplerSettings(walkers=16, steps=400), entropy=2026)
    return directory, analysis


def _encoded(record):
    arrays = {}
    return encode_value(record, "record", arrays), arrays


def test_v11_a_record_reads_back_bit_for_bit(recorded):
    directory, analysis = recorded
    assert analysis.path.parent == directory / "mcmc"
    assert latest_record(directory) == analysis.path
    document, arrays = _encoded(read_record(analysis.path))
    expected_document, expected_arrays = _encoded(analysis.record)
    assert document == expected_document
    assert arrays.keys() == expected_arrays.keys()
    for name, array in arrays.items():
        expected = expected_arrays[name]
        assert (array.dtype, array.shape) == (expected.dtype, expected.shape), name
        assert array.tobytes() == expected.tobytes(), name


def test_a_record_is_bound_to_its_parent_and_seed(recorded):
    directory, analysis = recorded
    record = read_record(analysis.path)
    archive = directory / "scientific.npz"
    outcome, _, context = read_scientific(archive)
    assert record.parent.sha256 == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert record.parent.run_id == context["metadata"]["run_id"]
    assert record.parent.spectrum_sha256 is None  # the archive retains its observations
    assert record.parent.correction_revision == outcome.terminal_correction_revision
    assert record.parent.noise == outcome.noise
    assert record.formulation == FORMULATION
    assert set(record.software) == {"peakfit", "numpy", "emcee"}
    assert [p.cluster_id for p in record.posteriors] == [c.cluster_id for c in outcome.clusters]
    for index, posterior in enumerate(record.posteriors):
        assert (posterior.run.entropy, posterior.run.spawn_key) == (2026, (index,))
        # The whole chain, burn-in included.
        assert posterior.run.chain.shape[0] == posterior.settings.steps == 400


def test_v11_regenerated_amplitude_draws_are_bit_identical(recorded):
    directory, analysis = recorded
    completed = ResultsLoader(directory).load_completed()
    for posterior in read_record(analysis.path).posteriors:
        problem = sampling_problem(completed, posterior.cluster_id)
        assert regenerate_amplitudes(problem, posterior) == posterior.amplitudes


def test_v11_summaries_read_back_without_the_parent_or_its_source(recorded, tmp_path):
    _, analysis = recorded
    alone = tmp_path / "alone" / "record.npz"
    alone.parent.mkdir()
    shutil.copy(analysis.path, alone)
    record = read_record(alone)
    for posterior, original in zip(record.posteriors, analysis.record.posteriors, strict=True):
        assert posterior.amplitudes == original.amplitudes
        assert posterior.reasons == original.reasons
        assert posterior.diagnosis.parameters == original.diagnosis.parameters
    plotted = generate_mcmc_diagnostics(record, tmp_path / "mcmc_diagnostics.pdf")
    assert plotted.n_plots >= 2 * len(record.posteriors)
    assert plotted.path.stat().st_size > 0


def test_records_are_never_overwritten_and_the_latest_is_found(recorded, tmp_path):
    directory, analysis = recorded
    copy = tmp_path / "results"
    shutil.copytree(directory, copy)
    later = replace(analysis.record, created="2999-01-01T00:00:00+00:00")
    path = write_record(copy, later)
    assert path != analysis.path
    assert sorted((copy / "mcmc").iterdir()) == sorted([copy / "mcmc" / analysis.path.name, path])
    assert latest_record(copy) == path


def test_a_record_is_refused_when_its_parent_changed(recorded, tmp_path):
    directory, analysis = recorded
    copy = tmp_path / "results"
    shutil.copytree(directory, copy)
    stale = replace(analysis.record, parent=replace(analysis.record.parent, sha256="0" * 64))
    with pytest.raises(ValueError, match="changed while MCMC was sampling"):
        write_record(copy, stale)
    assert list((copy / "mcmc").iterdir()) == [copy / "mcmc" / analysis.path.name]


def _rewritten(source, target, change):
    with ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    document = json.loads(members["metadata.json"])
    change(document)
    members["metadata.json"] = json.dumps(document).encode()
    with ZipFile(target, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return target


def _posterior(document):
    return document["record"]["posteriors"][0]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.update(schema_version="0.9.0"), "Unsupported MCMC record version '0.9.0'"),
        (lambda d: d["record"].update(extra=1), "Malformed MCMCRecord record"),
        (lambda d: d["record"]["parent"].pop("sha256"), "Malformed ParentFit record"),
        (lambda d: d["record"].update(entropy="2026"), "expected int, got str"),
        (lambda d: _posterior(d)["settings"].update(steps=401), "Inconsistent MCMC arrays"),
        (lambda d: _posterior(d)["diagnosis"].update(burn_in=400), "Burn-in outside the chain"),
        (lambda d: _posterior(d)["run"].update(spawn_key=[1]), "was not seeded by this run"),
    ],
)
def test_a_malformed_or_other_version_record_is_refused(recorded, tmp_path, change, message):
    _, analysis = recorded
    damaged = _rewritten(analysis.path, tmp_path / "damaged.npz", change)
    with pytest.raises(ScientificIntegrityError, match="Cannot read MCMC record") as error:
        read_record(damaged)
    assert message in str(error.value)
    assert "Rerun peakfit mcmc" in str(error.value)


def test_a_truncated_record_is_refused(recorded, tmp_path):
    _, analysis = recorded
    truncated = tmp_path / "truncated.npz"
    truncated.write_bytes(analysis.path.read_bytes()[:-100])
    with pytest.raises(ScientificIntegrityError, match="Cannot read MCMC record"):
        read_record(truncated)


def test_the_record_keeps_the_identifiability_blobs_of_each_cluster(recorded):
    _, analysis = recorded
    for posterior in read_record(analysis.path).posteriors:
        blobs = posterior.run.blobs
        assert blobs is not None
        assert blobs.dtype == bool
        peaks = len(posterior.peak_names)
        expected = posterior.run.log_probability.shape + ((peaks,) if peaks > 1 else ())
        assert blobs.shape == expected
        kept = blobs[posterior.diagnosis.burn_in :].reshape(-1, peaks)
        np.testing.assert_array_equal(kept.mean(axis=0), posterior.determined)
