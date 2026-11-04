"""MCMC records: docs/mcmc-formulation.md §8 and validation V11."""

import hashlib
import json
import shutil
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from builders import completed_result

from peakfit.cli.commands.mcmc_output import display_posterior
from peakfit.engine.scientific.amplitudes import Correlation
from peakfit.engine.scientific.sampler import (
    FORMULATION,
    Exchange,
    SamplerSettings,
    regenerate_amplitudes,
)
from peakfit.engine.scientific.sampling import sampling_problem
from peakfit.gui.fit_export import export_fit_result
from peakfit.io import mcmc_record
from peakfit.io.mcmc_record import latest_record, read_record, write_record
from peakfit.io.readers.results import ResultsLoader
from peakfit.io.scientific import encode_value, read_scientific
from peakfit.mcmc.analysis import run_mcmc_analysis
from peakfit.plot import outputs
from peakfit.plot.outputs import generate_mcmc_diagnostics
from peakfit.shared.exceptions import ScientificIntegrityError
from peakfit.ui.console import console


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


def _colliding(monkeypatch, path):
    """Make the next record's name ``path``, whatever the clock and random suffix."""
    created, suffix = path.stem.rsplit("-", 1)
    token_hex = mcmc_record.secrets.token_hex
    monkeypatch.setattr(
        mcmc_record.secrets, "token_hex", lambda n: suffix if n == 4 else token_hex(n)
    )
    moment = datetime.strptime(created, "%Y%m%dT%H%M%S%fZ").replace(tzinfo=UTC)
    return moment.isoformat()


def test_a_colliding_record_name_is_refused_and_the_record_kept(recorded, tmp_path, monkeypatch):
    directory, analysis = recorded
    copy = tmp_path / "results"
    shutil.copytree(directory, copy)
    existing = copy / "mcmc" / analysis.path.name
    before = existing.read_bytes()
    created = _colliding(monkeypatch, existing)
    with pytest.raises(FileExistsError, match="never overwritten"):
        write_record(copy, replace(analysis.record, created=created, entropy=1))
    assert existing.read_bytes() == before
    assert list((copy / "mcmc").iterdir()) == [existing]


def test_a_record_created_by_another_writer_before_publication_is_kept(
    recorded, tmp_path, monkeypatch
):
    directory, analysis = recorded
    copy = tmp_path / "results"
    shutil.copytree(directory, copy)
    (copy / "mcmc" / analysis.path.name).unlink()
    target = copy / "mcmc" / analysis.path.name
    created = _colliding(monkeypatch, target)
    write_archive = mcmc_record.write_archive

    exists = Path.exists

    def racing(path, document, arrays):
        write_archive(path, document, arrays)
        target.write_bytes(b"another writer's record")

    # The other writer's file appears just after any existence check of the target.
    monkeypatch.setattr(mcmc_record, "write_archive", racing)
    monkeypatch.setattr(Path, "exists", lambda p, **k: p != target and exists(p, **k))
    with pytest.raises(FileExistsError, match="never overwritten"):
        write_record(copy, replace(analysis.record, created=created))
    assert target.read_bytes() == b"another writer's record"
    assert list((copy / "mcmc").iterdir()) == [target]


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
        (lambda d: d.update(schema_version="1.0.0"), "Unsupported MCMC record version '1.0.0'"),
        (lambda d: _posterior(d).pop("reduced_chi_squared"), "Malformed PosteriorSample record"),
        (
            lambda d: _posterior(d).update(reduced_chi_squared=-1.0),
            "Invalid reduced chi-squared",
        ),
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


def test_the_record_keeps_each_parent_clusters_reduced_chi_squared(recorded):
    directory, analysis = recorded
    outcome, _, _ = read_scientific(directory / "scientific.npz")
    record = read_record(analysis.path)
    by_cluster = {
        c.cluster_id: c.analytical_evaluation.statistics.reduced_chi_squared
        for c in outcome.clusters
        if c.analytical_evaluation is not None
    }
    assert {p.cluster_id: p.reduced_chi_squared for p in record.posteriors} == by_cluster
    # Reported as a diagnostic beside the intervals, which share the fit's σ-only
    # convention (the fixture has one cluster on each side of 1).
    for posterior in record.posteriors:
        note = posterior.noise_convention
        assert f"Parent fit χ²ᵣ {posterior.reduced_chi_squared:.3g}" in note
        assert "independent-noise likelihood" in note
        assert "not calibrated for correlated noise, even when the convergence" in note
        assert "not rescaled by χ²ᵣ" in note
        assert "√" not in note
    assert {p.reduced_chi_squared > 1.0 for p in record.posteriors} == {True, False}


def _pages(record, tmp_path, monkeypatch):
    """Each saved PDF page: its title, and per panel the axes title, texts and labels."""
    pages = []
    save = outputs._save_pdf_figure

    def capture(pdf, fig, title=None):
        panels = {
            ax.get_title(): (
                "\n".join(t.get_text() for t in ax.texts),
                ax.get_legend_handles_labels()[1],
            )
            for ax in fig.axes
            if ax.get_visible()
        }
        texts = "\n".join(t.get_text() for t in fig.texts)
        pages.append((title or "", texts, panels))
        save(pdf, fig, title)

    monkeypatch.setattr(outputs, "_save_pdf_figure", capture)
    generate_mcmc_diagnostics(record, tmp_path / "mcmc_diagnostics.pdf")
    return pages


def _marginals(pages):
    panels = {}
    for title, _, page in pages:
        if "Marginal Distributions" in title:
            panels.update(page)
    return panels


def test_a_provisional_run_plots_no_interval_and_says_so_on_every_page(
    recorded, tmp_path, monkeypatch
):
    _, analysis = recorded
    record = read_record(analysis.path)
    assert all(p.provisional for p in record.posteriors)
    pages = _pages(record, tmp_path, monkeypatch)
    for title, texts, _ in pages:
        assert "PROVISIONAL" in (title or texts), title
    panels = _marginals(pages)
    assert set(panels) == {name for p in record.posteriors for name in p.names}
    for text, labels in panels.values():
        assert text == "No final interval\nProvisional"
        assert "68% CI" not in labels
        assert "Median" not in labels
    report = "\n".join(texts for title, texts, _ in pages if not title)
    assert "Noise Convention" in report
    assert "Parent fit χ²ᵣ" in report


def test_a_passed_run_plots_intervals_except_withheld_and_marks_cautions(
    recorded, tmp_path, monkeypatch
):
    _, analysis = recorded
    single, pair = read_record(analysis.path).posteriors

    def passed(posterior, **changes):
        diagnosis = replace(posterior.diagnosis, reasons=())
        return replace(posterior, diagnosis=diagnosis, **changes)

    caution = Correlation("p_0", "p_1", 0, -0.95)
    record = replace(
        analysis.record,
        posteriors=(
            passed(single),
            passed(pair, amplitudes=replace(pair.amplitudes, correlated=(caution,))),
        ),
    )
    pages = _pages(record, tmp_path, monkeypatch)
    assert not any("PROVISIONAL" in title for title, _, _ in pages)
    for name, (text, labels) in _marginals(pages).items():
        assert text.startswith("Median: "), name
        assert "68% CI: [" in text
        assert labels == ["Median", "68% CI"]
        assert ("‡ not separately resolved" in text) == name.startswith(("p_0.", "p_1."))

    # An exchange withholds both peaks' intervals; the run is then provisional too.
    exchange = Exchange("p_0", "p_1", "F2", 0.07)
    record = replace(record, posteriors=(passed(single), passed(pair, exchanges=(exchange,))))
    panels = _marginals(_pages(record, tmp_path, monkeypatch))
    assert panels["p_0.F2.cs"][0] == "No final interval\nWithheld: exchanges identity with p_1"
    assert panels["p_1.F3.lw"][0] == "No final interval\nWithheld: exchanges identity with p_0"
    assert panels["p_2.F2.cs"][0].startswith("Median: ")


def test_the_terminal_reports_status_and_the_noise_convention(recorded):
    _, analysis = recorded
    posterior = read_record(analysis.path).posteriors[0]
    with console.capture() as captured:
        display_posterior(posterior)
    text = captured.get()
    assert "Provisional: no final intervals" in text
    assert posterior.noise_convention.split(";")[0] in " ".join(text.split())
    assert "[" not in text.split("Parameter")[1].split("Amplitudes")[0]
