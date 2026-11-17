"""Plane exclusion: a fit-time choice, recorded with the fit (issue #213).

An excluded plane stays in the measured data and in every output, marked, with no
amplitude. The fit, its noise estimate and its clusters see only the other planes,
so the result must equal a fit of the same spectrum with those planes removed.
"""

import csv
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from builders import completed_result, two_cluster_dataset

from peakfit.cli.commands.fit_setup import build_fit_config
from peakfit.engine.algorithms.covariance_factors import (
    marginal_standard_errors,
    profile_standard_errors,
)
from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters, fitted_plane_indices
from peakfit.fit import fitting
from peakfit.io.config import load_config
from peakfit.io.readers.continuation import load_completed
from peakfit.io.readers.results import ResultsLoader
from peakfit.io.scientific import read_scientific

# Repeated values, so plane identity cannot rest on the value.
PLANE_VALUES = np.array([2.0, 0.0, 2.0, 1.0, 3.0, 0.5])
EXCLUDED = [1, 4]
KEPT = [0, 2, 3, 5]
RTOL = 1e-12


@pytest.fixture(scope="module")
def spectra():
    """Six planes of the two-cluster dataset, the planes to exclude corrupted."""
    amplitudes = np.array(
        [[5.0, -3.0, 2.0, 4.0, 1.0, 3.0], [3.0, -2.0, 4.0, 1.0, 2.5, -1.0]],
    )
    remote = np.array([4.0, -2.0, 3.0, 2.0, 1.0, 5.0])
    spectra, peaks, _config, _prepared = two_cluster_dataset(amplitudes, PLANE_VALUES, remote)
    # Anything leaking from an excluded plane into the fit would show.
    spectra.data[EXCLUDED] *= 40.0
    names = ("1N-H", "2N-H", "3N-H")
    lines = [
        f"{name} {p.positions[0]!r} {p.positions[1]!r}"
        for name, p in zip(names, peaks, strict=True)
    ]
    return spectra, "\n".join(lines) + "\n"


def _removed(spectra: Spectra) -> Spectra:
    """The same spectrum with the excluded planes physically removed."""
    pseudo = spectra.params[0].model_dump(exclude={"delta", "first"}) | {"size": len(KEPT)}
    return Spectra(
        dic=spectra.dic,
        data=spectra.data[KEPT].copy(),
        z_values=spectra.z_values[KEPT].copy(),
        params=[SpectralParameters(**pseudo), *spectra.params[1:]],
    )


def _fit(directory, monkeypatch, spectra, peak_list, exclude):
    """A CLI fit (``load_data``, ``run_fit``, ``write_fit_run_outputs``) of ``spectra``."""
    directory.mkdir()
    source, peaks = directory / "spectrum.ft2", directory / "peaks.list"
    source.write_bytes(b"synthetic spectrum, read through the patched reader")
    peaks.write_text(peak_list)
    monkeypatch.setattr(fitting, "read_spectra", lambda *_: spectra.model_copy(deep=True))
    config = PeakFitConfig.model_validate(
        {
            "fitting": {"lineshape": "gaussian", "refine_iterations": 2},
            "clustering": {"contour_level": 0.1},
            "output": {"formats": ["csv", "txt", "json"], "save_simulated": True},
            "exclude_planes": exclude,
        }
    )
    data = fitting.load_data(source, peaks, None, config)
    run = fitting.run_fit(data, config, directory / "results", workers=1)
    fitting.write_fit_run_outputs(run, config, {"spectrum": source, "peaklist": peaks})
    return run, directory / "results"


@pytest.fixture(scope="module")
def runs(spectra, tmp_path_factory):
    full, peak_list = spectra
    root = tmp_path_factory.mktemp("planes")
    with pytest.MonkeyPatch.context() as monkeypatch:
        excluded = _fit(root / "excluded", monkeypatch, full, peak_list, EXCLUDED)
        removed = _fit(root / "removed", monkeypatch, _removed(full), peak_list, [])
    return excluded, removed


def _close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=0.0, equal_nan=True)


def test_excluding_planes_equals_removing_them(runs):
    (excluded, _), (removed, _) = runs
    a, b = excluded.outcome, removed.outcome
    assert a.noise == pytest.approx(b.noise, rel=RTOL, abs=0.0)
    assert a.noise_model == b.noise_model
    for name in ("chi_squared", "reduced_chi_squared", "n_observations", "degrees_of_freedom"):
        _close(getattr(a.statistics, name), getattr(b.statistics, name))
    assert [c.cluster_id for c in a.clusters] == [c.cluster_id for c in b.clusters]
    for ca, cb in zip(a.clusters, b.clusters, strict=True):
        assert (ca.peak_names, ca.classification) == (cb.peak_names, cb.classification)
        assert [p.name for p in ca.final_nonlinear_parameters] == [
            p.name for p in cb.final_nonlinear_parameters
        ]
        for pa, pb in zip(
            ca.final_nonlinear_parameters, cb.final_nonlinear_parameters, strict=True
        ):
            _close([pa.value, pa.standard_error], [pb.value, pb.standard_error])
        ea, eb = ca.analytical_evaluation, cb.analytical_evaluation
        assert ea is not None
        assert eb is not None
        assert ea.uncertainty_warning == eb.uncertainty_warning
        for field in ("amplitudes", "amplitude_standard_errors", "amplitude_covariance"):
            _close(getattr(ea, field), getattr(eb, field))
        _close(ea.statistics.chi_squared, eb.statistics.chi_squared)
        fa, fb = ea.covariance_factors, eb.covariance_factors
        _close(
            profile_standard_errors(fa, ea.amplitudes), profile_standard_errors(fb, eb.amplitudes)
        )
        _close(
            marginal_standard_errors(fa, ea.amplitudes),
            marginal_standard_errors(fb, eb.amplitudes),
        )


def test_simulated_predictions_cover_the_fitted_planes(runs):
    (_, excluded), (_, removed) = runs
    _close(np.load(excluded / "simulated.npy"), np.load(removed / "simulated.npy"))


def test_the_validation_note_quotes_its_record():
    """docs/validation/plane-exclusion.md quotes tools/validate_plane_exclusion.py's record."""
    docs = Path(__file__).resolve().parents[1] / "docs/validation"
    record = json.loads((docs / "plane-exclusion.json").read_text())
    note = (docs / "plane-exclusion.md").read_text()
    assert record["passed"]
    assert record["excluded_planes"] == [0, 40, 41, 130]
    assert all(q["max_relative"] == 0 and q["bit_identical"] for q in record["quantities"].values())
    assert f"{record['clusters']} clusters" in note
    digits = "".join(note.split())  # the note groups thousands with spaces
    for noise in (record["noise_all_planes"], record["noise_fitted_planes"]):
        assert f"{noise:.2f}" in digits
    shift = record["noise_all_planes"] / record["noise_fitted_planes"] - 1
    assert f"{shift:.0e}" == "6e-05"
    assert "6 × 10⁻⁵ relative" in note


def test_the_noise_estimate_uses_the_fitted_planes_only(spectra, runs):
    """The corrupted planes would change the estimate if they took part in it."""
    from peakfit.engine.algorithms.noise import estimate_noise  # noqa: PLC0415

    full, _ = spectra
    (excluded, _), _ = runs
    assert excluded.outcome.noise == pytest.approx(estimate_noise(full.data[KEPT]), rel=RTOL)
    assert estimate_noise(full.data) > 1.5 * excluded.outcome.noise


def test_the_excluded_set_is_recorded_in_the_outcome_and_archive(runs):
    (excluded, directory), (removed, _) = runs
    assert excluded.outcome.excluded_planes == (1, 4)
    assert removed.outcome.excluded_planes == ()
    outcome, _samples, context = read_scientific(directory / "scientific.npz")
    assert outcome.excluded_planes == (1, 4)
    assert context["source_plane_values"].tolist() == PLANE_VALUES.tolist()
    assert context["source_plane_indices"].tolist() == KEPT
    assert context["plane_values"].tolist() == PLANE_VALUES[KEPT].tolist()
    assert context["data_shape"][0] == len(KEPT)
    assert context["metadata"]["configuration"]["exclude_planes"] == EXCLUDED


def test_excluded_planes_keep_marked_rows_without_amplitude(runs):
    (_, excluded_dir), (_, removed_dir) = runs

    def rows(directory):
        with (directory / "tables/intensities.csv").open() as stream:
            return list(csv.DictReader(stream))

    excluded, removed = rows(excluded_dir), rows(removed_dir)
    assert len(excluded) == len(removed) // len(KEPT) * len(PLANE_VALUES)
    assert all(r["plane_excluded"] == "no" for r in removed)
    fitted = iter(removed)
    for row in excluded:
        plane = int(row["source_plane_index"])
        assert float(row["z_value"]) == PLANE_VALUES[plane]
        if plane in EXCLUDED:
            assert row["plane_excluded"] == "yes"
            for column in (
                "intensity",
                "intensity_err",
                "intensity_err_marginal",
                "intensity_err_conditional",
                "plane_index",
            ):
                assert row[column] == ""
            continue
        reference = next(fitted)
        assert row["plane_excluded"] == "no"
        assert row["peak_name"] == reference["peak_name"]
        assert int(row["plane_index"]) == KEPT.index(plane) == int(reference["plane_index"])
        for column in ("intensity", "intensity_err", "intensity_err_marginal"):
            _close(float(row[column]), float(reference[column]))


def test_excluded_planes_appear_in_json_and_report(runs):
    (_, directory), _ = runs
    document = json.loads((directory / "summary/fit.json").read_text())
    assert document["excluded_planes"] == EXCLUDED
    assert document["z_axis"]["values"] == PLANE_VALUES.tolist()
    assert ResultsLoader(directory).load_summary().excluded_planes == EXCLUDED
    report = (directory / "summary/report.md").read_text()
    assert "Excluded planes (zero-based, not fitted): 1, 4" in report


def test_compact_continuation_reads_the_fitted_planes_of_the_verified_source(
    spectra, runs, monkeypatch
):
    full, _ = spectra
    (excluded, directory), _ = runs
    monkeypatch.setattr(
        "peakfit.io.readers.continuation.read_spectra", lambda *_: full.model_copy(deep=True)
    )
    restored = load_completed(directory / "scientific.npz")
    for a, b in zip(restored.samples, excluded.completed_fit.samples, strict=True):
        np.testing.assert_array_equal(a.data, b.data)
    removed = full.model_copy(update={"data": full.data[KEPT]})
    monkeypatch.setattr("peakfit.io.readers.continuation.read_spectra", lambda *_: removed)
    with pytest.raises(ValueError, match="plane count"):
        load_completed(directory / "scientific.npz")
    (directory.parent / "spectrum.ft2").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        load_completed(directory / "scientific.npz")


def _rewrite(archive, change):
    with ZipFile(archive) as source:
        members = {name: source.read(name) for name in source.namelist()}
    document = json.loads(members["metadata.json"])
    change(document)
    members["metadata.json"] = json.dumps(document).encode()
    with ZipFile(archive, "w") as target:
        for name, payload in members.items():
            target.writestr(name, payload)


def test_an_archive_dropping_a_plane_without_recording_it_is_refused(runs, tmp_path):
    (_, directory), _ = runs
    archive = tmp_path / "scientific.npz"
    archive.write_bytes((directory / "scientific.npz").read_bytes())
    _rewrite(archive, lambda d: d["outcome"].update(excluded_planes=[1]))
    with pytest.raises(ValueError, match="source plane indices"):
        read_scientific(archive)


def test_an_earlier_archive_is_refused_naming_the_peakfit_that_wrote_it(runs, tmp_path):
    (_, directory), _ = runs
    archive = tmp_path / "scientific.npz"
    archive.write_bytes((directory / "scientific.npz").read_bytes())
    _rewrite(archive, lambda d: d.update(schema_version="12.0.0"))
    with pytest.raises(
        ValueError, match=r"Unsupported scientific result version '12\.0\.0'.*open it with PeakFit"
    ):
        read_scientific(archive)


def test_excluding_planes_leaves_the_measured_spectrum_whole(spectra):
    full, _ = spectra
    fitted = full.excluding_planes(EXCLUDED)
    assert fitted.data.shape[0] == len(KEPT)
    assert fitted.source_plane_indices == tuple(KEPT)
    np.testing.assert_array_equal(fitted.data, full.data[KEPT])
    np.testing.assert_array_equal(fitted.z_values, PLANE_VALUES[KEPT])
    assert full.data.shape[0] == len(PLANE_VALUES)
    assert full.source_plane_indices == tuple(range(len(PLANE_VALUES)))
    assert full.excluding_planes([]) is full


@pytest.mark.parametrize(
    ("excluded", "message"),
    [
        ([6], "Cannot exclude plane 6: the spectrum has 6 planes, numbered 0 to 5"),
        ([0, 1, 2, 3, 4, 5], "Every plane is excluded"),
    ],
)
def test_impossible_exclusions_are_refused(excluded, message):
    with pytest.raises(ValueError, match=message):
        fitted_plane_indices(6, excluded)


def test_only_an_explicit_setting_excludes_a_plane(tmp_path):
    assert PeakFitConfig().exclude_planes == []
    cli = build_fit_config(
        config=None,
        output=None,
        formats=None,
        headless=None,
        noise=None,
        contour_level=None,
        lineshape="auto",
        refine=1,
        phx=False,
        phy=False,
        exclude=None,
    )
    assert cli.exclude_planes == []
    toml = tmp_path / "peakfit.toml"
    toml.write_text("[fitting]\nlineshape = 'gaussian'\n")
    assert load_config(toml).exclude_planes == []
    _spectra, _config, pipeline, gui = completed_result([], False)
    assert pipeline.completed_fit.outcome.excluded_planes == ()
    assert gui.outcome.excluded_planes == ()
