from __future__ import annotations

import csv
import json
from dataclasses import replace

import numpy as np
import pytest
from test_competing_fit_truth import _final_outcome_for_json, _spectral_parameters, _terminal_result
from test_varpro_numerics import _gaussian_group
from typer.testing import CliRunner

from peakfit.cli.app import app
from peakfit.cli.commands import fit
from peakfit.engine.algorithms import varpro
from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.constraints import ParameterConstraint, PeakConstraints
from peakfit.engine.domain.fit_steps import FitStep
from peakfit.engine.domain.spectrum import Spectra
from peakfit.fit import linewidth_comparison as comparison
from peakfit.fit.fitting import run_fit
from peakfit.fit.linewidth_comparison import (
    _candidate_rows,
    alternative_config,
    initial_parameters,
    run_linewidth_comparison,
)
from peakfit.fit.run_models import LoadedData


@pytest.fixture
def synthetic():
    cluster, truth = _gaussian_group(6.0)
    spectra = Spectra(
        dic={},
        data=cluster.data.T.copy(),
        z_values=np.arange(4.0),
        params=[
            _spectral_parameters(size=4, direct=False),
            _spectral_parameters(size=101, direct=True),
        ],
    )
    return LoadedData(
        spectra, cluster.peaks, 0.01, "synthetic", ["gaussian"], 0.05, [cluster]
    ), truth


def test_eligibility_preserves_explicit_equal_default_fixed_bounds_and_steps(synthetic, tmp_path):
    data, _ = synthetic
    config = PeakFitConfig()
    default = initial_parameters(data, config)["A.F2.lw"].value
    config.parameters.peaks["A"] = PeakConstraints(
        parameters={"F2.lw": ParameterConstraint(value=default)}
    )
    alternative, records = alternative_config(data, config, 13)
    assert initial_parameters(data, alternative)["A.F2.lw"].value == default
    assert initial_parameters(data, alternative)["B.F2.lw"].value == 13
    assert records[0]["reason"] == "explicit start"
    before, after = initial_parameters(data, config), initial_parameters(data, alternative)
    for name in before:
        if name != "B.F2.lw":
            assert before[name].model_dump() == after[name].model_dump()
    config.parameters.peaks["B"] = PeakConstraints(
        parameters={"F2.lw": ParameterConstraint(vary=False)}
    )
    with pytest.raises(ValueError, match="ineffective"):
        alternative_config(data, config, 13)
    config.parameters.peaks.clear()
    config.fitting.steps = [FitStep(fix=["A.*.lw"])]
    alternative, _ = alternative_config(data, config, 13)
    assert initial_parameters(data, alternative)["A.F2.lw"].value == default
    config.fitting.steps = []
    config.parameters.defaults.patterns["*.*.lw"] = ParameterConstraint(min=15)
    with pytest.raises(ValueError, match=r"outside.*bounds"):
        alternative_config(data, config, 13)
    config.parameters.defaults.patterns.clear()
    source = tmp_path / "starts.json"
    source.write_text(
        json.dumps({"clusters": [{"parameters": [{"name": "A.F2.lw", "value": default}]}]})
    )
    config.parameters.from_file = source
    alternative, _ = alternative_config(data, config, 13)
    assert initial_parameters(data, alternative)["A.F2.lw"].value == default
    assert initial_parameters(data, alternative)["B.F2.lw"].value == 13


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_starts(synthetic, value):
    with pytest.raises(ValueError, match="finite and positive"):
        alternative_config(synthetic[0], PeakFitConfig(), value)


def test_actual_optimizer_starts_independence_truth_and_provenance(
    synthetic, tmp_path, monkeypatch
):
    data, truth = synthetic
    config = PeakFitConfig()
    config.fitting.refine_iterations = 2
    config.output.include_timestamp = False
    config.output.formats = ["txt"]  # canonical JSON and tables are still retained

    original = varpro.scipy.optimize.least_squares
    submitted = []

    def capture(function, x0, *args, **kwargs):
        submitted.append(x0.copy())
        return original(function, x0, *args, **kwargs)

    monkeypatch.setattr(varpro.scipy.optimize, "least_squares", capture)
    original_data = data.clusters[0].data.copy()
    assert run_linewidth_comparison(data, config, tmp_path / "first", 13, {}, workers=1)
    assert len(submitted) == 4
    np.testing.assert_allclose(submitted[0][1::2], 25)
    np.testing.assert_allclose(submitted[2][1::2], 13)
    assert not np.allclose(submitted[3][1::2], 13)  # own warm start
    np.testing.assert_array_equal(data.clusters[0].data, original_data)
    assert not data.clusters[0].corrections.any()
    manifest = json.loads((tmp_path / "first/comparison.json").read_text())
    assert manifest["initialization"][0]["alternative_hz"] == 13
    for name in ("baseline", "alternative-13hz"):
        document = json.loads((tmp_path / "first" / name / "summary/fit.json").read_text())
        cluster = document["clusters"][0]
        for p in cluster["final_nonlinear_parameters"]:
            assert p["value"] == pytest.approx(truth[p["name"]].value, abs=2e-5)
        # Reconstruct the actual signed-model least-squares objective independently.
        params = truth.copy(deep=True)
        for p in cluster["final_nonlinear_parameters"]:
            params[p["name"]].value = p["value"]
        shapes = data.clusters[0].evaluate(params)
        amplitudes = np.linalg.lstsq(shapes.T, original_data, rcond=None)[0]
        chi = np.sum(((original_data - shapes.T @ amplitudes) / data.noise) ** 2)
        assert chi < 1e-10
        assert document["noise"] == data.noise
        assert document["z_axis"]["values"] == list(data.spectra.z_values)
    rows = list(csv.DictReader((tmp_path / "first/comparison.csv").read_text().splitlines()))
    assert {r["cluster_id"] for r in rows} == {"1"}
    assert {r["peak_identities"] for r in rows} == {"A; B"}
    assert run_linewidth_comparison(data, config, tmp_path / "repeat", 13, {}, workers=1)
    assert (tmp_path / "first/comparison.csv").read_bytes() == (
        tmp_path / "repeat/comparison.csv"
    ).read_bytes()
    with pytest.raises(ValueError, match="must be empty"):
        run_linewidth_comparison(data, config, tmp_path / "first", 13, {}, workers=1)
    # Baseline agrees with the unmodified single-run entry point.
    normal = run_fit(data, config, tmp_path / "normal", workers=1)
    assert normal.outcome.statistics.chi_squared < 1e-10


@pytest.mark.parametrize("failed_candidate", ["baseline", "alternative-13hz"])
def test_one_candidate_failure_preserves_other(synthetic, tmp_path, monkeypatch, failed_candidate):
    original = comparison.run_fit

    def fail(data, config, directory, **kwargs):
        if directory.name == failed_candidate:
            data.clusters[0].corrections[:] = 999
            raise RuntimeError("deliberate failure")
        assert not data.clusters[0].corrections.any()
        return original(data, config, directory, **kwargs)

    monkeypatch.setattr(comparison, "run_fit", fail)
    assert not run_linewidth_comparison(synthetic[0], PeakFitConfig(), tmp_path, 13, {}, workers=1)
    rows = list(csv.DictReader((tmp_path / "comparison.csv").read_text().splitlines()))
    failed = [r for r in rows if r["candidate"] == failed_candidate]
    assert failed[0]["classification"] == "run_failed"
    assert failed[0]["chi_squared"] == ""
    other = "baseline" if failed_candidate != "baseline" else "alternative-13hz"
    assert (tmp_path / other / "summary/fit.json").exists()
    assert not synthetic[0].clusters[0].corrections.any()


def test_unusable_candidate_remains_visible(synthetic, tmp_path, monkeypatch):
    data, _ = synthetic

    def unusable(task):
        result = _terminal_result(
            task.cluster, success=False, residual_value=np.nan, nfev=1, message="failed evaluation"
        )
        result.noise = task.noise
        return result

    monkeypatch.setattr("peakfit.fit.pipeline.fit_single_cluster_task", unusable)
    assert run_linewidth_comparison(data, PeakFitConfig(), tmp_path, 13, {}, workers=1)
    rows = list(csv.DictReader((tmp_path / "comparison.csv").read_text().splitlines()))
    assert len(rows) == 2
    assert all(r["classification"] == "unusable" and not r["chi_squared"] for r in rows)


def test_cli_without_comparison_retains_single_output(synthetic, tmp_path, monkeypatch):
    data, _ = synthetic

    monkeypatch.setattr(fit, "_load_fit_data", lambda **_kwargs: data)
    spectrum, peaks = tmp_path / "spectrum.ft2", tmp_path / "peaks.list"
    spectrum.touch()
    peaks.write_text("Assignment w1\nA 0\nB 0.01\n")
    monkeypatch.setattr(fit, "validate_inputs", lambda *_args: type("Valid", (), {"errors": []})())
    result = CliRunner().invoke(
        app,
        [
            "fit",
            str(spectrum),
            str(peaks),
            "--headless",
            "--workers",
            "1",
            "-o",
            str(tmp_path / "normal"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(list((tmp_path / "normal").glob("*/summary/fit.json"))) == 1
    assert not list((tmp_path / "normal").rglob("comparison.*"))


def test_rows_align_reordered_clusters_and_retain_nonconvergence_and_warnings(synthetic, tmp_path):
    outcome, _, _ = _final_outcome_for_json(
        {11: (True, True), 37: (False, True), 91: (False, False)}
    )
    baseline = replace(run_fit(synthetic[0], PeakFitConfig(), tmp_path, workers=1), outcome=outcome)
    warned = outcome.clusters[0]
    assert warned.analytical_evaluation is not None
    warned = replace(
        warned,
        analytical_evaluation=replace(
            warned.analytical_evaluation, uncertainty_warning="test warning"
        ),
        final_nonlinear_parameters=tuple(
            replace(p, standard_error=np.nan) for p in warned.final_nonlinear_parameters
        ),
    )
    alternative = replace(
        baseline,
        outcome=replace(outcome, clusters=(outcome.clusters[2], outcome.clusters[1], warned)),
    )
    rows = _candidate_rows(alternative, "alternative", baseline)
    assert {r["cluster_id"] for r in rows} == {11, 37, 91}
    assert all(
        r["position_delta_ppm"] == 0
        for r in rows
        if r["parameter"] and r["parameter"].endswith(".cs")
    )
    assert {r["classification"] for r in rows} == {"converged", "usable_non_converged", "unusable"}
    assert any(r["uncertainty_warning"] for r in rows)
    assert any(r["unavailable_parameter_uncertainties"] for r in rows)
    assert all(r["chi_squared"] is None for r in rows if r["cluster_id"] == 91)


def test_cli_comparison_retains_both_candidates(synthetic, tmp_path, monkeypatch):
    monkeypatch.setattr(fit, "_load_fit_data", lambda **_kwargs: synthetic[0])
    monkeypatch.setattr(fit, "validate_inputs", lambda *_args: type("Valid", (), {"errors": []})())
    spectrum, peaks = tmp_path / "spectrum.ft2", tmp_path / "peaks.list"
    spectrum.touch()
    peaks.touch()
    result = CliRunner().invoke(
        app,
        [
            "fit",
            str(spectrum),
            str(peaks),
            "--headless",
            "--workers",
            "1",
            "--compare-linewidth-start",
            "13",
            "-o",
            str(tmp_path / "comparison"),
        ],
    )
    assert result.exit_code == 0, result.output
    manifests = list((tmp_path / "comparison").glob("*/comparison.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text())["status"] == "complete"
    assert (manifests[0].parent / "baseline/summary/fit.json").exists()
    assert (manifests[0].parent / "alternative-13hz/summary/fit.json").exists()


def test_output_failure_does_not_claim_complete_execution(synthetic, tmp_path, monkeypatch):
    original = comparison.write_fit_run_outputs

    def fail(run, *args, **kwargs):
        if run.output_dir.name == "baseline":
            raise OSError("output unavailable")
        return original(run, *args, **kwargs)

    monkeypatch.setattr(comparison, "write_fit_run_outputs", fail)
    assert not run_linewidth_comparison(synthetic[0], PeakFitConfig(), tmp_path, 13, {}, workers=1)
    manifest = json.loads((tmp_path / "comparison.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["candidates"]["baseline"]["status"] == "failed"
    assert manifest["candidates"]["alternative-13hz"]["status"] == "complete"
    rows = list(csv.DictReader((tmp_path / "comparison.csv").read_text().splitlines()))
    assert all(r["execution_status"] == "failed" for r in rows if r["candidate"] == "baseline")
