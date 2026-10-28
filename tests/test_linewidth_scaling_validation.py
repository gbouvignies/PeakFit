"""Protect experimental changes of units without adding production options."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.optimize
from test_varpro_numerics import _gaussian_group

from peakfit.engine.algorithms.varpro import fit_cluster


@pytest.fixture
def investigation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("validate_linewidth_scaling")


def test_physical_scale_is_independent_of_absolute_chemical_shift(investigation):
    _, params = _gaussian_group(6.0)
    expected = investigation.physical_scale(params, {"F2": 500.0})
    for name, param in params.items():
        if name.endswith(".cs"):
            param.max += 1000.0
            param.value += 1000.0
            param.min += 1000.0
    np.testing.assert_array_equal(expected, investigation.physical_scale(params, {"F2": 500.0}))
    np.testing.assert_allclose(expected, [0.02, 10.0, 0.02, 10.0])


def test_width_initialization_respects_fixed_parameters_and_bounds(investigation):
    _, params = _gaussian_group(6.0)
    params["A.F2.lw"].vary = False
    fixed = params["A.F2.lw"].value
    params["B.F2.lw"].value = 7.0
    params["B.F2.lw"].max = 8.0
    investigation.set_widths(params, {"F2": 20.0})
    assert params["A.F2.lw"].value == fixed
    assert params["B.F2.lw"].value == 8.0


@pytest.mark.parametrize("scaling", ["standard", "jac", "physical"])
def test_experimental_solver_recovers_identifiable_truth_and_restores_scipy(investigation, scaling):
    cluster, truth = _gaussian_group(6.0)
    params = truth.copy(deep=True)
    investigation.set_widths(params, {"F2": 6.37})
    original = scipy.optimize.least_squares
    result, _, captured = investigation.solve(cluster, params, 0.01, scaling, {"F2": 500.0})
    assert scipy.optimize.least_squares is original
    assert captured["status"] > 0
    for name in truth:
        assert result.params[name].value == pytest.approx(truth[name].value, abs=2e-6)
        assert np.isfinite(result.params[name].stderr)
    assert np.linalg.norm(result.residual) < 1e-6


def test_common_intensity_noise_and_column_scaling_preserve_physical_model(investigation):
    cluster, truth = _gaussian_group(6.0, noise=0.01)
    result = fit_cluster(truth.copy(deep=True), cluster, 0.01)
    probe = investigation.normalization_probe(cluster, result.params, 0.01)
    assert probe["residual_relative_gap"] < 1e-12
    assert probe["jacobian_relative_gap"] < 1e-12
    assert probe["column_scaled_model_relative_gap"] < 1e-12
    assert probe["column_scaled_amplitude_relative_gap"] < 1e-12
    assert probe["intensity_scaled_amplitude_relative_gap"] < 1e-12
    assert probe["shape_rank_before"] == probe["shape_rank_after"] == 2


def test_synthetic_recovery_distinguishes_assignment_swap_from_model_error(investigation):
    cluster, truth = _gaussian_group(6.0)
    swapped = truth.copy(deep=True)
    for label in ("cs", "lw"):
        swapped[f"A.F2.{label}"].value = truth[f"B.F2.{label}"].value
        swapped[f"B.F2.{label}"].value = truth[f"A.F2.{label}"].value
        swapped[f"A.F2.{label}"].stderr = 0.001
        swapped[f"B.F2.{label}"].stderr = 0.001
    record = investigation.synthetic_record(
        cluster, truth, SimpleNamespace(params=swapped), {"F2": 500.0}
    )
    assert record["max_position_error_hz"] > 1
    assert record["matched_max_position_error_hz"] == 0
    assert record["matched_max_width_error_hz"] == 0
    assert record["permutation"] == [1, 0]
    assert record["position_errors_in_standard_errors"] == [0.0, 0.0]


def test_numerical_archive_round_trip_preserves_complete_records(investigation, tmp_path):
    summary = importlib.import_module("summarize_linewidth_scaling")
    payload = {
        "noise": 11922.544916060584,
        "full_runs": [{"amplitudes": [[-1e12, 1e-300]], "error": None}],
        "selection_controls": {"rounded": {"widths": {"F2": 12.0, "F3": 14.0}}},
        "frozen_correction_sweep": [{"value": 119.833975544}],
        "synthetic": [{"truth": 12.062472221247358}],
    }
    path = tmp_path / "results.json"
    investigation.write_results(payload, path)
    assert summary.read_results(path) == payload
