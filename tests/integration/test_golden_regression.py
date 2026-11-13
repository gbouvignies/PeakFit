"""The bundled example's CLI fit, compared with its golden reference.

The reference (``tests/data/golden``, written by ``tools/write_golden_reference.py``)
records every cluster's classification, chi-squared, nonlinear parameters,
amplitudes and their conditional and profile σ. The fit is deterministic: repeated
runs, single-threaded BLAS and serial workers reproduce it bit for bit. Tolerances
allow only the rounding-level differences expected between platforms. Starting
positions shifted by 1e-9 ppm moved values by at most 1.5e-6 standard errors
(parameters) and 2.4e-8 (amplitudes), standard errors by 2.6e-10 and chi-squared by
4e-11 (relative). Each tolerance keeps a margin of at least 250x over those
measurements, yet stays at 1e-3 of the reported uncertainty or finer, so exceeding
one is a numerical change to review (docs/development.md, "Golden fit reference").

Clusters whose recorded uncertainties are unreliable (overlapping or
unidentifiable parameters) are compared only through their model: classification,
the warning itself and chi-squared. Their individual amplitudes are not determined
by the data; the same rounding-level change moved them by 4e-4 (relative).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from peakfit.engine.algorithms.covariance_factors import profile_standard_errors
from peakfit.io.scientific import read_scientific

GOLDEN_REFERENCE = Path(__file__).resolve().parents[1] / "data/golden/example-fit.json"

# Fractions of the reference standard error.
VALUE_TOLERANCE_IN_SIGMA = 1e-3
# Relative tolerances.
CHI_SQUARED_RTOL = 1e-8
STANDARD_ERROR_RTOL = 1e-6
BOUND_RTOL = 1e-12


@pytest.fixture(scope="module")
def reference():
    record = json.loads(GOLDEN_REFERENCE.read_text())
    with np.load(GOLDEN_REFERENCE.with_name("example-fit-amplitudes.npz")) as arrays:
        amplitudes = {name: arrays[name] for name in arrays.files}
    return record, amplitudes


@pytest.fixture(scope="module")
def outcome(example_fit_dir):
    outcome, _samples, _context = read_scientific(example_fit_dir / "scientific.npz")
    return outcome


def test_structure_and_statistics_match_reference(outcome, reference):
    record, _ = reference
    expected = record["statistics"]
    statistics = outcome.statistics
    assert statistics.n_observations == expected["n_observations"]
    assert statistics.n_fitted_parameters == expected["n_fitted_parameters"]
    assert statistics.degrees_of_freedom == expected["degrees_of_freedom"]
    assert statistics.chi_squared == pytest.approx(expected["chi_squared"], rel=CHI_SQUARED_RTOL)

    assert [c.cluster_id for c in outcome.clusters] == [c["cluster_id"] for c in record["clusters"]]
    for cluster, expected in zip(outcome.clusters, record["clusters"], strict=True):
        evaluation = cluster.analytical_evaluation
        assert evaluation is not None, cluster.cluster_id
        assert list(cluster.peak_names) == expected["peak_names"]
        assert cluster.classification.value == expected["classification"]
        assert (evaluation.uncertainty_warning is not None) == expected["uncertainty_warning"]
        assert evaluation.statistics.chi_squared == pytest.approx(
            expected["chi_squared"], rel=CHI_SQUARED_RTOL
        ), cluster.cluster_id


def test_identifiable_parameters_match_reference(outcome, reference):
    record, _ = reference
    mismatches = []
    for cluster, expected in zip(outcome.clusters, record["clusters"], strict=True):
        parameters = cluster.final_nonlinear_parameters
        assert [p.name for p in parameters] == [p["name"] for p in expected["parameters"]]
        for parameter, recorded in zip(parameters, expected["parameters"], strict=True):
            assert parameter.vary == recorded["vary"], parameter.name
            assert parameter.min == pytest.approx(recorded["min"], rel=BOUND_RTOL)
            assert parameter.max == pytest.approx(recorded["max"], rel=BOUND_RTOL)
            if expected["uncertainty_warning"]:
                continue
            sigma = recorded["standard_error"]
            assert sigma is not None, f"{parameter.name} lost its reference uncertainty"
            shift = abs(parameter.value - recorded["value"]) / sigma
            if shift > VALUE_TOLERANCE_IN_SIGMA:
                mismatches.append((shift, parameter.name, parameter.value, recorded["value"]))
            assert parameter.standard_error == pytest.approx(sigma, rel=STANDARD_ERROR_RTOL), (
                parameter.name
            )
    worst = sorted(mismatches, reverse=True)[:10]
    assert not mismatches, f"{len(mismatches)} parameters moved (|Δ|/σ, name, new, ref): {worst}"


def test_identifiable_amplitudes_match_reference(outcome, reference):
    record, arrays = reference
    mismatches = []
    for cluster, expected in zip(outcome.clusters, record["clusters"], strict=True):
        if expected["uncertainty_warning"]:
            continue
        evaluation = cluster.analytical_evaluation
        assert evaluation is not None
        recorded = arrays[f"amplitudes_{cluster.cluster_id}"]
        sigma = arrays[f"amplitude_standard_errors_{cluster.cluster_id}"]
        assert evaluation.amplitudes.shape == recorded.shape
        np.testing.assert_allclose(
            evaluation.amplitude_standard_errors, sigma, rtol=STANDARD_ERROR_RTOL
        )
        shift = np.abs(evaluation.amplitudes - recorded) / sigma[:, None]
        if np.max(shift) > VALUE_TOLERANCE_IN_SIGMA:
            mismatches.append((float(np.max(shift)), cluster.cluster_id))
    assert not mismatches, f"Amplitudes moved (max |Δ|/σ, cluster_id): {sorted(mismatches)}"


def test_identifiable_profile_sigma_matches_reference(outcome, reference):
    # intensity_err: the profile σ, per peak and plane (profile-uncertainty §3.3).
    record, arrays = reference
    for cluster, expected in zip(outcome.clusters, record["clusters"], strict=True):
        if expected["uncertainty_warning"]:
            continue
        evaluation = cluster.analytical_evaluation
        assert evaluation is not None
        profile = profile_standard_errors(evaluation.covariance_factors, evaluation.amplitudes)
        recorded = arrays[f"profile_standard_errors_{cluster.cluster_id}"]
        np.testing.assert_allclose(profile, recorded, rtol=STANDARD_ERROR_RTOL)
        if len(cluster.peak_names) == 1:
            # An isolated peak's profile σ is its conditional σ (§3.2).
            np.testing.assert_allclose(
                profile,
                np.broadcast_to(evaluation.amplitude_standard_errors[:, None], profile.shape),
                rtol=1e-12,
            )
