"""The bundled example's CLI fit, compared with its golden reference.

The reference (``tests/data/golden``, written by ``tools/write_golden_reference.py``)
records every cluster's classification, chi-squared, nonlinear parameters and
amplitudes. The fit is deterministic: repeated runs, single-threaded BLAS and
serial workers reproduce it bit for bit. Tolerances allow only the rounding-level
differences expected between platforms. Starting positions shifted by 1e-9 ppm
moved values by at most 1.5e-6 standard errors (parameters) and 2.4e-8
(amplitudes), standard errors by 2.6e-10 and chi-squared by 4e-11 (relative).
Each tolerance keeps a margin of at least 250x over those measurements, yet stays
at 1e-3 of the reported uncertainty or finer, so exceeding one is a numerical
change to review (docs/development.md, "Golden fit reference").

Clusters whose recorded uncertainties are unreliable (overlapping or
unidentifiable parameters) are compared only through their model: classification,
the warning itself and chi-squared. Their individual amplitudes are not determined
by the data; the same rounding-level change moved them by 4e-4 (relative).
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


def test_csv_parameters_integrity(example_fit_dir):
    """
    Tests parameters.csv structure and content.

    Validate the clean-break parameter export contract:
    1. Required columns exist
    2. Parameter identifiers are canonical dotted names
    3. No amplitude series (I*) is included in parameters.csv
    4. Values and uncertainties are well-formed
    """
    output_dir = example_fit_dir
    new_csv_path = output_dir / "tables" / "parameters.csv"
    assert new_csv_path.exists(), f"parameters.csv missing at {new_csv_path}"

    # Read CSV file, skipping comment lines
    df_new = pd.read_csv(new_csv_path, comment="#")

    # Check required columns exist
    required_cols = [
        "cluster_id",
        "peak_name",
        "parameter_name",
        "value",
        "std_error",
    ]
    for col in required_cols:
        assert col in df_new.columns, f"Missing required column: {col}"

    assert len(df_new) > 0, "parameters.csv should not be empty"

    # Check canonical parameter naming: peak.axis.label
    canonical_name = re.compile(r"^[^.]+\.[^.]+\.[^.]+$")
    assert df_new["parameter_name"].map(lambda x: bool(canonical_name.match(str(x)))).all(), (
        "parameter_name contains non-canonical identifiers"
    )

    # Ensure amplitudes are excluded from parameters.csv
    assert not df_new["parameter_name"].str.contains(r"\.I\d+$").any(), (
        "parameters.csv should not contain amplitude series parameters"
    )

    # Values are always numerical. Final outcome parameter uncertainties may be
    # unavailable, which the CSV represents explicitly rather than fabricating
    # a numerical error estimate.
    assert not df_new["value"].isna().any(), "Some values are NaN"
    std_errors = pd.to_numeric(df_new["std_error"], errors="coerce")
    unavailable_errors = df_new["std_error"].eq("unavailable")
    assert (std_errors.notna() | unavailable_errors).all(), "Some std_errors are invalid"
    assert (std_errors.dropna() >= 0).all(), "Some std_errors are negative"


def test_real_fit_projections_agree_by_cluster_id(example_fit_dir):
    """The representative CLI fit has one consistent outcome across durable views."""
    output_dir = example_fit_dir
    with (output_dir / "summary" / "fit.json").open() as handle:
        payload = json.load(handle)
    clusters = pd.read_csv(output_dir / "tables" / "clusters.csv")
    parameters = pd.read_csv(output_dir / "tables" / "parameters.csv")
    intensities = pd.read_csv(output_dir / "tables" / "intensities.csv")
    report = (output_dir / "summary" / "report.md").read_text(encoding="utf-8")

    json_by_id = {cluster["cluster_id"]: cluster for cluster in payload["clusters"]}
    csv_by_id = clusters.set_index("cluster_id")
    assert set(json_by_id) == set(csv_by_id.index)
    assert payload["schema_version"] == "4.1.0"

    classifications = Counter(cluster["classification"] for cluster in payload["clusters"])
    for cluster_id, json_cluster in json_by_id.items():
        csv_cluster = csv_by_id.loc[cluster_id]
        assert csv_cluster["classification"] == json_cluster["classification"]
        assert csv_cluster["correction_revision"] == json_cluster["correction_revision"]
        assert (
            csv_cluster["function_evaluations"]
            == json_cluster["optimizer_provenance"]["function_evaluations"]
        )
        for parameter in json_cluster["final_nonlinear_parameters"]:
            row = parameters.loc[
                (parameters["cluster_id"] == cluster_id)
                & (parameters["parameter_name"] == parameter["name"])
            ]
            assert len(row) == 1
            assert row.iloc[0]["value"] == pytest.approx(parameter["value"])

        evaluation = json_cluster["analytical_evaluation"]
        if evaluation is None:
            assert parameters.loc[parameters["cluster_id"] == cluster_id].empty
            assert intensities.loc[intensities["cluster_id"] == cluster_id].empty
            continue
        amplitudes = evaluation["amplitudes"]
        cluster_intensities = intensities.loc[intensities["cluster_id"] == cluster_id]
        assert len(cluster_intensities) == len(json_cluster["peak_names"]) * len(amplitudes[0])
        for row in cluster_intensities.itertuples():
            peak_index = json_cluster["peak_names"].index(row.peak_name)
            assert row.intensity == pytest.approx(amplitudes[peak_index][row.plane_index])

    assert f"- Converged: {classifications['converged']}" in report
    assert f"- Unusable: {classifications['unusable']}" in report
