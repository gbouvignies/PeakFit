"""
Regression tests for PeakFit.

This module tests the full fitting pipeline on a real dataset, ensuring:
1. The pipeline runs successfully
2. Output structure is correct
3. Chi-squared values are reasonable
4. Key statistics remain stable

Note: Due to the stochastic nature of nonlinear optimization, exact numerical
reproducibility is not expected. Tests focus on structural correctness and
reasonable fit quality rather than exact value matching.
"""

import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

# --- Configuration ---
# Paths are relative to the repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "examples" / "02-advanced-fitting"
DATA_DIR = EXAMPLE_DIR / "data"
GOLDEN_DIR = REPO_ROOT / "tests" / "data" / "golden"
BASELINE_PATH = GOLDEN_DIR / "baseline.json"

# Inputs for the fit command
SPECTRUM_FILE = DATA_DIR / "pseudo3d.ft2"
PEAKLIST_FILE = DATA_DIR / "pseudo3d.list"
Z_VALUES_FILE = DATA_DIR / "b1_offsets.txt"

# Tolerances
FLOAT_TOLERANCE = 1e-5


def _load_baseline() -> dict[str, float | int]:
    with BASELINE_PATH.open() as f:
        data: dict[str, float | int] = json.load(f)
    return data


@pytest.fixture(scope="module")
def run_golden_fit(tmp_path_factory):
    """
    Runs the PeakFit fit command once per test session.
    Returns the directory containing the new output.
    """
    output_dir = tmp_path_factory.mktemp("regression_output")

    # Construct the command
    # We run via python -m to ensure we use the current source code
    cmd = [
        sys.executable,
        "-m",
        "peakfit",
        "fit",
        str(SPECTRUM_FILE),
        str(PEAKLIST_FILE),
        "--z-values",
        str(Z_VALUES_FILE),
        "--output",
        str(output_dir),
        "--refine",
        "1",
        "--headless",
        "--verbose",
        "--format",
        "json",
        "--format",
        "csv",
        "--format",
        "txt",
    ]

    result = subprocess.run(cmd, check=False, text=True)

    if result.returncode != 0:
        pytest.fail(f"PeakFit CLI failed with exit code {result.returncode}. check console output.")

    # Smart detection of output root
    # 1. Check if we generated directly into output_dir (no timestamp/subdir mode)
    if (output_dir / "summary").exists():
        return output_dir

    # 2. Check for timestamped subdirectory (standard PeakFit mode)
    subdirs = sorted([d for d in output_dir.iterdir() if d.is_dir()], key=os.path.getmtime)
    if subdirs:
        # Assuming the generated folder is the last modified
        return subdirs[-1]

    return output_dir


def test_json_output_integrity(run_golden_fit):
    """
    Tests fit.json structure and reasonable fit quality.

    We cannot test exact values due to stochastic optimization, but we can:
    1. Verify the output file exists and has correct structure
    2. Verify chi-squared is reasonable (between 0.1 and 100)
    3. Verify expected keys are present
    """
    output_dir = run_golden_fit
    new_json_path = output_dir / "summary" / "fit.json"

    # Unconditional debug of directory structure
    sys.stderr.write(f"\n[DEBUG] Checking output in: {output_dir}\n")
    if output_dir.exists():
        sys.stderr.write(f"[DEBUG] Tree of {output_dir}:\n")
        for p in sorted(output_dir.rglob("*")):
            sys.stderr.write(f"  {p}\n")
    else:
        sys.stderr.write(f"[DEBUG] {output_dir} does not exist!\n")
    sys.stderr.flush()

    if not new_json_path.exists():
        pytest.fail(f"fit.json missing at {new_json_path}")

    with new_json_path.open() as f:
        new_data = json.load(f)
    baseline = _load_baseline()

    # Check structural integrity - same keys present
    assert "statistics" in new_data, f"Missing 'statistics'. Keys found: {list(new_data.keys())}"

    new_stats = new_data["statistics"]

    # Check essential keys are present (clean-break output contract)
    expected_keys = {
        "chi_squared",
        "reduced_chi_squared",
        "degrees_of_freedom",
        "n_observations",
        "n_fitted_parameters",
    }
    for key in expected_keys:
        assert key in new_stats, f"Missing key '{key}' in statistics"

    # Check chi-squared is reasonable (same order of magnitude)
    new_chi2 = new_stats["chi_squared"]
    golden_chi2 = baseline["chi_squared"]

    # Chi-squared should be positive and reasonable
    assert new_chi2 > 0, "Chi-squared should be positive"

    # Allow 50% relative difference in chi-squared (stochastic optimization)
    relative_diff = abs(new_chi2 - golden_chi2) / max(golden_chi2, 1e-10)
    assert relative_diff < 0.5, (
        f"Chi-squared differs too much: {new_chi2:.2f} vs golden {golden_chi2:.2f} "
        f"(relative diff: {relative_diff:.1%})"
    )


def test_csv_parameters_integrity(run_golden_fit):
    """
    Tests parameters.csv structure and content.

    Validate the clean-break parameter export contract:
    1. Required columns exist
    2. Parameter identifiers are canonical dotted names
    3. No amplitude series (I*) is included in parameters.csv
    4. Values and uncertainties are well-formed
    """
    output_dir = run_golden_fit
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


def test_real_fit_projections_agree_by_cluster_id(run_golden_fit):
    """The representative CLI fit has one consistent outcome across durable views."""
    output_dir = run_golden_fit
    with (output_dir / "summary" / "fit.json").open() as handle:
        payload = json.load(handle)
    clusters = pd.read_csv(output_dir / "tables" / "clusters.csv")
    parameters = pd.read_csv(output_dir / "tables" / "parameters.csv")
    intensities = pd.read_csv(output_dir / "tables" / "intensities.csv")
    report = (output_dir / "summary" / "report.md").read_text(encoding="utf-8")
    readme = (output_dir / "README.md").read_text(encoding="utf-8")

    json_by_id = {cluster["cluster_id"]: cluster for cluster in payload["clusters"]}
    csv_by_id = clusters.set_index("cluster_id")
    assert set(json_by_id) == set(csv_by_id.index)
    assert payload["schema_version"] == "4.0.0"

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
    assert f"- **Unusable clusters**: {classifications['unusable']}" in readme
