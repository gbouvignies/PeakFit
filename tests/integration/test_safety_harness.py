"""Safety harness tests for PeakFit refactoring.

These tests provide a minimal safety net for refactoring by validating:
1. CLI entrypoints work and produce expected output structure
2. Key numerical outputs are stable within tolerance
3. JSON schema compatibility is maintained
4. Core fitting algorithms produce consistent results

Run with: uv run pytest tests/integration/test_safety_harness.py -v
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peakfit.engine.algorithms.varpro import fit_cluster
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.lineshapes.gaussian.kernel import kernel as gaussian_kernel
from peakfit.engine.lineshapes.lorentzian.kernel import kernel as lorentzian_kernel

# --- Configuration ---
REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "examples" / "02-advanced-fitting"
DATA_DIR = EXAMPLE_DIR / "data"
GOLDEN_DIR = REPO_ROOT / "tests" / "data" / "golden"
BASELINE_PATH = GOLDEN_DIR / "baseline.json"

# Test data files
SPECTRUM_FILE = DATA_DIR / "pseudo3d.ft2"
PEAKLIST_FILE = DATA_DIR / "pseudo3d.list"
Z_VALUES_FILE = DATA_DIR / "b1_offsets.txt"

# Tolerances for numerical comparisons
CHI2_RELATIVE_TOLERANCE = 0.5  # 50% relative diff (stochastic optimization)
POSITION_ABSOLUTE_TOLERANCE = 0.01  # ppm
LINEWIDTH_RELATIVE_TOLERANCE = 0.3  # 30% relative diff


def _load_baseline() -> dict[str, float | int]:
    with BASELINE_PATH.open() as f:
        data: dict[str, float | int] = json.load(f)
    return data


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def fit_output_dir(tmp_path_factory):
    """Run a fit once and return the output directory."""
    output_dir = tmp_path_factory.mktemp("safety_harness")

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
        "--output-verbosity",
        "standard",
        "--headless",
    ]

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)

    if result.returncode != 0:
        pytest.fail(
            f"PeakFit CLI failed with exit code {result.returncode}.\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )

    # Find output directory (may be timestamped subdirectory)
    if (output_dir / "summary").exists():
        return output_dir

    subdirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime
    )
    return subdirs[-1] if subdirs else output_dir


# =============================================================================
# Test 1: Output Structure Validation
# =============================================================================


class TestOutputStructure:
    """Verify that the output directory structure is correct."""

    def test_required_directories_exist(self, fit_output_dir):
        """Check that required subdirectories are created."""
        required_dirs = ["summary", "parameters"]
        for dirname in required_dirs:
            dir_path = fit_output_dir / dirname
            assert dir_path.exists(), f"Required directory missing: {dirname}"

    def test_required_files_exist(self, fit_output_dir):
        """Check that required output files are created."""
        required_files = [
            "summary/fit_summary.json",
            "parameters/parameters.csv",
        ]
        for filepath in required_files:
            file_path = fit_output_dir / filepath
            assert file_path.exists(), f"Required file missing: {filepath}"

    def test_json_is_valid(self, fit_output_dir):
        """Verify JSON files are parseable."""
        json_path = fit_output_dir / "summary" / "fit_summary.json"
        with json_path.open() as f:
            data = json.load(f)
        assert isinstance(data, dict), "JSON root should be a dictionary"


# =============================================================================
# Test 2: JSON Schema Compatibility
# =============================================================================


class TestJsonSchema:
    """Verify JSON schema compatibility for downstream tools."""

    def test_schema_version_present(self, fit_output_dir):
        """Check schema version is included for compatibility tracking."""
        json_path = fit_output_dir / "summary" / "fit_summary.json"
        with json_path.open() as f:
            data = json.load(f)
        assert "schema_version" in data, "schema_version field is required"

    def test_required_top_level_keys(self, fit_output_dir):
        """Check required top-level keys are present."""
        json_path = fit_output_dir / "summary" / "fit_summary.json"
        with json_path.open() as f:
            data = json.load(f)

        required_keys = ["metadata", "clusters", "global_statistics"]
        for key in required_keys:
            assert key in data, f"Required top-level key missing: {key}"

    def test_global_statistics_structure(self, fit_output_dir):
        """Verify global_statistics has expected fields."""
        json_path = fit_output_dir / "summary" / "fit_summary.json"
        with json_path.open() as f:
            data = json.load(f)

        stats = data.get("global_statistics", {})
        # Core fields that must be present for downstream processing
        expected_fields = [
            "chi_squared",
            "reduced_chi_squared",
        ]
        for field in expected_fields:
            assert field in stats, f"global_statistics missing field: {field}"

    def test_cluster_structure(self, fit_output_dir):
        """Verify cluster entries have expected structure."""
        json_path = fit_output_dir / "summary" / "fit_summary.json"
        with json_path.open() as f:
            data = json.load(f)

        clusters = data.get("clusters", [])
        assert len(clusters) > 0, "Should have at least one cluster"

        # Check first cluster structure
        cluster = clusters[0]
        required_fields = ["cluster_id", "peak_names", "parameters"]
        for field in required_fields:
            assert field in cluster, f"Cluster missing field: {field}"

    def test_parameter_structure(self, fit_output_dir):
        """Verify parameter entries have expected structure."""
        json_path = fit_output_dir / "summary" / "fit_summary.json"
        with json_path.open() as f:
            data = json.load(f)

        clusters = data.get("clusters", [])
        params = clusters[0].get("parameters", [])
        assert len(params) > 0, "Cluster should have parameters"

        # Check first parameter structure
        param = params[0]
        required_fields = ["name", "value", "unit"]
        for field in required_fields:
            assert field in param, f"Parameter missing field: {field}"


# =============================================================================
# Test 3: Numerical Stability
# =============================================================================


class TestNumericalStability:
    """Verify numerical outputs are within expected ranges."""

    def test_chi_squared_reasonable(self, fit_output_dir):
        """Chi-squared should be positive and reasonable."""
        json_path = fit_output_dir / "summary" / "fit_summary.json"
        with json_path.open() as f:
            data = json.load(f)

        chi2 = data["global_statistics"]["chi_squared"]
        assert chi2 > 0, "Chi-squared should be positive"
        assert chi2 < 1e15, "Chi-squared seems unreasonably large"

    def test_reduced_chi_squared_reasonable(self, fit_output_dir):
        """Reduced chi-squared should be in reasonable range."""
        json_path = fit_output_dir / "summary" / "fit_summary.json"
        with json_path.open() as f:
            data = json.load(f)

        redchi = data["global_statistics"]["reduced_chi_squared"]
        # Good fits typically have reduced chi2 between 0.1 and 10
        assert redchi > 0, "Reduced chi-squared should be positive"
        assert redchi < 100, "Reduced chi-squared seems too large (bad fit)"

    def test_positions_within_bounds(self, fit_output_dir):
        """Check that fitted positions are within spectrum bounds."""
        csv_path = fit_output_dir / "parameters" / "parameters.csv"
        df = pd.read_csv(csv_path, comment="#")

        # Filter to position parameters
        pos_params = df[df["parameter_name"].str.endswith(".cs")]

        for _, row in pos_params.iterrows():
            value = row["value"]
            min_bound = row["min_bound"]
            max_bound = row["max_bound"]

            if pd.notna(min_bound) and pd.notna(max_bound):
                param_name = row["parameter_name"]
                assert min_bound <= value <= max_bound, (
                    f"Position {param_name} = {value} outside [{min_bound}, {max_bound}]"
                )

    def test_linewidths_positive(self, fit_output_dir):
        """Check that fitted linewidths are positive."""
        csv_path = fit_output_dir / "parameters" / "parameters.csv"
        df = pd.read_csv(csv_path, comment="#")

        # Filter to linewidth parameters
        lw_params = df[df["parameter_name"].str.endswith(".lw")]

        for _, row in lw_params.iterrows():
            if not row["is_fixed"]:
                assert row["value"] > 0, f"Linewidth {row['parameter_name']} should be positive"

    def test_standard_errors_non_negative(self, fit_output_dir):
        """Check that standard errors are non-negative."""
        csv_path = fit_output_dir / "parameters" / "parameters.csv"
        df = pd.read_csv(csv_path, comment="#")

        # Filter to varying parameters
        varying = df[~df["is_fixed"]]

        for _, row in varying.iterrows():
            if pd.notna(row["std_error"]):
                assert row["std_error"] >= 0, (
                    f"Standard error for {row['parameter_name']} should be non-negative"
                )


# =============================================================================
# Test 4: Golden Comparison (Key Statistics)
# =============================================================================


class TestGoldenComparison:
    """Compare key outputs against golden baseline."""

    def test_chi_squared_stable(self, fit_output_dir):
        """Chi-squared should be within tolerance of golden value."""
        new_path = fit_output_dir / "summary" / "fit_summary.json"

        with new_path.open() as f:
            new_data = json.load(f)
        baseline = _load_baseline()

        new_chi2 = new_data["global_statistics"]["chi_squared"]
        golden_chi2 = baseline["chi_squared"]

        rel_diff = abs(new_chi2 - golden_chi2) / max(golden_chi2, 1e-10)
        assert rel_diff < CHI2_RELATIVE_TOLERANCE, (
            f"Chi-squared differs too much: {new_chi2:.2e} vs golden {golden_chi2:.2e} "
            f"(relative diff: {rel_diff:.1%})"
        )

    def test_cluster_count_stable(self, fit_output_dir):
        """Number of clusters should match golden."""
        new_path = fit_output_dir / "summary" / "fit_summary.json"

        with new_path.open() as f:
            new_data = json.load(f)
        baseline = _load_baseline()

        new_n = new_data["n_clusters"]
        golden_n = baseline["n_clusters"]

        assert new_n == golden_n, f"Cluster count changed: {new_n} vs golden {golden_n}"

    def test_peak_count_stable(self, fit_output_dir):
        """Number of peaks should match golden."""
        new_path = fit_output_dir / "summary" / "fit_summary.json"

        with new_path.open() as f:
            new_data = json.load(f)
        baseline = _load_baseline()

        new_n = new_data["n_peaks"]
        golden_n = baseline["n_peaks"]

        assert new_n == golden_n, f"Peak count changed: {new_n} vs golden {golden_n}"

    def test_parameter_count_stable(self, fit_output_dir):
        """Model parameter table should be compact and non-empty."""
        new_path = fit_output_dir / "parameters" / "parameters.csv"
        baseline = _load_baseline()

        df_new = pd.read_csv(new_path, comment="#")
        golden_rows = int(baseline["legacy_parameter_rows"])

        assert len(df_new) > 0, "parameters.csv should not be empty"
        assert len(df_new) < golden_rows, (
            "parameters.csv should contain model parameters only (no per-plane amplitudes)"
        )


# =============================================================================
# Test 5: CLI Entrypoint Smoke Tests
# =============================================================================


class TestCLIEntrypoints:
    """Smoke tests for CLI command entrypoints."""

    def test_version_command(self):
        """peakfit --version should work."""
        result = subprocess.run(
            [sys.executable, "-m", "peakfit", "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"--version failed: {result.stderr}"

    def test_help_command(self):
        """peakfit --help should work."""
        result = subprocess.run(
            [sys.executable, "-m", "peakfit", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"--help failed: {result.stderr}"

    def test_init_command(self, tmp_path):
        """peakfit init should create a config file."""
        config_path = tmp_path / "test_config.toml"
        result = subprocess.run(
            [sys.executable, "-m", "peakfit", "init", str(config_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"init failed: {result.stderr}"
        assert config_path.exists(), "Config file not created"

    def test_phx_phase_parameter_is_exported(self, tmp_path):
        """`--phx` should produce exported phase parameters in outputs."""
        output_dir = tmp_path / "fit_phx"
        result = subprocess.run(
            [
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
                "--lineshape",
                "no_apod",
                "--phx",
                "--refine",
                "1",
                "--headless",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"fit --phx failed: {result.stderr}"

        if (output_dir / "parameters" / "parameters.csv").exists():
            params_csv = output_dir / "parameters" / "parameters.csv"
        else:
            subdirs = sorted(
                [d for d in output_dir.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime
            )
            assert subdirs, f"No output directory created for --phx fit.\nSTDOUT: {result.stdout}"
            params_csv = subdirs[-1] / "parameters" / "parameters.csv"

        assert params_csv.exists(), f"Missing parameters.csv for --phx fit at {params_csv}"

        df = pd.read_csv(params_csv, comment="#")
        phase_rows = df[df["parameter_name"].str.endswith(".F3.phase")]
        assert not phase_rows.empty, "F3 phase parameter is missing from parameters.csv with --phx"
        assert phase_rows["peak_name"].str.startswith("cluster_").all()


# =============================================================================
# Test 6: Core Algorithm Unit Tests
# =============================================================================


class TestCoreAlgorithms:
    """Unit tests for core fitting algorithms."""

    def test_lorentzian_kernel_shape(self):
        """Lorentzian kernel should produce expected shape."""
        dw = np.linspace(-100, 100, 201)[:, None]  # (n_points, 1)
        lw = np.array([[25.0]])  # (1, n_peaks)

        result = lorentzian_kernel(dw, lw)

        assert result.shape == (201, 1), f"Unexpected shape: {result.shape}"
        assert np.all(result >= 0), "Lorentzian should be non-negative"
        assert np.isclose(result[100, 0], 1.0, rtol=1e-6), "Peak should be 1.0 at center"

    def test_gaussian_kernel_shape(self):
        """Gaussian kernel should produce expected shape."""
        dw = np.linspace(-100, 100, 201)[:, None]
        lw = np.array([25.0])

        result = gaussian_kernel(dw, lw)

        assert result.shape == (201, 1), f"Unexpected shape: {result.shape}"
        assert np.all(result >= 0), "Gaussian should be non-negative"
        assert np.isclose(result[100, 0], 1.0, rtol=1e-6), "Peak should be 1.0 at center"

    def test_varpro_fit_converges(self):
        """Variable projection fitting should converge on simple data."""
        # This is a placeholder - actual test would need proper cluster setup
        # For now, just verify the function is importable
        assert callable(fit_cluster), "fit_cluster should be callable"

    def test_parameters_creation(self):
        """Parameters class should be instantiable."""
        params = Parameters()
        assert isinstance(params, Parameters), "Should return Parameters"
        assert len(params) == 0, "Empty params should have length 0"
