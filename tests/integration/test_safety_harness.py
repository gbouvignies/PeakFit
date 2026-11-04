"""Safety harness tests for PeakFit refactoring.

These tests provide a minimal safety net for refactoring by validating:
1. CLI entrypoints work and produce expected output structure
2. Fitted values respect basic invariants (golden values: test_golden_regression.py)
3. JSON schema compatibility is maintained
4. Core fitting algorithms produce consistent results

Run with: uv run pytest tests/integration/test_safety_harness.py -v
"""

import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.lineshapes.gaussian.kernel import kernel as gaussian_kernel
from peakfit.engine.lineshapes.lorentzian.kernel import kernel as lorentzian_kernel
from peakfit.io.mcmc_record import latest_record, read_record
from peakfit.io.readers.results import ResultsLoader
from peakfit.io.schemas import OUTPUT_SCHEMA_VERSION, FitSummarySchema

# --- Configuration ---
REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "examples" / "02-advanced-fitting"
DATA_DIR = EXAMPLE_DIR / "data"

# Test data files
SPECTRUM_FILE = DATA_DIR / "pseudo3d.ft2"
PEAKLIST_FILE = DATA_DIR / "pseudo3d.list"
Z_VALUES_FILE = DATA_DIR / "b1_offsets.txt"


def _cli_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    mpl_config = tmp_path / "matplotlib"
    mpl_config.mkdir(exist_ok=True)
    env["MPLCONFIGDIR"] = str(mpl_config)
    # Rich wraps error panels at the console width, which would make text assertions
    # depend on the temporary path's length (longer under pytest-xdist).
    env["COLUMNS"] = "1000"
    return env


def _link_example_inputs_for_mcmc(results_dir: Path) -> None:
    """Mirror the current MCMC input-resolution convention in a temp tree."""
    data_dir = results_dir.parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    for source in [SPECTRUM_FILE, PEAKLIST_FILE, Z_VALUES_FILE]:
        target = data_dir / source.name
        if not target.exists():
            target.symlink_to(source)


def _summary_payload(directory):
    return ResultsLoader(directory).load_summary().model_dump(mode="json")


def _first_peak_name(results_dir: Path) -> str:
    data = _summary_payload(results_dir)
    return str(data["clusters"][0]["peak_names"][0])


# =============================================================================
# Test 1: Output Structure Validation
# =============================================================================


class TestOutputStructure:
    """Verify that the output directory structure is correct."""

    def test_required_directories_exist(self, example_fit_dir):
        """Check that required subdirectories are created."""
        required_dirs = ["summary", "tables"]
        for dirname in required_dirs:
            dir_path = example_fit_dir / dirname
            assert dir_path.exists(), f"Required directory missing: {dirname}"

    def test_required_files_exist(self, example_fit_dir):
        """Check that required output files are created."""
        required_files = [
            "scientific.npz",
            "summary/report.md",
            "tables/parameters.csv",
            "tables/intensities.csv",
        ]
        for filepath in required_files:
            file_path = example_fit_dir / filepath
            assert file_path.exists(), f"Required file missing: {filepath}"

    def test_native_summary_is_valid(self, example_fit_dir):
        """Verify JSON files are parseable."""
        data = _summary_payload(example_fit_dir)
        assert isinstance(data, dict), "JSON root should be a dictionary"

    def test_no_manifest_file_is_written(self, example_fit_dir):
        """The fixed output layout should not need a separate file index."""
        assert not (example_fit_dir / "manifest.json").exists()


# =============================================================================
# Test 1b: Public Workflow Characterization
# =============================================================================


class TestPublicWorkflows:
    """Characterize public workflows without depending on internal architecture."""

    def test_explicit_peaklist_fit_writes_core_outputs(self, example_fit_dir):
        """`peakfit fit spectrum peaklist` should produce the core public outputs."""
        core_outputs = [
            example_fit_dir / "scientific.npz",
            example_fit_dir / "summary" / "report.md",
            example_fit_dir / "tables" / "parameters.csv",
            example_fit_dir / "tables" / "intensities.csv",
        ]

        for path in core_outputs:
            assert path.exists(), f"Core output missing: {path.relative_to(example_fit_dir)}"
            assert path.stat().st_size > 0, (
                f"Core output is empty: {path.relative_to(example_fit_dir)}"
            )

    def test_scientific_summary_is_readable(self, example_fit_dir):
        """The summary JSON should be parseable and schema-readable."""
        payload = _summary_payload(example_fit_dir)

        summary = FitSummarySchema.model_validate(payload)

        assert summary.clusters
        assert payload["statistics"]["chi_squared"] > 0

    def test_parameters_csv_excludes_amplitudes(self, example_fit_dir):
        """Model parameters and per-plane amplitudes should not be mixed."""
        params_path = example_fit_dir / "tables" / "parameters.csv"
        df = pd.read_csv(params_path, comment="#")

        assert len(df) > 0
        assert "parameter_name" in df.columns
        assert "intensity" not in df.columns
        assert not df["parameter_name"].str.contains(r"\.I\d+$", regex=True).any()
        assert not df["parameter_name"].str.contains("amplitude", case=False, regex=False).any()

    def test_intensities_csv_contains_amplitudes(self, example_fit_dir):
        """Per-plane fitted amplitudes should be written to intensities.csv."""
        intensities_path = example_fit_dir / "tables" / "intensities.csv"
        df = pd.read_csv(intensities_path, comment="#")

        required_columns = {
            "cluster_id",
            "peak_name",
            "plane_index",
            "intensity",
            "intensity_err",
        }
        assert required_columns <= set(df.columns)
        assert len(df) > 0
        assert df["intensity"].notna().all()

    def test_plot_intensity_consumes_fit_output(self, example_fit_dir, tmp_path):
        """Plotting should consume a fit output directory and produce a PDF artifact."""
        output_path = tmp_path / "intensity_profiles.pdf"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "peakfit",
                "plot",
                "intensity",
                str(example_fit_dir),
                "--output",
                str(output_path),
                "--no-show",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=_cli_env(tmp_path),
            timeout=60,
        )

        assert result.returncode == 0, (
            f"plot intensity failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    @pytest.mark.mcmc
    def test_mcmc_can_start_from_fit_output(self, example_fit_dir, tmp_path):
        """MCMC should run for a selected peak from a completed fit output."""
        _link_example_inputs_for_mcmc(example_fit_dir)
        peak_name = _first_peak_name(example_fit_dir)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "peakfit",
                "mcmc",
                str(example_fit_dir),
                "--peaks",
                peak_name,
                "--walkers",
                "32",
                "--steps",
                "100",
                "--burn-in",
                "0",
                "--no-auto-burnin",
                "--workers",
                "1",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=_cli_env(tmp_path),
            timeout=90,
        )

        combined_output = f"{result.stdout}\n{result.stderr}"
        assert result.returncode == 0, (
            f"mcmc failed for peak {peak_name}.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
        assert "Sampling complete" in combined_output

        # The run is recorded, bound to its compact parent's spectrum, and plotted.
        path = latest_record(example_fit_dir)
        assert path is not None
        record = read_record(path)
        info = ResultsLoader(example_fit_dir).load_summary().metadata.input_files
        assert record.parent.spectrum_sha256 == info["spectrum"]["checksum_sha256"]
        (posterior,) = record.posteriors
        assert peak_name in posterior.peak_names
        output_path = tmp_path / "mcmc_diagnostics.pdf"
        plotted = subprocess.run(
            [
                sys.executable,
                "-m",
                "peakfit",
                "plot",
                "mcmc",
                str(example_fit_dir),
                "--output",
                str(output_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=_cli_env(tmp_path),
            timeout=90,
        )
        assert plotted.returncode == 0, f"STDOUT: {plotted.stdout}\nSTDERR: {plotted.stderr}"
        assert output_path.stat().st_size > 0


# =============================================================================
# Test 2: JSON Schema Compatibility
# =============================================================================


class TestScientificSummarySchema:
    """Verify native result summaries retain downstream scientific semantics."""

    def test_schema_version_present(self, example_fit_dir):
        """Check schema version is included for compatibility tracking."""
        data = _summary_payload(example_fit_dir)
        assert data["schema_version"] == OUTPUT_SCHEMA_VERSION

    def test_required_top_level_keys(self, example_fit_dir):
        """Check required top-level keys are present."""
        data = _summary_payload(example_fit_dir)

        required_keys = ["metadata", "clusters", "statistics"]
        for key in required_keys:
            assert key in data, f"Required top-level key missing: {key}"

    def test_final_statistics_structure(self, example_fit_dir):
        """Verify usable-only final statistics have expected fields."""
        data = _summary_payload(example_fit_dir)

        stats = data.get("statistics", {})
        # Core fields that must be present for downstream processing
        expected_fields = [
            "chi_squared",
            "reduced_chi_squared",
        ]
        for field in expected_fields:
            assert field in stats, f"statistics missing field: {field}"

    def test_cluster_structure(self, example_fit_dir):
        """Verify cluster entries have expected structure."""
        data = _summary_payload(example_fit_dir)

        clusters = data.get("clusters", [])
        assert len(clusters) > 0, "Should have at least one cluster"

        # Check first cluster structure
        cluster = clusters[0]
        required_fields = [
            "cluster_id",
            "peak_names",
            "classification",
            "optimizer_provenance",
            "final_nonlinear_parameters",
            "analytical_evaluation",
        ]
        for field in required_fields:
            assert field in cluster, f"Cluster missing field: {field}"

    def test_parameter_structure(self, example_fit_dir):
        """Verify parameter entries have expected structure."""
        data = _summary_payload(example_fit_dir)

        clusters = data.get("clusters", [])
        params = clusters[0].get("final_nonlinear_parameters", [])
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

    def test_positions_within_bounds(self, example_fit_dir):
        """Check that fitted positions are within spectrum bounds."""
        csv_path = example_fit_dir / "tables" / "parameters.csv"
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

    def test_linewidths_positive(self, example_fit_dir):
        """Check that fitted linewidths are positive."""
        csv_path = example_fit_dir / "tables" / "parameters.csv"
        df = pd.read_csv(csv_path, comment="#")

        # Filter to linewidth parameters
        lw_params = df[df["parameter_name"].str.endswith(".lw")]

        for _, row in lw_params.iterrows():
            if not row["is_fixed"]:
                assert row["value"] > 0, f"Linewidth {row['parameter_name']} should be positive"

    def test_standard_errors_non_negative(self, example_fit_dir):
        """Check that available standard errors are numeric and non-negative."""
        csv_path = example_fit_dir / "tables" / "parameters.csv"
        df = pd.read_csv(csv_path, comment="#")

        # Filter to varying parameters
        varying = df[~df["is_fixed"]]
        standard_errors = pd.to_numeric(varying["std_error"], errors="coerce")
        unavailable = varying["std_error"].eq("unavailable")

        assert (standard_errors.notna() | unavailable).all(), "Some std_errors are invalid"
        assert (standard_errors.dropna() >= 0).all(), "Some std_errors are negative"


# =============================================================================
# Test 4: Export Projections
# =============================================================================


class TestExportProjections:
    """CSV, JSON and report projections of the example fit agree with each other."""

    def test_csv_parameters_integrity(self, example_fit_dir):
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

    def test_real_fit_projections_agree_by_cluster_id(self, example_fit_dir):
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
        assert payload["schema_version"] == "6.0.0"

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

        if (output_dir / "tables" / "parameters.csv").exists():
            params_csv = output_dir / "tables" / "parameters.csv"
        else:
            subdirs = sorted(
                [d for d in output_dir.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime
            )
            assert subdirs, f"No output directory created for --phx fit.\nSTDOUT: {result.stdout}"
            params_csv = subdirs[-1] / "tables" / "parameters.csv"

        assert params_csv.exists(), f"Missing parameters.csv for --phx fit at {params_csv}"

        df = pd.read_csv(params_csv, comment="#")
        phase_rows = df[df["parameter_name"].str.endswith(".F3.phase")]
        assert not phase_rows.empty, "F3 phase parameter is missing from parameters.csv with --phx"
        phase_cluster_ids = phase_rows["parameter_name"].str.extract(r"^cluster_(\d+)\.F3\.phase$")[
            0
        ]
        assert phase_cluster_ids.notna().all(), (
            "F3 phase rows must use canonical cluster parameters"
        )
        assert phase_rows["cluster_id"].astype(str).eq(phase_cluster_ids).all()
        assert phase_rows["peak_name"].str.strip().ne("").all()
        assert not phase_rows["peak_name"].str.startswith("cluster_").any()

    def test_fit_missing_spectrum_error_is_actionable(self, tmp_path):
        """Invalid input errors should name the missing user-provided path."""
        missing_spectrum = tmp_path / "missing.ft2"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "peakfit",
                "fit",
                str(missing_spectrum),
                "--headless",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=_cli_env(tmp_path),
        )

        combined_output = f"{result.stdout}\n{result.stderr}"
        assert result.returncode != 0
        assert "missing.ft2" in combined_output
        assert "does not exist" in combined_output


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

    @pytest.mark.parametrize(
        "kernel",
        [gaussian_kernel, lorentzian_kernel],
        ids=["gaussian", "lorentzian"],
    )
    def test_lineshape_fwhm_convention_is_stable(self, kernel):
        """Representative lineshapes should be half-height at +/- FWHM/2."""
        linewidth = 25.0
        dw = np.array([[-0.5 * linewidth], [0.0], [0.5 * linewidth]])
        values = kernel(dw, np.array([linewidth]))

        assert values.shape == (3, 1)
        assert np.isclose(values[1, 0], 1.0, rtol=1e-12, atol=1e-12)
        assert np.allclose(values[[0, 2], 0], 0.5, rtol=1e-12, atol=1e-12)

    def test_parameters_creation(self):
        """Parameters class should be instantiable."""
        params = Parameters()
        assert isinstance(params, Parameters), "Should return Parameters"
        assert len(params) == 0, "Empty params should have length 0"
