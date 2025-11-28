"""Tests for the new output system components."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest

import numpy as np

from peakfit.core.results import (
    AmplitudeEstimate,
    ClusterEstimates,
    FitMethod,
    FitResults,
    FitResultsBuilder,
    FitStatistics,
    MCMCDiagnostics,
    ParameterDiagnostic,
    ParameterEstimate,
)
from peakfit.core.results.diagnostics import ConvergenceStatus
from peakfit.core.results.estimates import ParameterCategory
from peakfit.core.results.fit_results import RunMetadata
from peakfit.io.writers import ResultsWriter
from peakfit.io.writers.base import Verbosity


class TestParameterEstimate:
    """Tests for ParameterEstimate dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a basic parameter estimate."""
        param = ParameterEstimate(
            name="G23N_x0",
            value=100.5,
            std_error=0.5,
            unit="pts",
            category=ParameterCategory.LINESHAPE,
        )
        assert param.name == "G23N_x0"
        assert param.value == 100.5
        assert param.std_error == 0.5
        assert param.unit == "pts"
        assert not param.has_asymmetric_error

    def test_asymmetric_errors(self) -> None:
        """Test parameter with asymmetric confidence intervals."""
        param = ParameterEstimate(
            name="test",
            value=10.0,
            std_error=1.0,
            ci_68_lower=9.2,
            ci_68_upper=11.1,
        )
        assert param.has_asymmetric_error
        assert param.error_lower == pytest.approx(0.8)  # 10.0 - 9.2
        assert param.error_upper == pytest.approx(1.1)  # 11.1 - 10.0

    def test_boundary_detection(self) -> None:
        """Test detection of parameters at boundaries."""
        param = ParameterEstimate(
            name="fraction",
            value=1.0,
            std_error=0.1,
            min_bound=0.0,
            max_bound=1.0,
        )
        assert param.is_at_boundary

    def test_relative_error(self) -> None:
        """Test relative error calculation."""
        param = ParameterEstimate(name="test", value=100.0, std_error=5.0)
        assert param.relative_error == pytest.approx(0.05)


class TestAmplitudeEstimate:
    """Tests for AmplitudeEstimate dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating amplitude estimate."""
        amp = AmplitudeEstimate(
            peak_name="G23N",
            plane_index=0,
            z_value=0.01,
            value=1000.5,
            std_error=25.0,
        )
        assert amp.peak_name == "G23N"
        assert amp.plane_index == 0
        assert amp.z_value == 0.01
        assert amp.value == 1000.5

    def test_to_dict(self) -> None:
        """Test dictionary serialization."""
        amp = AmplitudeEstimate(
            peak_name="G23N",
            plane_index=0,
            z_value=0.01,
            value=1000.0,
            std_error=25.0,
        )
        d = amp.to_dict()
        assert d["peak_name"] == "G23N"
        assert d["plane_index"] == 0
        assert d["value"] == 1000.0


class TestClusterEstimates:
    """Tests for ClusterEstimates dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating cluster estimates."""
        params = [
            ParameterEstimate(name="G23N_x0", value=100.0, std_error=0.5),
            ParameterEstimate(name="G23N_y0", value=200.0, std_error=0.5),
        ]
        amps = [
            AmplitudeEstimate(
                peak_name="G23N", plane_index=0, z_value=0.01, value=1000.0, std_error=25.0
            ),
            AmplitudeEstimate(
                peak_name="G23N", plane_index=1, z_value=0.02, value=900.0, std_error=25.0
            ),
        ]
        cluster = ClusterEstimates(
            cluster_id=0,
            peak_names=["G23N"],
            lineshape_params=params,
            amplitudes=amps,
        )
        assert cluster.n_peaks == 1
        assert cluster.n_lineshape_params == 2
        assert cluster.n_planes == 2

    def test_get_amplitudes_for_peak(self) -> None:
        """Test filtering amplitudes by peak."""
        amps = [
            AmplitudeEstimate(
                peak_name="G23N", plane_index=0, z_value=0.01, value=1000.0, std_error=25.0
            ),
            AmplitudeEstimate(
                peak_name="A45C", plane_index=0, z_value=0.01, value=800.0, std_error=20.0
            ),
        ]
        cluster = ClusterEstimates(
            cluster_id=0,
            peak_names=["G23N", "A45C"],
            lineshape_params=[],
            amplitudes=amps,
        )
        g23n_amps = cluster.get_amplitudes_for_peak("G23N")
        assert len(g23n_amps) == 1
        assert g23n_amps[0].peak_name == "G23N"


class TestFitStatistics:
    """Tests for FitStatistics dataclass."""

    def test_from_residuals(self) -> None:
        """Test computing statistics from residuals."""
        rng = np.random.default_rng(42)
        residuals = rng.normal(0, 1, 100)
        noise = 1.0
        n_params = 5

        stats = FitStatistics.from_residuals(residuals, noise, n_params)

        assert stats.n_data == 100
        assert stats.n_params == 5
        assert stats.dof == 95
        assert stats.chi_squared > 0
        assert stats.aic is not None
        assert stats.bic is not None

    def test_is_good_fit(self) -> None:
        """Test good fit detection."""
        good_fit = FitStatistics(
            chi_squared=100.0,
            reduced_chi_squared=1.05,
            n_data=100,
            n_params=5,
            fit_converged=True,
        )
        assert good_fit.is_good_fit

        bad_fit = FitStatistics(
            chi_squared=300.0,
            reduced_chi_squared=3.15,
            n_data=100,
            n_params=5,
            fit_converged=True,
        )
        assert not bad_fit.is_good_fit


class TestMCMCDiagnostics:
    """Tests for MCMC diagnostics."""

    def test_parameter_diagnostic_from_values(self) -> None:
        """Test automatic status determination."""
        # Good convergence
        good = ParameterDiagnostic.from_values(
            name="param1",
            rhat=1.005,
            ess_bulk=5000.0,
            ess_tail=4000.0,
        )
        assert good.status == ConvergenceStatus.GOOD
        assert len(good.warnings) == 0

        # Poor convergence
        poor = ParameterDiagnostic.from_values(
            name="param2",
            rhat=1.08,
            ess_bulk=50.0,
            ess_tail=40.0,
        )
        assert poor.status == ConvergenceStatus.POOR
        assert len(poor.warnings) > 0

    def test_mcmc_diagnostics_overall_status(self) -> None:
        """Test overall status computation."""
        diags = MCMCDiagnostics(
            n_chains=4,
            n_samples=1000,
            burn_in=500,
            parameter_diagnostics=[
                ParameterDiagnostic.from_values("p1", 1.005, 5000.0, 4000.0),  # GOOD
                ParameterDiagnostic.from_values(
                    "p2", 1.03, 200.0, 150.0
                ),  # MARGINAL (rhat<=1.05, 100<=ess<400)
            ],
        )
        diags.update_overall_status()
        # Overall status is worst among all params
        assert diags.overall_status == ConvergenceStatus.MARGINAL


class TestRunMetadata:
    """Tests for run metadata capture."""

    def test_capture(self) -> None:
        """Test capturing current environment metadata."""
        config = {"fitting": {"lineshape": "auto"}}
        metadata = RunMetadata.capture(config)

        assert metadata.timestamp != ""
        assert metadata.software_version != ""
        assert metadata.python_version != ""
        assert metadata.platform != ""
        assert "fitting" in metadata.configuration

    def test_add_input_file(self, tmp_path: Path) -> None:
        """Test adding input file with checksum."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        metadata = RunMetadata.capture()
        metadata.add_input_file("spectrum", test_file)

        assert "spectrum" in metadata.input_files
        assert "path" in metadata.input_files["spectrum"]
        assert "checksum_sha256" in metadata.input_files["spectrum"]


class TestFitResults:
    """Tests for top-level FitResults."""

    def test_basic_creation(self) -> None:
        """Test creating FitResults."""
        cluster = ClusterEstimates(
            cluster_id=0,
            peak_names=["G23N"],
            lineshape_params=[],
            amplitudes=[],
        )
        results = FitResults(
            method=FitMethod.LEAST_SQUARES,
            clusters=[cluster],
        )

        assert results.n_clusters == 1
        assert results.n_peaks == 1

    def test_to_dict(self) -> None:
        """Test dictionary serialization."""
        cluster = ClusterEstimates(
            cluster_id=0,
            peak_names=["G23N"],
            lineshape_params=[],
            amplitudes=[],
        )
        results = FitResults(
            method=FitMethod.LEAST_SQUARES,
            clusters=[cluster],
            experiment_type="CPMG",
        )
        d = results.to_dict()

        assert "metadata" in d
        assert "method" in d
        assert d["method"] == "least_squares"
        assert d["experiment_type"] == "CPMG"


class TestResultsWriter:
    """Tests for the integrated results writer."""

    @pytest.fixture
    def sample_results(self) -> FitResults:
        """Create sample FitResults for testing."""
        params = [
            ParameterEstimate(name="G23N_x0", value=100.0, std_error=0.5, unit="pts"),
            ParameterEstimate(name="G23N_y0", value=200.0, std_error=0.5, unit="pts"),
        ]
        amps = [
            AmplitudeEstimate(
                peak_name="G23N", plane_index=0, z_value=0.01, value=1000.0, std_error=25.0
            ),
        ]
        cluster = ClusterEstimates(
            cluster_id=0,
            peak_names=["G23N"],
            lineshape_params=params,
            amplitudes=amps,
        )
        return FitResults(
            metadata=RunMetadata.capture({"test": True}),
            method=FitMethod.LEAST_SQUARES,
            clusters=[cluster],
            global_statistics=FitStatistics(chi_squared=100.0, reduced_chi_squared=1.05),
        )

    def test_write_minimal(self, tmp_path: Path, sample_results: FitResults) -> None:
        """Test minimal output writing."""
        writer = ResultsWriter(include_legacy=False)
        written = writer.write_minimal(sample_results, tmp_path)

        # New flat structure uses simpler keys
        assert "parameters" in written
        assert "fit_results" in written
        assert written["parameters"].exists()
        assert written["fit_results"].exists()

    def test_write_all(self, tmp_path: Path, sample_results: FitResults) -> None:
        """Test full output writing."""
        writer = ResultsWriter(include_legacy=False)
        written = writer.write_all(sample_results, tmp_path)

        # Check essential outputs (new flat structure)
        assert "parameters" in written
        assert "fit_results" in written
        assert "report" in written

        # Verify JSON is valid
        fit_json = written["fit_results"]
        data = json.loads(fit_json.read_text())
        assert "metadata" in data
        assert "clusters" in data

    def test_write_for_verbosity(self, tmp_path: Path, sample_results: FitResults) -> None:
        """Test verbosity-controlled writing."""
        writer = ResultsWriter(include_legacy=False)

        # Minimal
        minimal_dir = tmp_path / "minimal"
        minimal_dir.mkdir()
        minimal_written = writer.write_for_verbosity(sample_results, minimal_dir, Verbosity.MINIMAL)
        assert len(minimal_written) < 5  # Few files

        # Full
        full_dir = tmp_path / "full"
        full_dir.mkdir()
        full_written = writer.write_for_verbosity(sample_results, full_dir, Verbosity.FULL)
        assert len(full_written) >= len(minimal_written)


class TestFitResultsBuilder:
    """Tests for the FitResultsBuilder."""

    def test_basic_build(self) -> None:
        """Test basic builder usage."""
        builder = FitResultsBuilder()
        builder.set_metadata(config={"test": True})
        builder.set_z_values(np.array([0.01, 0.02, 0.03]))
        builder.set_experiment_type("CPMG")

        # We can't easily test add_cluster without actual cluster objects,
        # so test the basic build without clusters should raise
        with pytest.raises(ValueError, match="No cluster estimates added"):
            builder.build()

    def test_set_methods_chain(self) -> None:
        """Test that builder methods return self for chaining."""
        builder = FitResultsBuilder()
        result = builder.set_metadata({"test": True})
        assert result is builder

        result = builder.set_z_values(np.array([0.01]))
        assert result is builder

        result = builder.set_fit_method(FitMethod.MCMC)
        assert result is builder
