"""Integration tests for CLI commands."""

import pytest
from typer.testing import CliRunner

# Skip if typer not available
pytest.importorskip("typer")


class TestCLICommands:
    """Test CLI command invocation."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def app(self):
        """Import the CLI app."""
        from peakfit.cli.app import app

        return app

    def test_app_no_args(self, runner, app):
        """CLI with no args should show help."""
        result = runner.invoke(app, [])
        # Typer returns exit code 2 for --no-args-is-help (missing required args)
        assert result.exit_code in [0, 2]  # Both are acceptable
        assert "PeakFit" in result.output or "Usage" in result.output

    def test_version_flag(self, runner, app):
        """--version should show version."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower() or "0." in result.output

    def test_help_flag(self, runner, app):
        """--help should show help text."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "fit" in result.output
        assert "validate" in result.output
        assert "init" in result.output
        assert "info" in result.output

    def test_fit_command_help(self, runner, app):
        """fit --help should show fit-specific options."""
        result = runner.invoke(app, ["fit", "--help"])
        assert result.exit_code == 0
        assert "--parallel" in result.output
        assert "--fast" in result.output
        assert "--workers" in result.output
        assert "--refine" in result.output
        assert "--lineshape" in result.output

    def test_info_command(self, runner, app):
        """info command should run without errors."""
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "PeakFit" in result.output
        assert "NumPy" in result.output or "Numba" in result.output

    def test_init_command_creates_file(self, runner, app, tmp_path):
        """init command should create config file."""
        config_path = tmp_path / "test_config.toml"
        result = runner.invoke(app, ["init", str(config_path)])
        assert result.exit_code == 0
        assert config_path.exists()
        content = config_path.read_text()
        assert "[fitting]" in content
        assert "[clustering]" in content
        assert "[output]" in content

    def test_init_command_no_overwrite(self, runner, app, tmp_path):
        """init should not overwrite without --force."""
        config_path = tmp_path / "test_config.toml"
        config_path.write_text("# existing config")

        result = runner.invoke(app, ["init", str(config_path)])
        assert result.exit_code == 1
        assert "already exists" in result.output

    def test_init_command_force_overwrite(self, runner, app, tmp_path):
        """init --force should overwrite existing file."""
        config_path = tmp_path / "test_config.toml"
        config_path.write_text("# existing config")

        result = runner.invoke(app, ["init", str(config_path), "--force"])
        assert result.exit_code == 0
        content = config_path.read_text()
        assert "[fitting]" in content

    def test_validate_command_missing_files(self, runner, app, tmp_path):
        """validate should fail with missing files."""
        fake_spectrum = tmp_path / "nonexistent.ft2"
        fake_peaklist = tmp_path / "nonexistent.list"

        result = runner.invoke(app, ["validate", str(fake_spectrum), str(fake_peaklist)])
        # Should fail because files don't exist
        assert result.exit_code != 0

    def test_benchmark_command_help(self, runner, app):
        """benchmark --help should show benchmark options."""
        result = runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "--iterations" in result.output
        assert "--z-values" in result.output

    def test_plot_command_help(self, runner, app):
        """plot --help should show plot options."""
        result = runner.invoke(app, ["plot", "--help"])
        assert result.exit_code == 0
        assert "--type" in result.output
        assert "--output" in result.output
        assert "--show" in result.output


class TestCLIFitOptions:
    """Test specific fit command options."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def app(self):
        """Import the CLI app."""
        from peakfit.cli.app import app

        return app

    def test_fit_missing_required_args(self, runner, app):
        """fit without required args should fail."""
        result = runner.invoke(app, ["fit"])
        # Missing SPECTRUM argument
        assert result.exit_code != 0

    def test_fit_invalid_lineshape(self, runner, app, tmp_path):
        """fit with invalid lineshape should fail validation."""
        # Create dummy files
        spectrum = tmp_path / "test.ft2"
        peaklist = tmp_path / "test.list"
        spectrum.touch()
        peaklist.touch()

        result = runner.invoke(
            app,
            ["fit", str(spectrum), str(peaklist), "--lineshape", "invalid_shape"],
        )
        # Invalid lineshape
        assert result.exit_code != 0

    def test_fit_negative_refine(self, runner, app, tmp_path):
        """fit with negative refine should fail validation."""
        spectrum = tmp_path / "test.ft2"
        peaklist = tmp_path / "test.list"
        spectrum.touch()
        peaklist.touch()

        result = runner.invoke(
            app,
            ["fit", str(spectrum), str(peaklist), "--refine", "-1"],
        )
        # Negative refine iterations
        assert result.exit_code != 0


class TestProfilingUtilities:
    """Test profiling utilities."""

    def test_profiler_context_manager(self):
        """Profiler context manager should record timings."""
        import time

        from peakfit.core.profiling import Profiler

        profiler = Profiler()

        with profiler.timer("test_operation"):
            time.sleep(0.01)

        report = profiler.finalize()
        assert len(report.timings) == 1
        assert report.timings[0].name == "test_operation"
        assert report.timings[0].elapsed >= 0.01

    def test_profiler_start_stop(self):
        """Profiler start/stop should record timings."""
        import time

        from peakfit.core.profiling import Profiler

        profiler = Profiler()

        profiler.start("manual_timing")
        time.sleep(0.01)
        elapsed = profiler.stop(count=5, extra="data")

        report = profiler.finalize()
        assert elapsed >= 0.01
        assert len(report.timings) == 1
        assert report.timings[0].count == 5
        assert report.timings[0].metadata["extra"] == "data"

    def test_timing_result_per_call(self):
        """TimingResult should compute per-call average."""
        from peakfit.core.profiling import TimingResult

        result = TimingResult(name="test", elapsed=1.0, count=10)
        assert result.per_call == 0.1

    def test_profile_report_summary(self):
        """ProfileReport should generate summary."""
        from peakfit.core.profiling import ProfileReport, TimingResult

        report = ProfileReport()
        report.add_timing(TimingResult("op1", 0.5))
        report.add_timing(TimingResult("op2", 1.0))
        report.finalize()

        summary = report.summary()
        assert "op1" in summary
        assert "op2" in summary
        assert "Total time" in summary


class TestFastFitErrorHandling:
    """Test error handling in fast_fit module."""

    def test_negative_noise_raises(self):
        """Negative noise should raise ValueError."""
        from unittest.mock import MagicMock

        from peakfit.core.fast_fit import fit_cluster_fast

        cluster = MagicMock()
        cluster.peaks = [MagicMock()]

        with pytest.raises(ValueError, match="positive"):
            fit_cluster_fast(cluster, noise=-1.0)

    def test_zero_noise_raises(self):
        """Zero noise should raise ValueError."""
        from unittest.mock import MagicMock

        from peakfit.core.fast_fit import fit_cluster_fast

        cluster = MagicMock()
        cluster.peaks = [MagicMock()]

        with pytest.raises(ValueError, match="positive"):
            fit_cluster_fast(cluster, noise=0.0)

    def test_empty_peaks_raises(self):
        """Cluster with no peaks should raise FastFitError."""
        from unittest.mock import MagicMock

        from peakfit.core.fast_fit import FastFitError, fit_cluster_fast

        cluster = MagicMock()
        cluster.peaks = []

        with pytest.raises(FastFitError, match="no peaks"):
            fit_cluster_fast(cluster, noise=1.0)
