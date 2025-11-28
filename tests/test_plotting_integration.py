"""Tests for integrated plotting functionality."""

import pytest
from typer.testing import CliRunner

import numpy as np


class TestPlottingCLI:
    """Test plotting CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    @pytest.fixture
    def app(self):
        """Import CLI app."""
        from peakfit.cli.app import app

        return app

    @pytest.fixture
    def mock_intensity_data(self, tmp_path):
        """Create mock intensity data file."""
        data_file = tmp_path / "test_peak.out"
        # Create sample intensity data: xlabel, intensity, error
        data = np.array([
            [0, 100.0, 5.0],
            [1, 95.0, 4.5],
            [2, 90.0, 4.0],
            [3, 85.0, 3.5],
            [4, 80.0, 3.0],
        ])
        np.savetxt(data_file, data)
        return data_file

    @pytest.fixture
    def mock_cest_data(self, tmp_path):
        """Create mock CEST data file."""
        data_file = tmp_path / "cest_peak.out"
        # Create sample CEST data: offset, intensity, error
        offsets = np.array([-15000, -5000, -1000, 0, 1000, 5000, 15000])
        intensities = np.array([100, 95, 80, 60, 80, 95, 100])
        errors = np.ones_like(offsets) * 5.0
        data = np.column_stack([offsets, intensities, errors])
        np.savetxt(data_file, data)
        return data_file

    @pytest.fixture
    def mock_cpmg_data(self, tmp_path):
        """Create mock CPMG data file."""
        data_file = tmp_path / "cpmg_peak.out"
        # Create sample CPMG data: ncyc, intensity, error
        data = np.array([
            [0, 100.0, 5.0],  # Reference
            [10, 90.0, 4.5],
            [20, 85.0, 4.0],
            [40, 80.0, 3.5],
            [80, 75.0, 3.0],
        ])
        np.savetxt(data_file, data, fmt=["%d", "%.1f", "%.1f"])
        return data_file

    def test_plot_help(self, runner, app):
        """Test plot command help."""
        result = runner.invoke(app, ["plot", "--help"])
        assert result.exit_code == 0
        assert "intensity" in result.output
        assert "cest" in result.output
        assert "cpmg" in result.output
        assert "spectra" in result.output

    def test_plot_intensity_help(self, runner, app):
        """Test plot intensity subcommand help."""
        result = runner.invoke(app, ["plot", "intensity", "--help"])
        assert result.exit_code == 0
        assert "intensity profile" in result.output.lower()

    def test_plot_intensity_dry_run(self, runner, app, mock_intensity_data, tmp_path):
        """Test intensity plotting (without actually creating plots)."""
        output_file = tmp_path / "test_plots.pdf"

        # This will try to plot, but we're just checking it doesn't crash
        result = runner.invoke(
            app,
            [
                "plot",
                "intensity",
                str(mock_intensity_data.parent),
                "--output",
                str(output_file),
            ],
        )

        # Should complete successfully (exit code 0)
        assert result.exit_code == 0

    def test_plot_cest_requires_results(self, runner, app, tmp_path):
        """Test CEST plotting validates input."""
        nonexistent = tmp_path / "nonexistent"
        result = runner.invoke(
            app,
            [
                "plot",
                "cest",
                str(nonexistent),
            ],
        )
        # Should fail because path doesn't exist
        assert result.exit_code != 0

    def test_plot_cpmg_requires_time_t2(self, runner, app, mock_cpmg_data):
        """Test CPMG plotting requires --time-t2."""
        result = runner.invoke(
            app,
            [
                "plot",
                "cpmg",
                str(mock_cpmg_data.parent),
            ],
        )
        # Should show error about missing --time-t2
        assert result.exit_code != 0
        assert (
            "time-t2" in result.output.lower()
            or "time_t2" in result.output.lower()
            or "missing" in result.output.lower()
        )

    def test_plot_spectra_requires_spectrum(self, runner, app, tmp_path):
        """Test spectra plotting requires --spectrum."""
        result = runner.invoke(
            app,
            [
                "plot",
                "spectra",
                str(tmp_path),
            ],
        )
        # Should show error about missing spectrum
        assert result.exit_code != 0
        assert "spectrum" in result.output.lower() or "missing" in result.output.lower()

    def test_plot_invalid_subcommand(self, runner, app, tmp_path):
        """Test invalid plot subcommand."""
        result = runner.invoke(
            app,
            [
                "plot",
                "invalid_type",
                str(tmp_path),
            ],
        )
        # Should fail with unknown subcommand
        assert result.exit_code != 0


class TestPlottingFunctions:
    """Test plotting module functions directly."""

    def test_import_plot_commands(self):
        """Test that plot command module imports successfully."""
        from peakfit.cli.commands import plot

        assert hasattr(plot, "plot_app")

    def test_plot_commands_registered(self):
        """Test that all plot commands are registered."""
        from peakfit.cli.commands.plot import plot_app

        # Get registered command names
        command_names = [cmd.name for cmd in plot_app.registered_commands]

        # Check public commands exist
        assert "intensity" in command_names
        assert "cest" in command_names
        assert "cpmg" in command_names
        assert "spectra" in command_names
        assert "diagnostics" in command_names

    def test_ncyc_to_nu_cpmg_conversion(self):
        """Test CPMG ncyc to nu_CPMG conversion."""
        from peakfit.plotting.profiles import ncyc_to_nu_cpmg

        ncyc = np.array([0, 10, 20, 40])
        time_t2 = 0.04
        nu_cpmg = ncyc_to_nu_cpmg(ncyc, time_t2)

        # ncyc=0 -> 0.5/time_t2, others -> ncyc/time_t2
        assert nu_cpmg[0] == 0.5 / time_t2  # Reference point
        assert nu_cpmg[1] == 10 / time_t2
        assert nu_cpmg[2] == 20 / time_t2

    def test_intensity_to_r2eff_conversion(self):
        """Test intensity to R2eff conversion."""
        from peakfit.plotting.profiles import intensity_to_r2eff

        intensity = np.array([90.0, 80.0, 70.0])
        intensity_ref = 100.0
        time_t2 = 0.04

        r2eff = intensity_to_r2eff(intensity, intensity_ref, time_t2)

        # Should be -ln(I/I0)/T2
        expected = -np.log(intensity / intensity_ref) / time_t2
        np.testing.assert_array_almost_equal(r2eff, expected)


class TestPlottingBackwardCompatibility:
    """Test that plotting is fully integrated into new CLI."""

    def test_plotting_modules_available_as_library(self):
        """Test that plotting modules can still be imported as library functions."""
        # The main plotting functions are available through the public API
        from peakfit.plotting import (
            make_cest_figure,
            make_cpmg_figure,
            make_intensity_figure,
            plot_corner,
            plot_trace,
        )

        assert callable(make_cest_figure)
        assert callable(make_cpmg_figure)
        assert callable(make_intensity_figure)
        assert callable(plot_trace)
        assert callable(plot_corner)

    def test_no_peakfit_plot_command(self):
        """Verify old peakfit-plot command entry point is removed."""
        import tomllib

        with open("pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        scripts = config["project"]["scripts"]

        # Should have peakfit but not peakfit-plot
        assert "peakfit" in scripts
        assert "peakfit-plot" not in scripts
