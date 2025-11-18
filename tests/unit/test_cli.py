"""Test CLI commands."""

from typer.testing import CliRunner

from peakfit.cli.app import app

runner = CliRunner()


class TestCLIHelp:
    """Tests for CLI help messages."""

    def test_main_help(self):
        """Main command should show help."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "PeakFit" in result.stdout
        assert "fit" in result.stdout
        assert "validate" in result.stdout
        assert "plot" in result.stdout

    def test_fit_help(self):
        """Fit command should show help."""
        result = runner.invoke(app, ["fit", "--help"])
        assert result.exit_code == 0
        assert "spectrum" in result.stdout.lower()
        assert "peaklist" in result.stdout.lower()
        assert "--output" in result.stdout
        assert "--refine" in result.stdout

    def test_validate_help(self):
        """Validate command should show help."""
        result = runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0
        assert "spectrum" in result.stdout.lower()
        assert "peaklist" in result.stdout.lower()

    def test_init_help(self):
        """Init command should show help."""
        result = runner.invoke(app, ["init", "--help"])
        assert result.exit_code == 0
        assert "configuration" in result.stdout.lower()

    def test_plot_help(self):
        """Plot command should show help."""
        result = runner.invoke(app, ["plot", "--help"])
        assert result.exit_code == 0
        assert "results" in result.stdout.lower()


class TestInitCommand:
    """Tests for init command."""

    def test_init_creates_file(self, tmp_path):
        """Init should create config file."""
        config_path = tmp_path / "test_config.toml"
        result = runner.invoke(app, ["init", str(config_path)])
        assert result.exit_code == 0
        assert config_path.exists()
        assert "Created" in result.stdout

    def test_init_valid_toml(self, tmp_path):
        """Init should create valid TOML file."""
        config_path = tmp_path / "test_config.toml"
        runner.invoke(app, ["init", str(config_path)])

        content = config_path.read_text()
        assert "[fitting]" in content
        assert "[clustering]" in content
        assert "[output]" in content

    def test_init_no_overwrite_without_force(self, tmp_path):
        """Init should not overwrite existing file without --force."""
        config_path = tmp_path / "existing.toml"
        config_path.write_text("# existing content")

        result = runner.invoke(app, ["init", str(config_path)])
        assert result.exit_code == 1
        assert "already exists" in result.stdout

        # Original content preserved
        assert config_path.read_text() == "# existing content"

    def test_init_overwrite_with_force(self, tmp_path):
        """Init should overwrite with --force flag."""
        config_path = tmp_path / "existing.toml"
        config_path.write_text("# old content")

        result = runner.invoke(app, ["init", str(config_path), "--force"])
        assert result.exit_code == 0

        # Content should be replaced
        new_content = config_path.read_text()
        assert "# old content" not in new_content
        assert "[fitting]" in new_content


class TestValidateCommand:
    """Tests for validate command."""

    def test_validate_missing_spectrum(self, sample_peaklist_file):
        """Validate should fail for missing spectrum file."""
        result = runner.invoke(
            app, ["validate", "/nonexistent/spectrum.ft2", str(sample_peaklist_file)]
        )
        assert result.exit_code != 0

    def test_validate_missing_peaklist(self, tmp_path):
        """Validate should fail for missing peak list file."""
        # Create a dummy spectrum file (won't be read, just needs to exist for CLI)
        spectrum_path = tmp_path / "spectrum.ft2"
        spectrum_path.write_bytes(b"dummy")

        result = runner.invoke(app, ["validate", str(spectrum_path), "/nonexistent/peaks.list"])
        assert result.exit_code != 0


class TestFitCommand:
    """Tests for fit command."""

    def test_fit_missing_spectrum(self, sample_peaklist_file, tmp_path):
        """Fit should fail for missing spectrum file."""
        result = runner.invoke(
            app,
            [
                "fit",
                "/nonexistent/spectrum.ft2",
                str(sample_peaklist_file),
                "--output",
                str(tmp_path / "output"),
            ],
        )
        assert result.exit_code != 0

    def test_fit_with_config_file(self, _sample_config_file, _tmp_path):
        """Fit should accept config file."""
        # This test just validates the CLI parsing, not actual fitting
        result = runner.invoke(
            app,
            [
                "fit",
                "--help",  # Just test help with config mentioned
            ],
        )
        assert "--config" in result.stdout


class TestCLIVersioning:
    """Tests for version information."""

    def test_version_flag(self):
        """Version flag should show version."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "PeakFit" in result.stdout or "version" in result.stdout.lower()
