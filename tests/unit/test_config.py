"""Test configuration loading and saving."""

import tomllib
from pathlib import Path

import pytest

from peakfit.core.domain.config import ClusterConfig, FitConfig, OutputConfig, PeakFitConfig
from peakfit.io.config import generate_default_config, load_config, save_config


class TestConfigLoading:
    """Tests for configuration file loading."""

    def test_load_valid_config(self, sample_config_file):
        """Should load valid TOML configuration."""
        config = load_config(sample_config_file)
        assert config.fitting.lineshape == "gaussian"
        assert config.fitting.refine_iterations == 2
        assert config.clustering.contour_factor == 5.0
        assert config.output.directory == Path("Results")
        assert "csv" in config.output.formats

    def test_load_nonexistent_file(self, tmp_path):
        """Should raise error for nonexistent file."""
        nonexistent = tmp_path / "does_not_exist.toml"
        with pytest.raises(FileNotFoundError):
            load_config(nonexistent)

    def test_load_invalid_toml(self, tmp_path):
        """Should raise error for invalid TOML syntax."""
        invalid_file = tmp_path / "invalid.toml"
        invalid_file.write_text("not valid toml {{{")
        with pytest.raises(tomllib.TOMLDecodeError):
            load_config(invalid_file)

    def test_load_minimal_config(self, tmp_path):
        """Should load minimal config with defaults."""
        minimal_file = tmp_path / "minimal.toml"
        minimal_file.write_text("")
        config = load_config(minimal_file)
        # Should have all defaults
        assert config.fitting.lineshape == "auto"
        assert config.fitting.refine_iterations == 1


class TestConfigSaving:
    """Tests for configuration file saving."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Should save config that can be loaded back."""
        config = PeakFitConfig(
            fitting=FitConfig(lineshape="pvoigt", refine_iterations=5),
            clustering=ClusterConfig(contour_factor=7.5),
            output=OutputConfig(directory=Path("TestOutput"), formats=["json", "csv"]),
        )

        save_path = tmp_path / "roundtrip.toml"
        save_config(config, save_path)

        # Verify file was created
        assert save_path.exists()

        # Load it back
        loaded = load_config(save_path)
        assert loaded.fitting.lineshape == config.fitting.lineshape
        assert loaded.fitting.refine_iterations == config.fitting.refine_iterations
        assert loaded.clustering.contour_factor == config.clustering.contour_factor

    def test_save_creates_valid_toml(self, tmp_path):
        """Saved config should be valid TOML."""
        config = PeakFitConfig()
        save_path = tmp_path / "output.toml"
        save_config(config, save_path)

        # Should be readable as text
        content = save_path.read_text()
        assert "[fitting]" in content or "fitting" in content


class TestDefaultConfigGeneration:
    """Tests for default config generation."""

    def test_generate_default_config_format(self):
        """Generated config should be valid TOML."""
        content = generate_default_config()
        assert isinstance(content, str)
        assert "[fitting]" in content
        assert "[clustering]" in content
        assert "[output]" in content

    def test_generated_config_has_comments(self):
        """Generated config should include helpful comments."""
        content = generate_default_config()
        assert "#" in content  # Has comments
        assert "auto" in content
        assert "gaussian" in content

    def test_generated_config_is_loadable(self, tmp_path):
        """Generated config should be loadable."""
        content = generate_default_config()
        config_path = tmp_path / "generated.toml"
        config_path.write_text(content)

        config = load_config(config_path)
        assert config.fitting.lineshape == "auto"
        assert config.fitting.refine_iterations == 1
