"""Configuration file loading and saving."""

import tomllib
from pathlib import Path

import tomli_w

from peakfit.models import PeakFitConfig


def load_config(path: Path) -> PeakFitConfig:
    """Load configuration from a TOML file.

    Args:
        path: Path to the TOML configuration file.

    Returns:
        PeakFitConfig: Validated configuration object.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the configuration is invalid.
    """
    if not path.exists():
        msg = f"Configuration file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open("rb") as f:
        data = tomllib.load(f)

    return PeakFitConfig.model_validate(data)


def save_config(config: PeakFitConfig, path: Path) -> None:
    """Save configuration to a TOML file.

    Args:
        config: Configuration object to save.
        path: Path where to save the TOML file.
    """
    data = config.model_dump(mode="json", exclude_none=True)

    # Convert Path objects to strings
    def convert_paths(obj: object) -> object:
        if isinstance(obj, dict):
            return {k: convert_paths(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_paths(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        return obj

    data = convert_paths(data)

    with path.open("wb") as f:
        tomli_w.dump(data, f)


def generate_default_config() -> str:
    """Generate a default configuration file as a string.

    Returns:
        str: TOML-formatted default configuration.
    """
    return """# PeakFit Configuration File
# Generated automatically - edit as needed

[fitting]
lineshape = "auto"  # auto, gaussian, lorentzian, pvoigt, sp1, sp2, no_apod
refine_iterations = 1
fix_positions = false
fit_j_coupling = false
fit_phase_x = false
fit_phase_y = false
max_iterations = 1000
tolerance = 1e-8

[clustering]
contour_factor = 5.0
# contour_level = 1000.0  # Uncomment to set explicit contour level

[output]
directory = "Fits"
formats = ["txt"]
save_simulated = true
save_html_report = true

# Optional settings
# noise_level = 100.0  # Uncomment to set manual noise level
# exclude_planes = []  # List of plane indices to exclude
"""
