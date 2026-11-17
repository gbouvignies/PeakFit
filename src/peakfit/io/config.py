"""Configuration file loading and the default template."""

import tomllib
from typing import TYPE_CHECKING

from peakfit.engine.domain.config import PeakFitConfig
from peakfit.shared.paths import format_path

if TYPE_CHECKING:
    from pathlib import Path


def load_config(path: Path) -> PeakFitConfig:
    """Load configuration from a TOML file.

    Args:
        path: Path to the TOML configuration file.

    Returns:
    -------
        PeakFitConfig: Validated configuration object.

    Raises:
    ------
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the configuration is invalid.
    """
    if not path.exists():
        msg = f"Configuration file not found: {format_path(path)}"
        raise FileNotFoundError(msg)

    with path.open("rb") as f:
        data = tomllib.load(f)

    return PeakFitConfig.model_validate(data)


def generate_default_config() -> str:
    """Generate a default configuration file as a string.

    Returns:
    -------
        str: TOML-formatted default configuration.
    """
    return """# PeakFit Configuration File
# Generated automatically - edit as needed

[fitting]
family = "auto"  # every axis: auto, gaussian, lorentzian, pvoigt, sp1, sp2, no_apod
refine_iterations = 1
max_iterations = 1000
tolerance = 1e-8

# Per-axis lineshape model; unset fields keep the defaults above.
# [fitting.axes.F2]
# family = "sp1"
# phase = true                # fit this axis's phase (sp1, sp2, no_apod)
# doublet = { j = 92.0 }      # J start in Hz; bounds default to [j/2, 2j]
# doublet = { j = 92.0, min = 80.0, max = 100.0, vary = false }

[clustering]
contour_factor = 5.0
# contour_level = 1000.0  # Uncomment to set explicit contour level

[output]
directory = "Fits"
formats = ["csv", "txt"]
detailed_evidence = false
save_simulated = false
include_timestamp = true
headless = false

# Optional settings
# noise_level = 100.0  # Uncomment to set manual noise level
# exclude_planes = []  # List of plane indices to exclude
"""
