"""Configuration file loading and saving."""

import tomllib
from pathlib import Path

import tomli_w

from peakfit.engine.domain.config import PeakFitConfig
from peakfit.shared.paths import format_path


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


def save_config(config: PeakFitConfig, path: Path) -> None:
    """Save configuration to a TOML file.

    Args:
        config: Configuration object to save.
        path: Path where to save the TOML file.
    """
    data = config.model_dump(mode="json", exclude_none=True)

    type TomlScalar = str | int | float | bool
    type TomlValue = TomlScalar | list["TomlValue"] | dict[str, "TomlValue"]

    def to_toml_value(obj: object) -> TomlValue:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, list):
            return [to_toml_value(v) for v in obj]
        if isinstance(obj, dict):
            out: dict[str, TomlValue] = {}
            for k, v in obj.items():
                if not isinstance(k, str):
                    msg = f"TOML keys must be strings (got {type(k).__name__})"
                    raise TypeError(msg)
                out[k] = to_toml_value(v)
            return out
        msg = f"Unsupported value type for TOML serialization: {type(obj).__name__}"
        raise TypeError(msg)

    normalized = to_toml_value(data)
    if not isinstance(normalized, dict):
        msg = "Serialized configuration must be a mapping"
        raise TypeError(msg)

    with path.open("wb") as f:
        tomli_w.dump(normalized, f)


def generate_default_config() -> str:
    """Generate a default configuration file as a string.

    Returns:
    -------
        str: TOML-formatted default configuration.
    """
    return """# PeakFit Configuration File
# Generated automatically - edit as needed

[fitting]
lineshape = "auto"  # auto, gaussian, lorentzian, pvoigt, sp1, sp2, no_apod
refine_iterations = 1
fit_phase = []  # List of dimensions to fit phase for, e.g. ["F2"]
max_iterations = 1000
tolerance = 1e-8

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
