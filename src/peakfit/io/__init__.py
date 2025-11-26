"""I/O module for PeakFit.

Handles file operations including:
- Configuration file loading/saving (TOML)
- Result file output
- Fitting state persistence
"""

from peakfit.io.config import generate_default_config, load_config, save_config
from peakfit.io.output import write_profiles, write_shifts
from peakfit.io.state import StateRepository

__all__ = [
    "StateRepository",
    "generate_default_config",
    "load_config",
    "save_config",
    "write_profiles",
    "write_shifts",
]
