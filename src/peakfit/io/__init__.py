"""I/O module for PeakFit.

Handles file operations including:
- Configuration file loading/saving (TOML)
- Result file output
- Fitting state persistence
"""

from peakfit.io.config import generate_default_config, load_config, save_config
from peakfit.io.readers import ResultsLoader
from peakfit.io.state import default_state_path, save_state
from peakfit.io.writers.orchestrator import ResultsWriter

__all__ = [
    "ResultsLoader",
    "ResultsWriter",
    "default_state_path",
    "generate_default_config",
    "load_config",
    "save_config",
    "save_state",
]
