"""IO module for PeakFit - handles file I/O operations.

Note:
- Configuration I/O (load_config, save_config) has been moved to peakfit.models
- Output file writing (write_profiles, write_shifts) has been moved to peakfit.fitting.output
These functions are now located with the modules they're tightly coupled with.
"""

# Re-export for backward compatibility
from peakfit.models import load_config, save_config

__all__ = ["load_config", "save_config"]
