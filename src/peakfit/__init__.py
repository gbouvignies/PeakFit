"""PeakFit - lineshape fitting for pseudo-3D NMR spectra."""

import contextlib
from importlib import metadata

__version__ = "0.3.0"

with contextlib.suppress(metadata.PackageNotFoundError):
    __version__ = metadata.version(__name__)

# Services (primary API)
# Configuration
from peakfit.engine.domain.config import ClusterConfig, FitConfig, OutputConfig, PeakFitConfig

# Domain objects (read-only access)
from peakfit.engine.domain.state import FittingState

__all__ = [
    "ClusterConfig",
    "FitConfig",
    "FittingState",
    "OutputConfig",
    "PeakFitConfig",
    "__version__",
]
