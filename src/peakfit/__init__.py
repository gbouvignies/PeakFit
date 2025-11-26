"""PeakFit - Lineshape fitting for pseudo-3D NMR spectra.

Public API:
    - FitService: Main fitting service
    - PlotService: Visualization generation

Configuration:
    - PeakFitConfig: Main configuration object
    - FitConfig, ClusterConfig, OutputConfig: Sub-configurations

Domain Objects:
    - FittingState: Complete fitting state
"""

import contextlib
from importlib import metadata

__version__ = "0.3.0"

with contextlib.suppress(metadata.PackageNotFoundError):
    __version__ = metadata.version(__name__)

# Services (primary API)
# Configuration
from peakfit.core.domain.config import ClusterConfig, FitConfig, OutputConfig, PeakFitConfig

# Domain objects (read-only access)
from peakfit.core.domain.state import FittingState
from peakfit.services import FitResult, FitService, PlotOutput, PlotService

__all__ = [
    # Version
    "__version__",
    # Services
    "FitService",
    "FitResult",
    "PlotService",
    "PlotOutput",
    # Configuration
    "PeakFitConfig",
    "FitConfig",
    "ClusterConfig",
    "OutputConfig",
    # Domain
    "FittingState",
]
