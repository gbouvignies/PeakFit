"""PeakFit - lineshape fitting for pseudo-ND NMR spectra."""

import contextlib
from importlib import metadata

__version__ = "0+unknown"

with contextlib.suppress(metadata.PackageNotFoundError):
    __version__ = metadata.version(__name__)

__all__ = ["__version__"]
