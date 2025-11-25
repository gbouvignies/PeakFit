"""Lineshapes package for NMR peak fitting.

This package provides:
- Pure NumPy lineshape functions (functions module)
- Lineshape model classes (models module)
- Shape registration system (registry module)
"""

# Import functions for direct access
from peakfit.core.lineshapes import functions
from peakfit.core.lineshapes.functions import gaussian, lorentzian, no_apod, pvoigt, sp1, sp2

# Import models (this will populate SHAPES via decorators)
from peakfit.core.lineshapes.models import (
    SP1,
    SP2,
    ApodShape,
    BaseShape,
    Gaussian,
    Lorentzian,
    NoApod,
    PeakShape,
    PseudoVoigt,
)

# Import registry before models (models uses register_shape)
from peakfit.core.lineshapes.registry import SHAPES, Shape, get_shape, list_shapes, register_shape

# (registry already imported above)

__all__ = [
    # Functions
    "functions",
    "gaussian",
    "lorentzian",
    "pvoigt",
    "no_apod",
    "sp1",
    "sp2",
    # Models
    "BaseShape",
    "PeakShape",
    "ApodShape",
    "Gaussian",
    "Lorentzian",
    "PseudoVoigt",
    "NoApod",
    "SP1",
    "SP2",
    # Registry
    "SHAPES",
    "Shape",
    "register_shape",
    "get_shape",
    "list_shapes",
]
