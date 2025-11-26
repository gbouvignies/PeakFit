"""Lineshapes package for NMR peak fitting.

This package provides:
- Pure NumPy lineshape functions (functions module)
- Lineshape model classes (models module)
- Shape registration system (registry module)
"""

# Import functions for direct access
from peakfit.core.lineshapes import functions
from peakfit.core.lineshapes.factory import LineshapeFactory
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
    "SHAPES",
    "SP1",
    "SP2",
    "ApodShape",
    "BaseShape",
    "Gaussian",
    "LineshapeFactory",
    "Lorentzian",
    "NoApod",
    "PeakShape",
    "PseudoVoigt",
    "Shape",
    "functions",
    "gaussian",
    "get_shape",
    "list_shapes",
    "lorentzian",
    "no_apod",
    "pvoigt",
    "register_shape",
    "sp1",
    "sp2",
]
