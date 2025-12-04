"""Lineshapes package for NMR peak fitting.

This package provides:
- Pure NumPy lineshape functions (functions module)
- Lineshape model classes (models module)
- Shape registration system (registry module)
"""

# Import functions for direct access
from peakfit.core.lineshapes import functions
from peakfit.core.lineshapes.factory import LineshapeFactory

# Import convenience functions for simple lineshape evaluation
from peakfit.core.lineshapes.functions import gaussian, lorentzian, pvoigt

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
    "lorentzian",
    "pvoigt",
    "register_shape",
]
