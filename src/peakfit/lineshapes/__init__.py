"""Lineshapes package for NMR peak fitting.

This package provides:
- Pure NumPy lineshape functions (functions module)
- Lineshape model classes (models module)
- Shape registration system (registry module)
"""

# Import functions for direct access
from peakfit.lineshapes import functions
from peakfit.lineshapes.functions import (
    calculate_lstsq_amplitude,
    gaussian,
    lorentzian,
    no_apod,
    pvoigt,
    sp1,
    sp2,
)

# Import registry before models (models uses register_shape)
from peakfit.lineshapes.registry import SHAPES, get_shape, list_shapes, register_shape

# Import models (this will populate SHAPES via decorators)
from peakfit.lineshapes.models import (
    ApodShape,
    BaseShape,
    Gaussian,
    Lorentzian,
    NoApod,
    PeakShape,
    PseudoVoigt,
    SP1,
    SP2,
)

__all__ = [
    # Functions
    "functions",
    "gaussian",
    "lorentzian",
    "pvoigt",
    "no_apod",
    "sp1",
    "sp2",
    "calculate_lstsq_amplitude",
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
    "register_shape",
    "get_shape",
    "list_shapes",
]
