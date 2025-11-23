"""Lineshapes package for NMR peak fitting.

This package provides:
- Numba-accelerated lineshape functions (functions module)
- Lineshape model classes (models module)
- Shape registration system (registry module)

All functions are Numba-accelerated with production-level optimizations.
Numba is a required dependency.
"""

# Import registry FIRST (before models, as models use register_shape decorator)
# Import functions for direct access
from peakfit.lineshapes import functions
from peakfit.lineshapes.functions import (
    calculate_lstsq_amplitude,
    compute_all_gaussian_shapes,
    compute_all_lorentzian_shapes,
    compute_all_no_apod_shapes,
    compute_all_pvoigt_shapes,
    compute_all_sp1_shapes,
    compute_all_sp2_shapes,
    compute_ata_symmetric,
    evaluate_apod_shape,
    evaluate_apod_shape_no_apod,
    evaluate_apod_shape_sp1,
    evaluate_apod_shape_sp2,
    gaussian,
    lorentzian,
    no_apod,
    pvoigt,
    sp1,
    sp2,
    warm_numba_cache,
)

# Import models LAST (this will populate SHAPES via decorators)
from peakfit.lineshapes.models import (
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
from peakfit.lineshapes.registry import SHAPES, Shape, get_shape, list_shapes, register_shape

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
    "compute_all_gaussian_shapes",
    "compute_all_lorentzian_shapes",
    "compute_all_no_apod_shapes",
    "compute_all_pvoigt_shapes",
    "compute_all_sp1_shapes",
    "compute_all_sp2_shapes",
    "compute_ata_symmetric",
    "evaluate_apod_shape",
    "evaluate_apod_shape_no_apod",
    "evaluate_apod_shape_sp1",
    "evaluate_apod_shape_sp2",
    "warm_numba_cache",
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
