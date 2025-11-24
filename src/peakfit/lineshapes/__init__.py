"""Lineshapes package for NMR peak fitting.

This package provides:
- Numba-accelerated lineshape functions (functions module)
- Lineshape model classes (models module)
- Shape registration system (registry module)

All functions are Numba-accelerated with production-level optimizations.
Numba is a required dependency.
"""

# Import functions for direct access (no circular dependency)
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

# Import registry and all model classes
from peakfit.lineshapes.models import (
    SHAPES,
    SP1,
    SP2,
    ApodShape,
    BaseShape,
    Gaussian,
    Lorentzian,
    NoApod,
    PeakShape,
    PseudoVoigt,
    Shape,
    get_shape,
    list_shapes,
    register_shape,
)

__all__ = [
    # Registry
    "SHAPES",
    "SP1",
    "SP2",
    "ApodShape",
    # Models
    "BaseShape",
    "Gaussian",
    "Lorentzian",
    "NoApod",
    "PeakShape",
    "PseudoVoigt",
    "Shape",
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
    # Functions
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
    "warm_numba_cache",
]
