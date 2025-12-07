"""Lineshape models package.

This package contains various lineshape models for fitting NMR peaks,
organized into modules by shape type.
"""

# Import convenience functions for simple lineshape evaluation
from peakfit.core.lineshapes.apodization import ApodizationEvaluator, ApodShape
from peakfit.core.lineshapes.base import BaseEvaluator, BaseShape, PeakShape
from peakfit.core.lineshapes.factory import LineshapeFactory
from peakfit.core.lineshapes.gaussian import Gaussian, GaussianEvaluator, gaussian
from peakfit.core.lineshapes.lorentzian import Lorentzian, LorentzianEvaluator, lorentzian
from peakfit.core.lineshapes.no_apod import NoApod, NoApodEvaluator, no_apod
from peakfit.core.lineshapes.pvoigt import PseudoVoigt, PseudoVoigtEvaluator, pvoigt
from peakfit.core.lineshapes.sp1 import (
    SP1Evaluator,
    create_sp1_shape,
    make_sp1_evaluator,
)
from peakfit.core.lineshapes.sp2 import (
    SP2Evaluator,
    create_sp2_shape,
    make_sp2_evaluator,
)
from peakfit.core.lineshapes.utils import CachedResult

# Alias factory functions to class names for backward compatibility
SP1 = create_sp1_shape
SP2 = create_sp2_shape

__all__ = [
    "SP1",
    "SP2",
    "ApodShape",
    "ApodizationEvaluator",
    "BaseEvaluator",
    "BaseShape",
    "CachedResult",
    "Gaussian",
    "GaussianEvaluator",
    "LineshapeFactory",
    "Lorentzian",
    "LorentzianEvaluator",
    "NoApod",
    "NoApodEvaluator",
    "PeakShape",
    "PseudoVoigt",
    "PseudoVoigtEvaluator",
    "SP1Evaluator",
    "SP2Evaluator",
    "gaussian",
    "lorentzian",
    "make_sp1_evaluator",
    "make_sp2_evaluator",
    "no_apod",
    "pvoigt",
]
