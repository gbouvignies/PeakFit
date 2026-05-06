"""Lineshape models.

This module provides various lineshape models for fitting spectral peaks.
All models utilize a unified protocol for evaluation and parameter management.
"""

from peakfit.engine.lineshapes.factory import LineshapeFactory
from peakfit.engine.lineshapes.gaussian import Gaussian
from peakfit.engine.lineshapes.lorentzian import Lorentzian
from peakfit.engine.lineshapes.no_apod import NoApod
from peakfit.engine.lineshapes.protocol import LineshapeContext, LineshapeProtocol
from peakfit.engine.lineshapes.pvoigt import PseudoVoigt
from peakfit.engine.lineshapes.registry import (
    get_lineshape,
    list_lineshapes,
    register_lineshape,
    register_shape,
)
from peakfit.engine.lineshapes.sp1 import SP1
from peakfit.engine.lineshapes.sp2 import SP2
from peakfit.engine.types import (
    ClusterParameters,
    KernelResult,
    LineshapeResult,
    ParamSpec,
    Shape,
)

__all__ = [
    "SP1",
    "SP2",
    "ClusterParameters",
    "Gaussian",
    "KernelResult",
    "LineshapeContext",
    "LineshapeFactory",
    "LineshapeProtocol",
    "LineshapeResult",
    "Lorentzian",
    "NoApod",
    "ParamSpec",
    "PseudoVoigt",
    "Shape",
    "get_lineshape",
    "list_lineshapes",
    "register_lineshape",
    "register_shape",
]
