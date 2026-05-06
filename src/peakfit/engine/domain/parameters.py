"""Parameter management for NMR peak fitting."""

from peakfit.engine.domain.param_id import PSEUDO_AXIS, ParameterId
from peakfit.engine.domain.param_map import ParameterMap
from peakfit.engine.domain.params_scalar import Parameter, Parameters
from peakfit.engine.domain.params_vector import FitParameters, FitParametersIndex

__all__ = [
    "PSEUDO_AXIS",
    "FitParameters",
    "FitParametersIndex",
    "Parameter",
    "ParameterId",
    "ParameterMap",
    "Parameters",
]
