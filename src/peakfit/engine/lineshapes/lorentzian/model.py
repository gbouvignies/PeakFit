"""Lorentzian lineshape model.

This module provides Lorentzian singlet and doublet lineshapes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import numpy.typing as npt

from peakfit.engine.lineshapes.doublet import doublet_kernel
from peakfit.engine.lineshapes.registry import register_shape
from peakfit.engine.lineshapes.templates import SimpleDoubletBase, SimpleSingletBase
from peakfit.engine.lineshapes.utils import estimate_cs_bounds_ppm, require_grid
from peakfit.engine.types import ParamSpec

from .kernel import kernel, kernel_with_derivs

if TYPE_CHECKING:
    from peakfit.engine.lineshapes.utils import LineshapeContext
    from peakfit.shared.typing import FloatArray


# =============================================================================
# Module-level protocol attributes
# =============================================================================

NAME = "lorentzian"
PARAM_NAMES: tuple[str, ...] = ("cs", "lw")


def param_specs(
    center: float,
    context: LineshapeContext | None = None,
) -> tuple[ParamSpec, ...]:
    """Return default parameter specs for a Lorentzian singlet."""
    pos_bounds_ppm = estimate_cs_bounds_ppm(context)

    return (
        ParamSpec(
            name="cs",
            default=center,
            min_val=center - pos_bounds_ppm,
            max_val=center + pos_bounds_ppm,
            unit="ppm",
        ),
        ParamSpec(
            name="lw",
            default=25.0,
            min_val=0.1,
            max_val=200.0,
            unit="Hz",
        ),
    )


def bounds(
    center: float, context: LineshapeContext | None = None
) -> tuple[tuple[float, float], ...]:
    """Return parameter bounds aligned with PARAM_NAMES."""
    specs = param_specs(center, context)
    return tuple((spec.min_val, spec.max_val) for spec in specs)


def function(
    x: npt.ArrayLike,
    cs: npt.ArrayLike,
    lw: npt.ArrayLike,
    *,
    context: LineshapeContext | None = None,
) -> FloatArray:
    """Evaluate Lorentzian singlet on a grid.

    Parameters must be arrays with one value per peak.
    """
    grid = require_grid(context, shape="Lorentzian singlet")
    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    cs_arr = np.atleast_1d(np.asarray(cs, dtype=np.float64))
    lw_arr = np.atleast_1d(np.asarray(lw, dtype=np.float64))
    dw_hz, sign = grid.compute_offsets(x_arr, cs_arr)
    values = sign * kernel(dw_hz, lw_arr[None, :])
    return values


# =============================================================================
# Singlet class
# =============================================================================


@register_shape(NAME)
class Lorentzian(SimpleSingletBase):
    """Lorentzian singlet lineshape: L(Δω) = 1 / (1 + (2Δω/R)²)."""

    shape_name: ClassVar[str] = NAME
    param_specs = staticmethod(param_specs)
    kernel: ClassVar = staticmethod(kernel)
    kernel_with_derivs: ClassVar = staticmethod(kernel_with_derivs)


# =============================================================================
# Doublet
# =============================================================================

NAME_DOUBLET = "lorentzian_doublet"
PARAM_NAMES_DOUBLET: tuple[str, ...] = ("cs", "lw", "j")


def param_specs_doublet(
    center: float,
    context: LineshapeContext | None = None,
) -> tuple[ParamSpec, ...]:
    """Return default parameter specs for a Lorentzian doublet."""
    pos_bounds_ppm = estimate_cs_bounds_ppm(context)

    return (
        ParamSpec(
            name="cs",
            default=center,
            min_val=center - pos_bounds_ppm,
            max_val=center + pos_bounds_ppm,
            unit="ppm",
        ),
        ParamSpec(
            name="lw",
            default=25.0,
            min_val=0.1,
            max_val=200.0,
            unit="Hz",
        ),
        ParamSpec(
            name="j",
            default=5.0,
            min_val=1.0,
            max_val=10.0,
            unit="Hz",
        ),
    )


def bounds_doublet(
    center: float, context: LineshapeContext | None = None
) -> tuple[tuple[float, float], ...]:
    """Return parameter bounds aligned with PARAM_NAMES_DOUBLET."""
    specs = param_specs_doublet(center, context)
    return tuple((spec.min_val, spec.max_val) for spec in specs)


def function_doublet(
    x: npt.ArrayLike,
    cs: npt.ArrayLike,
    lw: npt.ArrayLike,
    j: npt.ArrayLike,
    *,
    context: LineshapeContext | None = None,
) -> FloatArray:
    """Evaluate Lorentzian doublet on a grid.

    Parameters must be arrays with one value per peak.
    """
    grid = require_grid(context, shape="Lorentzian doublet")
    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    cs_arr = np.atleast_1d(np.asarray(cs, dtype=np.float64))
    lw_arr = np.atleast_1d(np.asarray(lw, dtype=np.float64))
    j_arr = np.atleast_1d(np.asarray(j, dtype=np.float64))
    return doublet_kernel(
        x_arr,
        cs_arr,
        j_arr,
        grid,
        kernel=kernel,
        kernel_args=(lw_arr[None, :],),
    )


@register_shape(NAME_DOUBLET)
class LorentzianDoublet(SimpleDoubletBase):
    """Lorentzian doublet lineshape with J-coupling."""

    shape_name: ClassVar[str] = NAME_DOUBLET
    param_specs = staticmethod(param_specs_doublet)
    kernel: ClassVar = staticmethod(kernel)
    kernel_with_derivs: ClassVar = staticmethod(kernel_with_derivs)


__all__ = [
    "NAME",
    "NAME_DOUBLET",
    "PARAM_NAMES",
    "PARAM_NAMES_DOUBLET",
    "Lorentzian",
    "LorentzianDoublet",
    "bounds",
    "bounds_doublet",
    "function",
    "function_doublet",
    "param_specs",
    "param_specs_doublet",
]
