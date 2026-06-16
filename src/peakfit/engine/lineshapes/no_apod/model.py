"""NoApod lineshape model.

NoApod (no apodization) uses simple exponential decay with phase correction.
Parameters: cs (chemical shift), lw (linewidth), phase, and optionally j (coupling).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import numpy.typing as npt

from peakfit.engine.lineshapes.doublet import doublet_kernel
from peakfit.engine.lineshapes.templates import PhasedDoubletBase, PhasedSingletBase
from peakfit.engine.lineshapes.utils import (
    apply_phase,
    estimate_cs_bounds_ppm,
    require_grid,
)
from peakfit.engine.types import ParamSpec

from .kernel import kernel, kernel_with_derivs

if TYPE_CHECKING:
    from peakfit.engine.lineshapes.utils import LineshapeContext
    from peakfit.shared.typing import FloatArray


# =============================================================================
# Constants
# =============================================================================

NAME = "no_apod"
NAME_DOUBLET = "no_apod_doublet"
PARAM_NAMES: tuple[str, ...] = ("cs", "lw", "phase")
PARAM_NAMES_DOUBLET: tuple[str, ...] = ("cs", "lw", "j", "phase")

_DEFAULT_PHASE = np.array([0.0], dtype=np.float64)


def make_state(aq: float, apodq1: float, apodq2: float) -> dict[str, complex | float]:
    """Create kernel state for NoApod (just acquisition time)."""
    return {"aq": aq}


def _aq_from_context(context: LineshapeContext) -> float:
    """Extract acquisition time from context."""
    if "aq" in context.extras:
        return float(context.extras["aq"])
    if context.grid is not None:
        return float(context.grid.spec_params.aq_time)
    msg = "NoApod requires acquisition time via context.extras['aq'] or grid."
    raise ValueError(msg)


# =============================================================================
# Singlet parameter specs
# =============================================================================


def param_specs(
    center: float,
    context: LineshapeContext | None = None,
) -> tuple[ParamSpec, ...]:
    """Return parameter specs for NoApod singlet."""
    pos_bounds_ppm = estimate_cs_bounds_ppm(context)
    return (
        ParamSpec(
            name="cs",
            default=center,
            min_val=center - pos_bounds_ppm,
            max_val=center + pos_bounds_ppm,
            unit="ppm",
        ),
        ParamSpec(name="lw", default=20.0, min_val=0.1, max_val=200.0, unit="Hz"),
        ParamSpec(name="phase", default=0.0, min_val=-15.0, max_val=15.0, unit="deg"),
    )


def bounds(
    center: float, context: LineshapeContext | None = None
) -> tuple[tuple[float, float], ...]:
    """Return parameter bounds for NoApod singlet."""
    specs = param_specs(center, context)
    return tuple((s.min_val, s.max_val) for s in specs)


def function(
    x: npt.ArrayLike,
    cs: npt.ArrayLike,
    lw: npt.ArrayLike,
    phase: npt.ArrayLike = _DEFAULT_PHASE,
    *,
    context: LineshapeContext | None = None,
) -> FloatArray:
    """Evaluate NoApod singlet on a grid."""
    if context is None:
        msg = "NoApod function requires LineshapeContext."
        raise ValueError(msg)

    aq = _aq_from_context(context)
    state = make_state(aq, 0.0, 0.0)  # NoApod ignores apodq1/apodq2
    grid = require_grid(context, shape="NoApod singlet")

    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    cs_arr = np.atleast_1d(np.asarray(cs, dtype=np.float64))
    lw_arr = np.atleast_1d(np.asarray(lw, dtype=np.float64))
    phase_arr = np.atleast_1d(np.asarray(phase, dtype=np.float64))

    dw_hz, sign = grid.compute_offsets(x_arr, cs_arr)
    z_values = kernel(dw_hz, lw_arr, state)
    values = sign * apply_phase(z_values, phase_arr)
    return values


class NoApod(PhasedSingletBase):
    """NoApod singlet lineshape (no apodization, exponential decay)."""

    shape_name: ClassVar[str] = NAME
    param_specs = staticmethod(param_specs)
    kernel: ClassVar = staticmethod(kernel)
    kernel_with_derivs: ClassVar = staticmethod(kernel_with_derivs)
    make_state: ClassVar = staticmethod(make_state)

    def _param_context_extras(self) -> dict[str, Any]:
        return {"aq": self._aq}


# =============================================================================
# Doublet parameter specs
# =============================================================================


def param_specs_doublet(
    center: float,
    context: LineshapeContext | None = None,
) -> tuple[ParamSpec, ...]:
    """Return parameter specs for NoApod doublet."""
    pos_bounds_ppm = estimate_cs_bounds_ppm(context)
    return (
        ParamSpec(
            name="cs",
            default=center,
            min_val=center - pos_bounds_ppm,
            max_val=center + pos_bounds_ppm,
            unit="ppm",
        ),
        ParamSpec(name="lw", default=20.0, min_val=0.1, max_val=200.0, unit="Hz"),
        ParamSpec(name="j", default=5.0, min_val=1.0, max_val=10.0, unit="Hz"),
        ParamSpec(name="phase", default=0.0, min_val=-15.0, max_val=15.0, unit="deg"),
    )


def bounds_doublet(
    center: float, context: LineshapeContext | None = None
) -> tuple[tuple[float, float], ...]:
    """Return parameter bounds for NoApod doublet."""
    specs = param_specs_doublet(center, context)
    return tuple((s.min_val, s.max_val) for s in specs)


def function_doublet(
    x: npt.ArrayLike,
    cs: npt.ArrayLike,
    lw: npt.ArrayLike,
    j: npt.ArrayLike,
    phase: npt.ArrayLike = _DEFAULT_PHASE,
    *,
    context: LineshapeContext | None = None,
) -> FloatArray:
    """Evaluate NoApod doublet on a grid."""
    if context is None:
        msg = "NoApod function requires LineshapeContext."
        raise ValueError(msg)

    aq = _aq_from_context(context)
    state = make_state(aq, 0.0, 0.0)  # NoApod ignores apodq1/apodq2
    grid = require_grid(context, shape="NoApod doublet")

    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    cs_arr = np.atleast_1d(np.asarray(cs, dtype=np.float64))
    lw_arr = np.atleast_1d(np.asarray(lw, dtype=np.float64))
    j_arr = np.atleast_1d(np.asarray(j, dtype=np.float64))
    phase_arr = np.atleast_1d(np.asarray(phase, dtype=np.float64))

    z_values = doublet_kernel(
        x_arr, cs_arr, j_arr, grid, kernel=kernel, kernel_args=(lw_arr, state)
    )
    values = apply_phase(z_values, phase_arr)
    return values


class NoApodDoublet(PhasedDoubletBase):
    """NoApod doublet lineshape (no apodization, exponential decay)."""

    shape_name: ClassVar[str] = NAME_DOUBLET
    param_specs = staticmethod(param_specs_doublet)
    kernel: ClassVar = staticmethod(kernel)
    kernel_with_derivs: ClassVar = staticmethod(kernel_with_derivs)
    make_state: ClassVar = staticmethod(make_state)

    def _param_context_extras(self) -> dict[str, Any]:
        return {"aq": self._aq}


__all__ = [
    "NAME",
    "NAME_DOUBLET",
    "PARAM_NAMES",
    "PARAM_NAMES_DOUBLET",
    "NoApod",
    "NoApodDoublet",
    "bounds",
    "bounds_doublet",
    "function",
    "function_doublet",
    "param_specs",
    "param_specs_doublet",
]
