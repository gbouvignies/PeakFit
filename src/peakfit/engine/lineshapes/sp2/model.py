"""SP2 lineshape model.

SP2 uses squared sine-bell apodization with phase correction.
Parameters: cs (chemical shift), lw (linewidth), phase, and optionally j (coupling).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import numpy.typing as npt

from peakfit.engine.lineshapes.doublet import doublet_kernel
from peakfit.engine.lineshapes.templates import PhasedDoubletBase, PhasedSingletBase
from peakfit.engine.lineshapes.utils import (
    apply_phase,
    estimate_cs_bounds_ppm,
    get_apodization_state,
    require_grid,
)
from peakfit.engine.types import ParamSpec

from .kernel import kernel, kernel_with_derivs, make_state

if TYPE_CHECKING:
    from peakfit.engine.lineshapes.utils import LineshapeContext
    from peakfit.shared.typing import FloatArray


# =============================================================================
# Constants
# =============================================================================

NAME = "sp2"
NAME_DOUBLET = "sp2_doublet"
PARAM_NAMES: tuple[str, ...] = ("cs", "lw", "phase")
PARAM_NAMES_DOUBLET: tuple[str, ...] = ("cs", "lw", "j", "phase")

_DEFAULT_PHASE = np.array([0.0], dtype=np.float64)
_STATE_KEY = "sp2_state"


# =============================================================================
# Singlet parameter specs
# =============================================================================


def param_specs(
    center: float,
    context: LineshapeContext | None = None,
) -> tuple[ParamSpec, ...]:
    """Return parameter specs for SP2 singlet."""
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
    """Return parameter bounds for SP2 singlet."""
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
    """Evaluate SP2 singlet on a grid."""
    if context is None:
        msg = "SP2 function requires LineshapeContext."
        raise ValueError(msg)

    state = get_apodization_state(context, state_key=_STATE_KEY, shape="SP2", make_state=make_state)
    grid = require_grid(context, shape="SP2 singlet")

    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    cs_arr = np.atleast_1d(np.asarray(cs, dtype=np.float64))
    lw_arr = np.atleast_1d(np.asarray(lw, dtype=np.float64))
    phase_arr = np.atleast_1d(np.asarray(phase, dtype=np.float64))

    dw_hz, sign = grid.compute_offsets(x_arr, cs_arr)
    z_values = kernel(dw_hz, lw_arr, state)
    return sign * apply_phase(z_values, phase_arr)


class SP2(PhasedSingletBase):
    """SP2 singlet lineshape with squared sine-bell apodization."""

    shape_name: ClassVar[str] = NAME
    param_specs = staticmethod(param_specs)
    kernel: ClassVar = staticmethod(kernel)
    kernel_with_derivs: ClassVar = staticmethod(kernel_with_derivs)
    make_state: ClassVar = staticmethod(make_state)


# =============================================================================
# Doublet parameter specs
# =============================================================================


def param_specs_doublet(
    center: float,
    context: LineshapeContext | None = None,
) -> tuple[ParamSpec, ...]:
    """Return parameter specs for SP2 doublet."""
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
    """Return parameter bounds for SP2 doublet."""
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
    """Evaluate SP2 doublet on a grid."""
    if context is None:
        msg = "SP2 function requires LineshapeContext."
        raise ValueError(msg)

    state = get_apodization_state(context, state_key=_STATE_KEY, shape="SP2", make_state=make_state)
    grid = require_grid(context, shape="SP2 doublet")

    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    cs_arr = np.atleast_1d(np.asarray(cs, dtype=np.float64))
    lw_arr = np.atleast_1d(np.asarray(lw, dtype=np.float64))
    j_arr = np.atleast_1d(np.asarray(j, dtype=np.float64))
    phase_arr = np.atleast_1d(np.asarray(phase, dtype=np.float64))

    z_values = doublet_kernel(
        x_arr, cs_arr, j_arr, grid, kernel=kernel, kernel_args=(lw_arr, state)
    )
    return apply_phase(z_values, phase_arr)


class SP2Doublet(PhasedDoubletBase):
    """SP2 doublet lineshape with squared sine-bell apodization."""

    shape_name: ClassVar[str] = NAME_DOUBLET
    param_specs = staticmethod(param_specs_doublet)
    kernel: ClassVar = staticmethod(kernel)
    kernel_with_derivs: ClassVar = staticmethod(kernel_with_derivs)
    make_state: ClassVar = staticmethod(make_state)


__all__ = [
    "NAME",
    "NAME_DOUBLET",
    "PARAM_NAMES",
    "PARAM_NAMES_DOUBLET",
    "SP2",
    "SP2Doublet",
    "bounds",
    "bounds_doublet",
    "function",
    "function_doublet",
    "param_specs",
    "param_specs_doublet",
]
