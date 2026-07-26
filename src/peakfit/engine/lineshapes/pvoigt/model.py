"""Pseudo-Voigt lineshape model.

This module provides Pseudo-Voigt singlet and doublet lineshapes.
The Pseudo-Voigt is a weighted sum of Gaussian and Lorentzian profiles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import numpy.typing as npt

from peakfit.engine.lineshapes.doublet import doublet_kernel
from peakfit.engine.lineshapes.templates import SimpleDoubletBase, SimpleSingletBase
from peakfit.engine.lineshapes.utils import estimate_cs_bounds_ppm, require_grid
from peakfit.engine.types import ClusterParameters, ParamSpec

from .kernel import kernel, kernel_with_derivs

if TYPE_CHECKING:
    from peakfit.engine.lineshapes.utils import LineshapeContext
    from peakfit.shared.typing import FloatArray


# =============================================================================
# Lineshape metadata
# =============================================================================

NAME = "pvoigt"
PARAM_NAMES: tuple[str, ...] = ("cs", "lw", "eta")


def param_specs(
    center: float,
    context: LineshapeContext | None = None,
) -> tuple[ParamSpec, ...]:
    """Return default parameter specs for a Pseudo-Voigt singlet."""
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
            name="eta",
            default=0.5,
            min_val=-1.0,
            max_val=1.0,
            unit="",
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
    eta: npt.ArrayLike,
    *,
    context: LineshapeContext | None = None,
) -> FloatArray:
    """Evaluate Pseudo-Voigt singlet on a grid.

    Parameters must be arrays with one value per peak.
    """
    grid = require_grid(context, shape="Pseudo-Voigt singlet")
    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    cs_arr = np.atleast_1d(np.asarray(cs, dtype=np.float64))
    lw_arr = np.atleast_1d(np.asarray(lw, dtype=np.float64))
    eta_arr = np.atleast_1d(np.asarray(eta, dtype=np.float64))
    dw_hz, sign = grid.compute_offsets(x_arr, cs_arr)
    return sign * kernel(dw_hz, lw_arr[None, :], eta_arr)


# =============================================================================
# Singlet class
# =============================================================================


class PseudoVoigt(SimpleSingletBase):
    """Pseudo-Voigt singlet: eta*Lorentzian + (1-eta)*Gaussian."""

    shape_name: ClassVar[str] = NAME
    param_specs = staticmethod(param_specs)
    kernel: ClassVar = staticmethod(kernel)
    kernel_with_derivs: ClassVar = staticmethod(kernel_with_derivs)

    def _get_extra_params(self, cluster_params: ClusterParameters) -> tuple[Any, ...]:
        """Extract eta parameter."""
        eta = cluster_params.extras.get("eta", np.full(cluster_params.n_peaks, 0.5))
        return (eta,)

    def _process_extra_derivs(
        self,
        derivs: dict[str, FloatArray],
        raw_derivs: dict[str, FloatArray],
        sign: FloatArray,
    ) -> dict[str, FloatArray]:
        """Add eta derivative."""
        if "eta" in raw_derivs:
            derivs["eta"] = sign * raw_derivs["eta"]
        return derivs


# =============================================================================
# Doublet
# =============================================================================

NAME_DOUBLET = "pvoigt_doublet"
PARAM_NAMES_DOUBLET: tuple[str, ...] = ("cs", "lw", "eta", "j")


def param_specs_doublet(
    center: float,
    context: LineshapeContext | None = None,
) -> tuple[ParamSpec, ...]:
    """Return default parameter specs for a Pseudo-Voigt doublet."""
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
            name="eta",
            default=0.5,
            min_val=-1.0,
            max_val=1.0,
            unit="",
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
    eta: npt.ArrayLike,
    j: npt.ArrayLike,
    *,
    context: LineshapeContext | None = None,
) -> FloatArray:
    """Evaluate Pseudo-Voigt doublet on a grid.

    Parameters must be arrays with one value per peak.
    """
    grid = require_grid(context, shape="Pseudo-Voigt doublet")
    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    cs_arr = np.atleast_1d(np.asarray(cs, dtype=np.float64))
    lw_arr = np.atleast_1d(np.asarray(lw, dtype=np.float64))
    eta_arr = np.atleast_1d(np.asarray(eta, dtype=np.float64))
    j_arr = np.atleast_1d(np.asarray(j, dtype=np.float64))
    return doublet_kernel(
        x_arr,
        cs_arr,
        j_arr,
        grid,
        kernel=kernel,
        kernel_args=(lw_arr[None, :], eta_arr),
    )


class PseudoVoigtDoublet(SimpleDoubletBase):
    """Pseudo-Voigt doublet lineshape with J-coupling."""

    shape_name: ClassVar[str] = NAME_DOUBLET
    param_specs = staticmethod(param_specs_doublet)
    kernel: ClassVar = staticmethod(kernel)
    kernel_with_derivs: ClassVar = staticmethod(kernel_with_derivs)

    def _get_extra_params(self, cluster_params: ClusterParameters) -> tuple[Any, ...]:
        """Extract eta parameter."""
        eta = cluster_params.extras.get("eta", np.full(cluster_params.n_peaks, 0.5))
        return (eta,)

    def _process_extra_derivs(
        self,
        derivs: dict[str, FloatArray],
        raw_derivs: dict[str, FloatArray],
    ) -> dict[str, FloatArray]:
        """Add eta derivative."""
        if "eta" in raw_derivs:
            derivs["eta"] = raw_derivs["eta"]
        return derivs


__all__ = [
    "NAME",
    "NAME_DOUBLET",
    "PARAM_NAMES",
    "PARAM_NAMES_DOUBLET",
    "PseudoVoigt",
    "PseudoVoigtDoublet",
    "bounds",
    "bounds_doublet",
    "function",
    "function_doublet",
    "param_specs",
    "param_specs_doublet",
]
