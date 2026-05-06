"""Base classes for lineshape plugins.

This module provides template base classes for lineshape implementations.
There are two families:

1. **Simple lineshapes** (Gaussian, Lorentzian, PVoigt):
   - Real-valued output
   - Kernel receives `lw` in Hz (FWHM)
   - Kernel returns `lw` derivative (already in Hz units)

2. **Phased lineshapes** (NoApod, SP1, SP2):
   - Complex-valued with phase correction
   - Kernel receives `lw` in Hz (FWHM)
   - Kernel returns `lw` derivative (already in Hz units)
   - May have apodization state

In all cases, `lw` represents the FWHM (Full Width at Half Maximum) in Hz.
For apodized lineshapes, this is the FWHM as if there were no apodization.

Subclasses only need to define:
- `shape_name`: The lineshape name for registry
- `param_specs`: Static method returning parameter specs
- `kernel` / `kernel_with_derivs`: The mathematical kernel function
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np

from peakfit.engine.lineshapes.base import ShapeBase
from peakfit.engine.lineshapes.doublet import doublet_kernel, doublet_kernel_with_derivs
from peakfit.engine.lineshapes.utils import apply_phase, apply_phase_with_derivs
from peakfit.engine.types import ClusterParameters, LineshapeResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from peakfit.engine.domain.config import FitConfig
    from peakfit.engine.domain.spectrum import Spectra
    from peakfit.shared.typing import ComplexArray, FloatArray

    # Type aliases for kernel functions
    SimpleKernelFn = Callable[..., FloatArray]
    SimpleKernelWithDerivsFn = Callable[..., tuple[FloatArray, dict[str, FloatArray]]]

    PhasedKernelFn = Callable[..., ComplexArray]
    PhasedKernelWithDerivsFn = Callable[..., tuple[ComplexArray, dict[str, ComplexArray]]]

    StateFactory = Callable[[float, float, float], dict[str, complex | float]]


# =============================================================================
# Simple lineshapes (real-valued, no phase)
# =============================================================================


class SimpleSingletBase(ShapeBase):
    """Base class for simple real-valued singlet lineshapes.

    Used by: Gaussian, Lorentzian, PVoigt

    Kernels receive `(dw, lw, *extra_params)` where:
    - dw: frequency offsets in Hz
    - lw: linewidth FWHM in Hz

    Derivatives should use key "lw" and "dw", both in Hz units.
    """

    kernel: ClassVar[SimpleKernelFn]
    kernel_with_derivs: ClassVar[SimpleKernelWithDerivsFn]

    def _get_extra_params(self, cluster_params: ClusterParameters) -> tuple[Any, ...]:
        """Override to extract additional kernel parameters (e.g., eta for PVoigt)."""
        return ()

    def _process_extra_derivs(
        self,
        derivs: dict[str, FloatArray],
        raw_derivs: dict[str, FloatArray],
        sign: FloatArray,
    ) -> dict[str, FloatArray]:
        """Override to add derivatives for extra parameters."""
        return derivs

    def evaluate_cluster(
        self,
        x_grid: FloatArray,
        cluster_params: ClusterParameters,
        compute_derivs: bool = False,
    ) -> LineshapeResult:
        """Evaluate lineshape for a cluster of peaks."""
        positions = cluster_params.extras.get("cs", np.full(cluster_params.n_peaks, self.center))
        lw_hz = cluster_params.extras.get("lw", np.full(cluster_params.n_peaks, 25.0))[None, :]

        dw_hz, sign = self._grid.compute_offsets(x_grid, positions)

        extra_params = self._get_extra_params(cluster_params)

        if compute_derivs:
            values, raw_derivs = type(self).kernel_with_derivs(dw_hz, lw_hz, *extra_params)
            values = sign * values

            derivs: dict[str, FloatArray] = {}
            if "dw" in raw_derivs:
                derivs["cs"] = sign * raw_derivs["dw"] * (-self._grid.spec_params.ppm2hz(1.0))
            if "lw" in raw_derivs:
                derivs["lw"] = sign * raw_derivs["lw"]

            derivs = self._process_extra_derivs(derivs, raw_derivs, sign)

            return LineshapeResult(
                values=cast("FloatArray", values.real),
                derivatives={k: cast("FloatArray", v.real) for k, v in derivs.items()},
            )

        values = sign * type(self).kernel(dw_hz, lw_hz, *extra_params)
        return LineshapeResult(values=cast("FloatArray", values.real))


class SimpleDoubletBase(ShapeBase):
    """Base class for simple real-valued doublet lineshapes.

    Used by: Gaussian doublet, Lorentzian doublet, PVoigt doublet

    Same kernel interface as SimpleSingletBase.
    The doublet is constructed as: kernel(dw - J/2) + kernel(dw + J/2)
    """

    kernel: ClassVar[SimpleKernelFn]
    kernel_with_derivs: ClassVar[SimpleKernelWithDerivsFn]

    def _get_extra_params(self, cluster_params: ClusterParameters) -> tuple[Any, ...]:
        """Override to extract additional kernel parameters."""
        return ()

    def _process_extra_derivs(
        self,
        derivs: dict[str, FloatArray],
        raw_derivs: dict[str, FloatArray],
    ) -> dict[str, FloatArray]:
        """Override to add derivatives for extra parameters."""
        return derivs

    def evaluate_cluster(
        self,
        x_grid: FloatArray,
        cluster_params: ClusterParameters,
        compute_derivs: bool = False,
    ) -> LineshapeResult:
        """Evaluate doublet lineshape for a cluster of peaks."""
        positions = cluster_params.extras.get("cs", np.full(cluster_params.n_peaks, self.center))
        lw_hz = cluster_params.extras.get("lw", np.full(cluster_params.n_peaks, 25.0))[None, :]
        j_coupling = cluster_params.extras.get("j", np.zeros(cluster_params.n_peaks))

        extra_params = self._get_extra_params(cluster_params)

        if compute_derivs:
            values, raw_derivs = doublet_kernel_with_derivs(
                x_grid,
                positions,
                j_coupling,
                self._grid,
                kernel_with_derivs=type(self).kernel_with_derivs,
                kernel_args=(lw_hz, *extra_params),
            )

            derivs: dict[str, FloatArray] = {}
            if "dw" in raw_derivs:
                derivs["cs"] = raw_derivs["dw"] * (-self._grid.spec_params.ppm2hz(1.0))
            if "lw" in raw_derivs:
                derivs["lw"] = raw_derivs["lw"]

            derivs = self._process_extra_derivs(derivs, raw_derivs)
            return LineshapeResult(values=values, derivatives=derivs)

        values = doublet_kernel(
            x_grid,
            positions,
            j_coupling,
            self._grid,
            kernel=type(self).kernel,
            kernel_args=(lw_hz, *extra_params),
        )
        return LineshapeResult(values=values)


# =============================================================================
# Phased lineshapes (complex-valued with phase correction)
# =============================================================================


class PhasedSingletBase(ShapeBase):
    """Base class for complex-valued singlet lineshapes with phase.

    Used by: NoApod, SP1, SP2

    Kernels receive `(dw, lw, state)` where:
    - dw: frequency offsets in Hz
    - lw: linewidth FWHM in Hz
    - state: apodization state dict (from make_state factory)

    Derivatives should use key "lw" and "dw", both in Hz units.
    """

    kernel: ClassVar[PhasedKernelFn]
    kernel_with_derivs: ClassVar[PhasedKernelWithDerivsFn]
    make_state: ClassVar[StateFactory]

    def __init__(
        self,
        name: str,
        center: float,
        spectra: Spectra,
        dim: int,
        config: FitConfig,
        aq: float | None = None,
    ) -> None:
        super().__init__(name, center, spectra, dim, config)
        spec = self._grid.spec_params
        self._aq = aq if aq is not None else spec.aq_time
        self._kernel_state = type(self).make_state(self._aq, spec.apodq1, spec.apodq2)

    def evaluate_cluster(
        self,
        x_grid: FloatArray,
        cluster_params: ClusterParameters,
        compute_derivs: bool = False,
    ) -> LineshapeResult:
        """Evaluate lineshape for a cluster of peaks."""
        positions = cluster_params.extras.get("cs", np.full(cluster_params.n_peaks, self.center))
        lw_hz = cluster_params.extras.get("lw", np.full(cluster_params.n_peaks, 20.0))
        phases = cluster_params.extras.get("phase", np.zeros(cluster_params.n_peaks))

        dw_hz, sign = self._grid.compute_offsets(x_grid, positions)

        if compute_derivs:
            z_values, z_derivs = type(self).kernel_with_derivs(dw_hz, lw_hz, self._kernel_state)
            values, derivs = apply_phase_with_derivs(z_values, z_derivs, phases, self._aq)
            values = sign * values

            if "dw" in derivs:
                derivs["cs"] = sign * derivs["dw"] * (-self._grid.spec_params.ppm2hz(1.0))
            if "lw" in derivs:
                derivs["lw"] = sign * derivs["lw"]
            if "phase" in derivs:
                derivs["phase"] = sign * derivs["phase"]

            return LineshapeResult(values=values, derivatives=derivs)

        z_values = type(self).kernel(dw_hz, lw_hz, self._kernel_state)
        values = sign * apply_phase(z_values, phases)
        return LineshapeResult(values=values)


class PhasedDoubletBase(ShapeBase):
    """Base class for complex-valued doublet lineshapes with phase.

    Used by: NoApod doublet, SP1 doublet, SP2 doublet

    Same kernel interface as PhasedSingletBase.
    """

    kernel: ClassVar[PhasedKernelFn]
    kernel_with_derivs: ClassVar[PhasedKernelWithDerivsFn]
    make_state: ClassVar[StateFactory]

    def __init__(
        self,
        name: str,
        center: float,
        spectra: Spectra,
        dim: int,
        config: FitConfig,
        aq: float | None = None,
    ) -> None:
        super().__init__(name, center, spectra, dim, config)
        spec = self._grid.spec_params
        self._aq = aq if aq is not None else spec.aq_time
        self._kernel_state = type(self).make_state(self._aq, spec.apodq1, spec.apodq2)

    def evaluate_cluster(
        self,
        x_grid: FloatArray,
        cluster_params: ClusterParameters,
        compute_derivs: bool = False,
    ) -> LineshapeResult:
        """Evaluate doublet lineshape for a cluster of peaks."""
        positions = cluster_params.extras.get("cs", np.full(cluster_params.n_peaks, self.center))
        lw_hz = cluster_params.extras.get("lw", np.full(cluster_params.n_peaks, 20.0))
        j_coupling = cluster_params.extras.get("j", np.zeros(cluster_params.n_peaks))
        phases = cluster_params.extras.get("phase", np.zeros(cluster_params.n_peaks))

        if compute_derivs:
            z_values, z_derivs = doublet_kernel_with_derivs(
                x_grid,
                positions,
                j_coupling,
                self._grid,
                kernel_with_derivs=type(self).kernel_with_derivs,
                kernel_args=(lw_hz, self._kernel_state),
            )
            values, derivs = apply_phase_with_derivs(z_values, z_derivs, phases, self._aq)

            if "dw" in derivs:
                derivs["cs"] = derivs["dw"] * (-self._grid.spec_params.ppm2hz(1.0))
            if "lw" in derivs:
                derivs["lw"] = derivs["lw"]

            return LineshapeResult(values=values, derivatives=derivs)

        z_values = doublet_kernel(
            x_grid,
            positions,
            j_coupling,
            self._grid,
            kernel=type(self).kernel,
            kernel_args=(lw_hz, self._kernel_state),
        )
        values = apply_phase(z_values, phases)
        return LineshapeResult(values=values)


__all__ = [
    "PhasedDoubletBase",
    "PhasedSingletBase",
    "SimpleDoubletBase",
    "SimpleSingletBase",
]
