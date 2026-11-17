"""Shared base class for lineshape objects."""

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from peakfit.engine.domain.param_id import ParameterId
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.lineshapes.grid import SpectralGrid
from peakfit.engine.types import ClusterParameters, ParamSpec

if TYPE_CHECKING:
    from collections.abc import Callable

    from peakfit.engine.domain.param_map import ParameterMap
    from peakfit.engine.domain.spectrum import AxisLineshape, Spectra
    from peakfit.shared.typing import FloatArray


class ShapeBase:
    """Shared helpers for lineshape implementations."""

    shape_name: ClassVar[str]  # the lineshape family
    doublet: ClassVar[bool] = False
    param_specs: ClassVar[Callable[..., tuple[ParamSpec, ...]]]
    phase_param_name: ClassVar[str] = "phase"

    def __init__(
        self,
        name: str,
        center: float,
        spectra: Spectra,
        dim: int,
        lineshape: AxisLineshape,
    ) -> None:
        self.name = name
        self.center = center
        self._grid = SpectralGrid(spectra, dim)
        self.axis = self._grid.axis_label
        if lineshape.axis != self.axis or lineshape.family != self.shape_name:
            msg = f"{type(self).__name__} on {self.axis} cannot take the {lineshape} model."
            raise ValueError(msg)
        if (lineshape.doublet is not None) != self.doublet:
            msg = f"{type(self).__name__} disagrees with the doublet of {lineshape}."
            raise ValueError(msg)
        self.lineshape = lineshape
        self.cluster_id = 0
        self.param_names: list[str] = []

    @property
    def dim_ctx(self) -> Any:
        """Get the dimension context."""
        return self._grid.dim_ctx

    @property
    def center_i(self) -> int:
        """Get the integer index of the center position."""
        return self._grid.dim_ctx.ppm2pt_i(self.center)

    def __repr__(self) -> str:
        """Return a concise representation of the shape."""
        return (
            f"{type(self).__name__}(name={self.name!r}, center={self.center!r}, axis={self.axis!r})"
        )

    @property
    def description(self) -> str:
        """The family, and ``doublet`` for a doublet: ``lorentzian doublet``."""
        return f"{self.shape_name} doublet" if self.doublet else self.shape_name

    def print(self, params: Parameters) -> str:
        """Return textual representation."""
        del params
        return f"# Shape: {self.name} ({self.description})"

    def get_parameter_spec(self) -> list[ParamSpec]:
        """Get parameter specs for this shape; a doublet's J follows its axis model."""
        return list(self.param_specs(self.center, self._grid, self.lineshape.doublet))

    def _parameter_id(self, label: str) -> ParameterId | None:
        if label == self.phase_param_name:
            if not self.lineshape.phase:
                return None
            return ParameterId(cluster_id=self.cluster_id, axis=self.axis, label=label)
        return ParameterId(peak_name=self.name, axis=self.axis, label=label)

    def parameter_names(self) -> list[str]:
        """Names of this shape's model parameters; an unfitted phase has none."""
        return [
            pid.name
            for spec in self.get_parameter_spec()
            if (pid := self._parameter_id(spec.name)) is not None
        ]

    def create_params(self) -> Parameters:
        """Create lmfit Parameters object for this shape."""
        params = Parameters()

        for spec in self.get_parameter_spec():
            pid = self._parameter_id(spec.name)
            if pid is None:
                continue
            params.add(
                pid,
                value=spec.default,
                min_value=spec.min_val,
                max_value=spec.max_val,
                vary=spec.vary,
                unit=spec.unit,
            )

        self.param_names = list(params.keys())
        return params

    def fix_params(self, params: Parameters) -> None:
        """Fix all parameters of this shape (set vary=False)."""
        for name in self.param_names:
            if name in params:
                params[name].vary = False

    def release_params(self, params: Parameters) -> None:
        """Release all parameters of this shape (set vary=True)."""
        for name in self.param_names:
            if name in params:
                params[name].vary = True

    def _param_name_for_spec(self, spec_name: str, shape: Any) -> str:
        if spec_name == self.phase_param_name:
            pid = ParameterId(
                cluster_id=shape.cluster_id,
                axis=shape.axis,
                label=spec_name,
            )
        else:
            pid = ParameterId(
                peak_name=shape.name,
                axis=shape.axis,
                label=spec_name,
            )
        return pid.name

    def get_cluster_parameters(
        self,
        peaks: Any,
        params: Parameters,
        param_map: ParameterMap | None = None,
    ) -> ClusterParameters:
        """Extract vectorized parameters for a cluster of peaks.

        Every model parameter must be present: a missing one raises rather than
        silently taking its default. A phase that is not fitted on this axis is
        absent from the model, so it has no entry here.
        """
        n_peaks = len(peaks)
        extras: dict[str, FloatArray] = {}
        index_map: dict[str, Any] = {}

        specs = self.get_parameter_spec()

        for spec in specs:
            if self._parameter_id(spec.name) is None:
                continue
            vals = np.zeros(n_peaks, dtype=np.float64)
            idxs = np.full(n_peaks, -1, dtype=np.intp)

            for k, peak in enumerate(peaks):
                shape = next((s for s in peak.shapes if s.axis == self.axis), None)
                if shape is None:
                    msg = f"Peak {peak.name} has no {self.axis} lineshape."
                    raise ValueError(msg)

                p_name = self._param_name_for_spec(spec.name, shape)
                if p_name not in params:
                    msg = f"Model parameter {p_name} is missing from the parameters."
                    raise ValueError(msg)
                vals[k] = params[p_name].value
                if param_map:
                    idxs[k] = param_map.get(p_name, -1)

            extras[spec.name] = vals
            index_map[spec.name] = idxs

        return ClusterParameters(extras, index_map)


__all__ = ["ShapeBase"]
