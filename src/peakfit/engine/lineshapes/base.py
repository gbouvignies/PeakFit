"""Shared base class for lineshape objects."""

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from peakfit.engine.domain.param_id import ParameterId
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.lineshapes.grid import SpectralGrid
from peakfit.engine.lineshapes.utils import LineshapeContext
from peakfit.engine.types import ClusterParameters, ParamSpec

if TYPE_CHECKING:
    from collections.abc import Callable

    from peakfit.engine.domain.config import FitConfig
    from peakfit.engine.domain.param_map import ParameterMap
    from peakfit.engine.domain.spectrum import Spectra
    from peakfit.shared.typing import FloatArray


class ShapeBase:
    """Shared helpers for lineshape implementations."""

    shape_name: ClassVar[str]
    param_specs: ClassVar[Callable[[float, LineshapeContext | None], tuple[ParamSpec, ...]]]
    phase_param_name: ClassVar[str] = "phase"

    def __init__(
        self,
        name: str,
        center: float,
        spectra: Spectra,
        dim: int,
        config: FitConfig,
    ) -> None:
        self.name = name
        self.center = center
        self._grid = SpectralGrid(spectra, dim)
        self.axis = self._grid.axis_label
        self.cluster_id = 0
        self.param_names: list[str] = []
        self._fit_phase_axes = set(config.fit_phase)

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

    def print(self, params: Parameters) -> str:
        """Return textual representation."""
        del params
        return f"# Shape: {self.name} ({self.shape_name})"

    def _param_context_extras(self) -> dict[str, Any]:
        return {}

    def get_parameter_spec(self) -> list[ParamSpec]:
        """Get parameter specs for this shape."""
        context = LineshapeContext(grid=self._grid, extras=self._param_context_extras())
        return list(self.param_specs(self.center, context))

    def _parameter_id(self, label: str) -> ParameterId | None:
        if label == self.phase_param_name:
            if self.axis not in self._fit_phase_axes:
                return None
            return ParameterId(cluster_id=self.cluster_id, axis=self.axis, label=label)
        return ParameterId(peak_name=self.name, axis=self.axis, label=label)

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
                min=spec.min_val,
                max=spec.max_val,
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
        """Extract vectorized parameters for a cluster of peaks."""
        n_peaks = len(peaks)
        extras: dict[str, FloatArray] = {}
        index_map: dict[str, Any] = {}

        specs = self.get_parameter_spec()

        for spec in specs:
            vals = np.zeros(n_peaks, dtype=np.float64)
            idxs = np.full(n_peaks, -1, dtype=np.intp)

            for k, peak in enumerate(peaks):
                try:
                    shape = next(s for s in peak.shapes if s.axis == self.axis)
                except StopIteration:
                    continue

                p_name = self._param_name_for_spec(spec.name, shape)

                if p_name in params:
                    vals[k] = params[p_name].value
                    if param_map:
                        idxs[k] = param_map.get(p_name, -1)
                else:
                    vals[k] = spec.default

            extras[spec.name] = vals
            index_map[spec.name] = idxs

        return ClusterParameters(extras, index_map)


__all__ = ["ShapeBase"]
