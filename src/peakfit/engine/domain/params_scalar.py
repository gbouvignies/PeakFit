"""Scalar parameter models and collections."""

from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    model_validator,
)

from peakfit.engine.domain.param_id import ParameterId

if TYPE_CHECKING:
    from collections.abc import (
        ItemsView,
        Iterable,
        Iterator,
        KeysView,
        Sequence,
        ValuesView,
    )

    from peakfit.engine.domain.peaks import Peak


_BOUND_DISPLAY_THRESHOLD = 1e10


class Parameter(BaseModel):
    """Single NMR fitting parameter with bounds and metadata."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    name: str
    value: float
    min: float = -np.inf
    max: float = np.inf
    vary: bool = True
    unit: str = ""  # Optional unit string (e.g., "Hz", "ppm", "deg")
    stderr: float = 0.0  # Standard error from fitting
    # True for parameters computed analytically (e.g., amplitudes)
    computed: bool = False
    param_id: ParameterId | None = None  # Optional structured identifier

    _parent: Parameters | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_parameter(self) -> Parameter:
        """Validate parameter bounds."""
        # Enforce invariant: computed parameters cannot vary
        if self.computed and self.vary:
            msg = f"Parameter {self.name}: computed=True requires vary=False"
            raise ValueError(msg)

        if self.min > self.max:
            msg = f"Parameter {self.name}: min ({self.min}) > max ({self.max})"
            raise ValueError(msg)

        # Check bounds only if not infinite
        epsilon = 1e-5
        if not (np.isinf(self.min) and np.isinf(self.max)) and not (
            self.min - epsilon <= self.value <= self.max + epsilon
        ):
            msg = (
                f"Parameter {self.name}: value ({self.value}) "
                f"outside bounds [{self.min}, {self.max}]"
            )
            raise ValueError(msg)
        return self

    def __repr__(self) -> str:
        """Return a string representation of the parameter."""
        if self.computed:
            vary_str = "computed"
        elif self.vary:
            vary_str = "vary"
        else:
            vary_str = "fixed"
        min_str = f"{self.min:.4g}" if self.min > -_BOUND_DISPLAY_THRESHOLD else "-inf"
        max_str = f"{self.max:.4g}" if self.max < _BOUND_DISPLAY_THRESHOLD else "inf"
        unit_str = f" {self.unit}" if self.unit else ""
        return (
            f"<Parameter {self.name}={self.value:.6g}{unit_str} "
            f"[{min_str}, {max_str}] ({vary_str})>"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute with optional cache invalidation for vary changes."""
        if name == "vary" and "vary" in self.__dict__:
            previous = self.__dict__["vary"]
            super().__setattr__(name, value)
            self._notify_vary_change(previous, value)
            return
        super().__setattr__(name, value)

    def _notify_vary_change(self, previous: bool, current: bool) -> None:
        """Invalidate parent cache when vary status changes."""
        if previous != current and self._parent is not None:
            self._parent._invalidate_vary_cache()

    def is_at_boundary(self, tol: float = 1e-6) -> bool:
        """Check if parameter is at or near its boundary."""
        at_min = abs(self.value - self.min) < tol * (1 + abs(self.value))
        at_max = abs(self.value - self.max) < tol * (1 + abs(self.value))
        return at_min or at_max

    def relative_position(self) -> float:
        """Get the relative position of value within bounds (0 to 1)."""
        if self.max == self.min:
            return 0.5
        if np.isinf(self.min) or np.isinf(self.max):
            return 0.5
        return (self.value - self.min) / (self.max - self.min)


class Parameters(BaseModel):
    """Collection of fitting parameters."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    params: dict[str, Parameter] = Field(default_factory=dict)

    _vary_names_cache: list[str] | None = PrivateAttr(default=None)
    _vary_values_cache: np.ndarray | None = PrivateAttr(default=None)
    _vary_bounds_cache: tuple[np.ndarray, np.ndarray] | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _init_parents(self) -> Parameters:
        """Attach parent reference to parameters for cache invalidation."""
        for param in self.params.values():
            param._parent = self
        return self

    def _register_param(self, param: Parameter) -> None:
        """Attach parent to a parameter and invalidate caches."""
        param._parent = self
        self._invalidate_vary_cache()

    def _invalidate_vary_cache(self) -> None:
        """Invalidate cached vary names/values/bounds."""
        self._vary_names_cache = None
        self._vary_values_cache = None
        self._vary_bounds_cache = None

    def _get_vary_names_cached(self) -> list[str]:
        """Return cached vary names, computing when needed."""
        if self._vary_names_cache is None:
            self._vary_names_cache = [name for name, param in self.params.items() if param.vary]
        return self._vary_names_cache

    def _refresh_vary_values_cache(self, names: list[str]) -> np.ndarray:
        """Refresh cached array of varying values."""
        if self._vary_values_cache is None or self._vary_values_cache.shape != (len(names),):
            self._vary_values_cache = np.array([self.params[name].value for name in names])
        else:
            for i, name in enumerate(names):
                self._vary_values_cache[i] = self.params[name].value
        return self._vary_values_cache

    def _refresh_vary_bounds_cache(
        self,
        names: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Refresh cached arrays of varying bounds."""
        if self._vary_bounds_cache is None or self._vary_bounds_cache[0].shape != (len(names),):
            lower = np.array([self.params[name].min for name in names])
            upper = np.array([self.params[name].max for name in names])
            self._vary_bounds_cache = (lower, upper)
        else:
            lower, upper = self._vary_bounds_cache
            for i, name in enumerate(names):
                lower[i] = self.params[name].min
                upper[i] = self.params[name].max
        return self._vary_bounds_cache

    def add(
        self,
        name: str | ParameterId,
        value: float = 0.0,
        min: float = -np.inf,
        max: float = np.inf,
        vary: bool = True,
        unit: str = "",
        stderr: float = 0.0,
        computed: bool = False,
    ) -> None:
        """Add a parameter."""
        # Handle ParameterId input
        if isinstance(name, ParameterId):
            param_id = name
            name_str = param_id.name
        else:
            param_id = None
            name_str = name

        self.params[name_str] = Parameter(
            name=name_str,
            value=value,
            min=min,
            max=max,
            vary=vary,
            unit=unit,
            stderr=stderr,
            computed=computed,
            param_id=param_id,
        )
        self._register_param(self.params[name_str])

    def __getitem__(self, key: str) -> Parameter:
        """Get parameter by name."""
        return self.params[key]

    def __setitem__(self, key: str, value: Parameter) -> None:
        """Set parameter."""
        self.params[key] = value
        self._register_param(value)

    def __contains__(self, key: str) -> bool:
        """Check if parameter exists."""
        return key in self.params

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        """Iterate over parameter names."""
        return iter(self.params)

    def keys(self) -> KeysView[str]:
        """Get parameter names."""
        return self.params.keys()

    def values(self) -> ValuesView[Parameter]:
        """Get parameter objects."""
        return self.params.values()

    def items(self) -> ItemsView[str, Parameter]:
        """Get parameter name-value pairs."""
        return self.params.items()

    def update(self, other: Parameters) -> None:
        """Update parameters from another Parameters object."""
        for name, param in other.items():
            self.params[name] = param
            self._register_param(param)

    @classmethod
    def from_peaks(cls, peaks: Sequence[Peak], *, fixed: bool = False) -> Parameters:
        """Build parameters for a list of peaks."""
        params = cls()
        for peak in peaks:
            params.update(peak.create_params())

        if fixed:
            for name in params:
                if name.endswith("0"):
                    params[name].vary = False

        return params

    def copy(
        self,
        *,
        include: Any = None,
        exclude: Any = None,
        update: dict[str, Any] | None = None,
        deep: bool = False,
    ) -> Parameters:
        """Create a copy of parameters (signature aligned with BaseModel)."""
        new_params = Parameters()
        items_iter: Iterable[tuple[str, Parameter]] = self.params.items()
        if include is not None:
            items_iter = ((k, v) for k, v in items_iter if k in include)
        if exclude is not None:
            items_iter = ((k, v) for k, v in items_iter if k not in exclude)

        for name, param in items_iter:
            new_params.params[name] = param.model_copy(deep=deep)
            new_params._register_param(new_params.params[name])

        if update:
            for name, value in update.items():
                new_params.params[name] = value
                if isinstance(value, Parameter):
                    new_params._register_param(value)

        return new_params

    def get_vary_names(self) -> list[str]:
        """Get names of parameters that vary (nonlinear optimization)."""
        return self._get_vary_names_cached()

    def get_computed_names(self) -> list[str]:
        """Get names of computed parameters (e.g., amplitudes)."""
        return [name for name, param in self.params.items() if param.computed]

    def get_fitted_names(self) -> list[str]:
        """Get names of all fitted parameters (vary=True or computed=True)."""
        return [name for name, param in self.params.items() if param.vary or param.computed]

    def get_n_fitted_params(self) -> int:
        """Get total number of fitted parameters for DOF calculation."""
        return sum(1 for param in self.params.values() if param.vary or param.computed)

    def get_vary_values(self) -> np.ndarray:
        """Get values of varying parameters as array."""
        names = self._get_vary_names_cached()
        return self._refresh_vary_values_cache(names)

    def get_vary_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get bounds for varying parameters."""
        names = self._get_vary_names_cached()
        return self._refresh_vary_bounds_cache(names)

    def set_vary_values(self, values: np.ndarray) -> None:
        """Set values of varying parameters from array."""
        names = self.get_vary_names()
        for name, value in zip(names, values, strict=True):
            self.params[name].value = value
        if self._vary_values_cache is not None and self._vary_values_cache.shape == values.shape:
            self._vary_values_cache[:] = values

    def set_errors(self, errors: np.ndarray) -> None:
        """Set standard errors for varying parameters."""
        names = self.get_vary_names()
        for name, error in zip(names, errors, strict=True):
            self.params[name].stderr = error

    def get_vary_bounds_list(self) -> list[tuple[float, float]]:
        """Get bounds for varying parameters as list of tuples."""
        lower, upper = self.get_vary_bounds()
        return list(zip(lower, upper, strict=True))

    def __len__(self) -> int:
        """Return number of parameters."""
        return len(self.params)

    def __repr__(self) -> str:
        """Return a string representation of the parameters collection."""
        n_total = len(self.params)
        n_vary = len(self.get_vary_names())
        n_computed = len(self.get_computed_names())
        if n_computed > 0:
            return f"<Parameters: {n_total} total, {n_vary} varying, {n_computed} computed>"
        return f"<Parameters: {n_total} total, {n_vary} varying>"

    def summary(self) -> str:
        """Get a formatted summary of all parameters."""
        lines = ["Parameters:", "=" * 60]
        for name in self.params:
            param = self.params[name]
            if param.computed:
                vary_str = "computed"
            elif param.vary:
                vary_str = "vary"
            else:
                vary_str = "fixed"
            min_str = f"{param.min:.4g}" if param.min > -_BOUND_DISPLAY_THRESHOLD else "-inf"
            max_str = f"{param.max:.4g}" if param.max < _BOUND_DISPLAY_THRESHOLD else "inf"
            line = (
                f"  {name:20s} = {param.value:12.6g} [{min_str:>10s}, {max_str:<10s}] ({vary_str})"
            )
            lines.append(line)
        lines.append("=" * 60)
        return "\n".join(lines)

    def get_boundary_params(self) -> list[str]:
        """Get names of parameters that are at their boundaries."""
        return [
            name for name, param in self.params.items() if param.vary and param.is_at_boundary()
        ]

    def freeze(self, names: list[str] | None = None) -> None:
        """Set parameters to not vary (freeze them)."""
        if names is None:
            names = list(self.params.keys())
        for name in names:
            if name in self.params:
                param = self.params[name]
                param.vary = False

    def unfreeze(self, names: list[str] | None = None) -> None:
        """Set parameters to vary (unfreeze them)."""
        if names is None:
            names = list(self.params.keys())
        for name in names:
            if name in self.params:
                param = self.params[name]
                param.vary = True

    def get_by_peak(self, peak_name: str) -> list[Parameter]:
        """Get all parameters belonging to a specific peak."""
        return [
            param
            for param in self.params.values()
            if param.param_id is not None and param.param_id.peak_name == peak_name
        ]

    def get_by_axis(self, axis: str) -> list[Parameter]:
        """Get all parameters for a specific axis/dimension."""
        return [
            param
            for param in self.params.values()
            if param.param_id is not None and param.param_id.axis == axis
        ]


__all__ = ["Parameter", "Parameters"]
