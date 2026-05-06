"""Shared core types and protocols.

Lower-level module to avoid circular dependencies across domain components.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from peakfit.engine.domain.param_map import ParameterMap
    from peakfit.engine.domain.params_scalar import Parameters
    from peakfit.shared.typing import ComplexArray, FloatArray, IntArray

# =============================================================================
# Type Aliases
# =============================================================================

type JsonValue = dict[str, "JsonValue"] | list["JsonValue"] | str | int | float | bool | None

# =============================================================================
# Kernel Result Structure
# =============================================================================


class KernelResult(NamedTuple):
    """Result container for kernel computations with derivatives.

    Attributes:
        values: Lineshape values of shape (N, K) or complex equivalents
        derivatives: Dictionary mapping strings to derivative arrays
    """

    values: FloatArray | ComplexArray
    derivatives: dict[str, FloatArray | ComplexArray]


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(slots=True)
class ParamSpec:
    """Specification for a lineshape parameter.

    Defines the default value, bounds, and unit for a single parameter.
    """

    name: str  # Parameter name (e.g. "eta", "j")
    default: float
    min_val: float
    max_val: float
    unit: str

    def create_bounds(self, center: float | None = None) -> tuple[float, float]:
        """Return the (min, max) bounds, optionally offset by center."""
        if center is not None:
            return (center + self.min_val, center + self.max_val)
        return (self.min_val, self.max_val)


@dataclass(slots=True)
class LineshapeResult:
    """Container for vectorized lineshape evaluation results.

    Attributes:
        values: Array of shape (N_points, K_peaks) containing lineshape values.
        derivatives: Dictionary mapping parameter names to derivative arrays.
    """

    values: FloatArray
    derivatives: dict[str, FloatArray] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the values array."""
        return self.values.shape

    @property
    def n_points(self) -> int:
        """Number of points in the values array."""
        return int(self.values.shape[0])

    @property
    def n_peaks(self) -> int:
        """Number of peaks in the values array."""
        return int(self.values.shape[1]) if self.values.ndim > 1 else 1

    def sum_peaks(self) -> FloatArray:
        """Sum across all peaks to get total lineshape (N,)."""
        result: FloatArray = np.sum(self.values, axis=1)
        return result


@dataclass(slots=True)
class ClusterParameters:
    """Vectorized parameter container for cluster-based evaluation.

    All parameters are handled uniformly through this container.
    """

    # All parameters are stored in extras using string keys
    extras: dict[str, FloatArray] = field(default_factory=dict)
    index_map: dict[str, IntArray] = field(default_factory=dict)

    @property
    def n_peaks(self) -> int:
        """Number of peaks in the cluster."""
        # Infer n_peaks from the first array in extras
        if not self.extras:
            return 0
        return len(next(iter(self.extras.values())))

    def get(self, name: str, default: float = 0.0) -> FloatArray | float:
        """Get extra parameter by name with default."""
        return self.extras.get(name, default)

    def __hash__(self) -> int:
        """Hash based on parameter values for caching."""
        key_parts = []
        for name, arr in sorted(self.extras.items()):
            key_parts.append((name, tuple(arr)))
        return hash(tuple(key_parts))


# =============================================================================
# Protocol Definition
# =============================================================================


@runtime_checkable
class Shape(Protocol):
    """Protocol defining the contract for all lineshape models.

    A Shape object is responsible for evaluating itself on a grid,
    managing its own parameters, and providing basis matrices for solvers.
    """

    @property
    def name(self) -> str:
        """Name of this specific shape instance (e.g. 'P1')."""
        ...

    @property
    def axis(self) -> str:
        """Axis label this shape belongs to (e.g. 'F2')."""
        ...

    @property
    def shape_name(self) -> str:
        """Type name of the shape (e.g. 'gaussian')."""
        ...

    def create_params(self) -> Parameters:
        """Create and return the Parameters object for this shape."""
        ...

    def get_parameter_spec(self) -> list[ParamSpec]:
        """Get specifications for this shape's parameters."""
        ...

    def fix_params(self, params: Parameters) -> None:
        """Fix all parameters of this shape (set vary=False)."""
        ...

    def release_params(self, params: Parameters) -> None:
        """Release all parameters of this shape (set vary=True)."""
        ...

    def evaluate_cluster(
        self,
        x_grid: FloatArray,
        cluster_params: ClusterParameters,
        compute_derivs: bool = False,
    ) -> LineshapeResult:
        """Evaluate lineshape for a cluster of peaks."""
        ...

    def get_cluster_parameters(
        self,
        peaks: Any,
        params: Parameters,
        param_map: ParameterMap | None = None,
    ) -> ClusterParameters:
        """Extract vectorized parameters for a cluster of peaks."""
        ...

    @property
    def center_i(self) -> int:
        """Get the integer index of the center position."""
        ...

    @property
    def dim_ctx(self) -> Any:
        """Get the dimension context."""
        ...

    def print(self, params: Parameters) -> str:
        """Return textual representation of shape parameters."""
        ...

    @property
    def center(self) -> float:
        """Get the center position (float, Hz or ppm)."""
        ...

    @center.setter
    def center(self, value: float) -> None:
        """Set the center position."""
        ...

    @property
    def cluster_id(self) -> int:
        """Get/Set cluster ID."""
        ...

    @cluster_id.setter
    def cluster_id(self, value: int) -> None: ...


__all__ = [
    "ClusterParameters",
    "JsonValue",
    "KernelResult",
    "LineshapeResult",
    "ParamSpec",
    "Shape",
]
