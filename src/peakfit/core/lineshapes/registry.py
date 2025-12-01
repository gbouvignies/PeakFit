"""Shape registry for dynamic lineshape model registration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from peakfit.core.domain.spectrum import SpectralParameters
    from peakfit.core.fitting.parameters import Parameters
    from peakfit.core.shared.typing import FloatArray, IntArray


@runtime_checkable
class Shape(Protocol):
    """Protocol for lineshape models."""

    axis: str
    name: str
    cluster_id: int
    center: float
    size: int
    spec_params: SpectralParameters

    def create_params(self) -> Parameters:
        """Create a Parameters collection for the shape."""
        ...

    def fix_params(self, params: Parameters) -> None:
        """Fix/lock parameters in a Parameters collection."""
        ...

    def release_params(self, params: Parameters) -> None:
        """Release/unlock parameters in a Parameters collection."""
        ...

    def evaluate(self, x_pt: IntArray, params: Parameters) -> FloatArray:
        """Evaluate the shape at given grid point indices and params."""
        ...

    def print(self, params: Parameters) -> str:
        """Format shape parameters to a textual representation."""
        ...

    @property
    def center_i(self) -> int:
        """Return the center point index (in integer points) for the shape."""
        ...


# Global shape registry
SHAPES: dict[str, type[Shape]] = {}


def register_shape(
    shape_names: str | Iterable[str],
) -> Callable[[type[Shape]], type[Shape]]:
    """Register a shape class.

    Args:
        shape_names: Single name or iterable of names to register the shape under

    Returns
    -------
        Decorator that registers the shape class

    Example:
        @register_shape("gaussian")
        class Gaussian(BaseShape):
            ...

        @register_shape(["lorentzian", "lorentz"])
        class Lorentzian(BaseShape):
            ...
    """
    if isinstance(shape_names, str):
        shape_names = [shape_names]

    def decorator(shape_class: type[Shape]) -> type[Shape]:
        for name in shape_names:
            SHAPES[name] = shape_class
        return shape_class

    return decorator


def get_shape(name: str) -> type[Shape]:
    """Get a shape class by name.

    Args:
        name: Name of the shape to retrieve

    Returns
    -------
        Shape class

    Raises
    ------
        KeyError: If shape name not found in registry
    """
    return SHAPES[name]


def list_shapes() -> list[str]:
    """List all registered shape names.

    Returns
    -------
        List of registered shape names
    """
    return list(SHAPES.keys())
