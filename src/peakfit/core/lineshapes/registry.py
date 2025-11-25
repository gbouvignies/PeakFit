"""Shape registry for dynamic lineshape model registration."""

from collections.abc import Callable, Iterable
from typing import Protocol


class Shape(Protocol):
    """Protocol for lineshape models."""

    axis: str
    name: str
    cluster_id: int
    center: float
    size: int

    def create_params(self) -> object: ...
    def fix_params(self, params: object) -> None: ...
    def release_params(self, params: object) -> None: ...
    def evaluate(self, x_pt: object, params: object) -> object: ...
    def print(self, params: object) -> str: ...
    @property
    def center_i(self) -> int: ...
    @property
    def prefix(self) -> str: ...


# Global shape registry
SHAPES: dict[str, Callable[..., Shape]] = {}


def register_shape(
    shape_names: str | Iterable[str],
) -> Callable[[type[Shape]], type[Shape]]:
    """Decorator to register a shape class.

    Args:
        shape_names: Single name or iterable of names to register the shape under

    Returns:
        Decorator function that registers the shape class

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


def get_shape(name: str) -> Callable[..., Shape]:
    """Get a shape class by name.

    Args:
        name: Name of the shape to retrieve

    Returns:
        Shape class

    Raises:
        KeyError: If shape name not found in registry
    """
    return SHAPES[name]


def list_shapes() -> list[str]:
    """List all registered shape names.

    Returns:
        List of registered shape names
    """
    return list(SHAPES.keys())
