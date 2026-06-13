"""Shape registry for lineshape model registration."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from peakfit.engine.types import Shape


# Global shape registry
SHAPES: dict[str, Callable[..., Shape]] = {}

_DISCOVERY_STATE = {"done": False}
_BUILTIN_LINESHAPE_MODULES = (
    "peakfit.engine.lineshapes.gaussian.model",
    "peakfit.engine.lineshapes.lorentzian.model",
    "peakfit.engine.lineshapes.no_apod.model",
    "peakfit.engine.lineshapes.pvoigt.model",
    "peakfit.engine.lineshapes.sp1.model",
    "peakfit.engine.lineshapes.sp2.model",
)


def register_shape(
    shape_names: str | Iterable[str],
) -> Callable[[Callable[..., Shape]], Callable[..., Shape]]:
    """Register a shape class.

    Args:
        shape_names: Single name or iterable of names to register the shape under

    Returns:
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

    def decorator(shape_provider: Callable[..., Shape]) -> Callable[..., Shape]:
        for name in shape_names:
            SHAPES[name] = shape_provider
        return shape_provider

    return decorator


def get_shape(name: str) -> Callable[..., Shape]:
    """Get a shape class by name.

    Args:
        name: Name of the shape to retrieve

    Returns:
    -------
        Shape provider (callable/class)

    Raises:
    ------
        KeyError: If shape name not found in registry
    """
    discover_lineshapes()
    return SHAPES[name]


def list_shapes() -> list[str]:
    """List all registered shape names.

    Returns:
    -------
        List of registered shape names
    """
    discover_lineshapes()
    return list(SHAPES.keys())


def discover_lineshapes() -> None:
    """Import explicitly registered lineshapes so they can self-register."""
    if _DISCOVERY_STATE["done"]:
        return
    for module_name in _BUILTIN_LINESHAPE_MODULES:
        importlib.import_module(module_name)
    _DISCOVERY_STATE["done"] = True


__all__ = [
    "SHAPES",
    "discover_lineshapes",
    "get_shape",
    "list_shapes",
    "register_shape",
]
