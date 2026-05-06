"""Shape registry for lineshape model registration."""

import importlib
from typing import TYPE_CHECKING, TypeVar, cast

from peakfit.engine.types import Shape

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from types import ModuleType

    from peakfit.engine.lineshapes.protocol import LineshapeProtocol


TShape = TypeVar("TShape", bound=Shape)


# Global shape registry
SHAPES: dict[str, Callable[..., Shape]] = {}
LINESHAPE_MODULES: dict[str, LineshapeProtocol] = {}

_DISCOVERY_STATE = {"done": False}


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


def register_lineshape(module: LineshapeProtocol | ModuleType) -> LineshapeProtocol:
    """Register a module-based lineshape."""
    shape = cast("LineshapeProtocol", module)
    LINESHAPE_MODULES[shape.NAME] = shape
    return shape


def discover_lineshapes() -> None:
    """Import explicitly registered lineshapes so they can self-register."""
    if _DISCOVERY_STATE["done"]:
        return
    importlib.import_module("peakfit.engine.lineshapes")
    _DISCOVERY_STATE["done"] = True


def get_lineshape(name: str) -> LineshapeProtocol:
    """Get a registered module-based lineshape by name."""
    discover_lineshapes()
    return LINESHAPE_MODULES[name]


def list_lineshapes() -> list[str]:
    """List all registered module-based lineshape names."""
    discover_lineshapes()
    return sorted(LINESHAPE_MODULES.keys())


def reset_registry() -> None:
    """Reset registry state (useful for tests)."""
    SHAPES.clear()
    LINESHAPE_MODULES.clear()
    _DISCOVERY_STATE["done"] = False


__all__ = [
    "LINESHAPE_MODULES",
    "SHAPES",
    "discover_lineshapes",
    "get_lineshape",
    "get_shape",
    "list_lineshapes",
    "list_shapes",
    "register_lineshape",
    "register_shape",
    "reset_registry",
]
