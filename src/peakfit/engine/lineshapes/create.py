"""Construct registered lineshape model instances."""

from typing import TYPE_CHECKING

from peakfit.engine.domain.spectrum import determine_shape_name
from peakfit.engine.lineshapes import registry

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from peakfit.engine.domain.config import FitConfig
    from peakfit.engine.domain.spectrum import Spectra
    from peakfit.engine.lineshapes.registry import Shape


def available_shapes() -> list[str]:
    """Return the available registered shape names."""
    return sorted(registry.list_shapes())


def create_shape(
    spectra: Spectra,
    config: FitConfig,
    shape_type: str,
    *,
    peak_name: str,
    center: float,
    dim: int,
) -> Shape:
    """Create a single shape instance for the provided peak."""
    shape_provider = _resolve_shape_provider(shape_type)
    return shape_provider(peak_name, center, spectra, dim, config)


def create_shapes(
    spectra: Spectra,
    config: FitConfig,
    peak_name: str,
    positions: Sequence[float],
    shape_names: Sequence[str],
) -> list[Shape]:
    """Create shapes for each peak dimension using provided names."""
    if len(positions) != len(shape_names):
        msg = "Number of positions and shape names must match"
        raise ValueError(msg)

    shapes: list[Shape] = []
    for dim, (center, shape_name) in enumerate(zip(positions, shape_names, strict=False), start=1):
        shapes.append(
            create_shape(
                spectra,
                config,
                shape_name,
                peak_name=peak_name,
                center=center,
                dim=dim,
            )
        )
    return shapes


def auto_shape_names(spectra: Spectra) -> list[str]:
    """Detect shape names for each indirect dimension."""
    return [determine_shape_name(param) for param in spectra.params[1:]]


def detect_shape_name(spectra: Spectra, dim: int) -> str:
    """Detect the shape name for a single dimension index."""
    try:
        params = spectra.params[dim]
    except IndexError as exc:  # pragma: no cover - defensive guard
        raise ValueError("Dimension index out of range for spectra") from exc
    return determine_shape_name(params)


def _resolve_shape_provider(shape_type: str) -> Callable[..., Shape]:
    try:
        return registry.get_shape(shape_type)
    except KeyError as exc:
        available = ", ".join(available_shapes())
        msg = f"Unknown lineshape '{shape_type}'. Available: {available}"
        raise ValueError(msg) from exc


__all__ = [
    "auto_shape_names",
    "available_shapes",
    "create_shape",
    "create_shapes",
    "detect_shape_name",
]
