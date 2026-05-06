"""Protocol for stateless, module-based lineshape definitions."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy.typing as npt

    from peakfit.engine.lineshapes.grid import SpectralGrid
    from peakfit.engine.types import ParamSpec
    from peakfit.shared.typing import FloatArray


@dataclass(frozen=True, slots=True)
class LineshapeContext:
    """Optional context for lineshape evaluation and parameter defaults."""

    grid: SpectralGrid | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LineshapeProtocol(Protocol):
    """Module interface for self-contained lineshape definitions."""

    NAME: str
    PARAM_NAMES: tuple[str, ...]

    @staticmethod
    def function(
        x: npt.ArrayLike,
        *params: npt.ArrayLike,
        context: LineshapeContext | None = None,
    ) -> FloatArray:
        """Evaluate the lineshape at x for the given parameters.

        Parameters must be arrays with one value per peak.
        """
        ...

    @staticmethod
    def param_specs(
        center: float,
        context: LineshapeContext | None = None,
    ) -> tuple[ParamSpec, ...]:
        """Return parameter specs (defaults, bounds, units) for this lineshape."""
        ...

    @staticmethod
    def bounds(
        center: float,
        context: LineshapeContext | None = None,
    ) -> tuple[tuple[float, float], ...]:
        """Return parameter bounds aligned with PARAM_NAMES."""
        ...


__all__ = ["LineshapeContext", "LineshapeProtocol"]
