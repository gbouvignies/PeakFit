"""Factory helpers for constructing lineshape model instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

from peakfit.core.domain.spectrum import determine_shape_name
from peakfit.core.lineshapes.registry import SHAPES

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from peakfit.core.domain.spectrum import Spectra
    from peakfit.core.lineshapes.registry import Shape
    from peakfit.core.shared.typing import FittingOptions


class LineshapeFactory:
    """Factory for instantiating registered lineshape models."""

    def __init__(self, spectra: Spectra, options: FittingOptions) -> None:
        self._spectra = spectra
        self._options = options

    def available_shapes(self) -> list[str]:
        """Return the available registered shape names."""
        return sorted(SHAPES.keys())

    def create(
        self,
        shape_type: str,
        *,
        peak_name: str,
        center: float,
        dim: int,
    ) -> Shape:
        """Create a single shape instance for the provided peak."""
        shape_cls = self._resolve_shape_class(shape_type)

        return shape_cls(peak_name, center, self._spectra, dim, self._options)

    def create_shapes(
        self,
        peak_name: str,
        positions: Sequence[float],
        shape_names: Sequence[str],
    ) -> list[Shape]:
        """Create shapes for each peak dimension using provided names."""
        self._validate_lengths(positions, shape_names)
        shapes: list[Shape] = []
        for dim, (center, shape_name) in enumerate(
            zip(positions, shape_names, strict=False), start=1
        ):
            shapes.append(
                self.create(
                    shape_name,
                    peak_name=peak_name,
                    center=center,
                    dim=dim,
                )
            )
        return shapes

    def create_auto_shapes(self, peak_name: str, positions: Sequence[float]) -> list[Shape]:
        """Create shapes using automatically detected names for each dimension."""
        auto_names = self.auto_shape_names()
        if len(auto_names) < len(positions):
            msg = "Not enough automatically detected shapes for peak positions"
            raise ValueError(msg)
        return self.create_shapes(peak_name, positions, auto_names[: len(positions)])

    def auto_shape_names(self) -> list[str]:
        """Detect shape names for each indirect dimension."""
        params = self._spectra.params[1:]
        return [determine_shape_name(param) for param in params]

    def detect_shape_name(self, dim: int) -> str:
        """Detect the shape name for a single dimension index."""
        try:
            params = self._spectra.params[dim]
        except IndexError as exc:  # pragma: no cover - defensive guard
            raise ValueError("Dimension index out of range for spectra") from exc
        return determine_shape_name(params)

    @staticmethod
    def _validate_lengths(positions: Sequence[float], shape_names: Sequence[str]) -> None:
        if len(positions) != len(shape_names):
            msg = "Number of positions and shape names must match"
            raise ValueError(msg)

    def _resolve_shape_class(self, shape_type: str) -> Callable[..., Shape]:
        try:
            return SHAPES[shape_type]
        except KeyError as exc:
            available = ", ".join(self.available_shapes())
            msg = f"Unknown lineshape '{shape_type}'. Available: {available}"
            raise ValueError(msg) from exc


__all__ = ["LineshapeFactory"]
