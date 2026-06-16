"""Construct built-in lineshape model instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

from peakfit.engine.lineshapes.gaussian.model import Gaussian, GaussianDoublet
from peakfit.engine.lineshapes.lorentzian.model import Lorentzian, LorentzianDoublet
from peakfit.engine.lineshapes.no_apod.model import NoApod, NoApodDoublet
from peakfit.engine.lineshapes.pvoigt.model import PseudoVoigt, PseudoVoigtDoublet
from peakfit.engine.lineshapes.sp1.model import SP1, SP1Doublet
from peakfit.engine.lineshapes.sp2.model import SP2, SP2Doublet

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from peakfit.engine.domain.config import FitConfig
    from peakfit.engine.domain.spectrum import Spectra
    from peakfit.engine.types import Shape


SHAPES: dict[str, Callable[..., Shape]] = {
    "gaussian": Gaussian,
    "gaussian_doublet": GaussianDoublet,
    "lorentzian": Lorentzian,
    "lorentzian_doublet": LorentzianDoublet,
    "no_apod": NoApod,
    "no_apod_doublet": NoApodDoublet,
    "pvoigt": PseudoVoigt,
    "pvoigt_doublet": PseudoVoigtDoublet,
    "sp1": SP1,
    "sp1_doublet": SP1Doublet,
    "sp2": SP2,
    "sp2_doublet": SP2Doublet,
}


def _available_shapes() -> list[str]:
    """Return the available built-in shape names."""
    return sorted(SHAPES)


def _create_shape(
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
            _create_shape(
                spectra,
                config,
                shape_name,
                peak_name=peak_name,
                center=center,
                dim=dim,
            )
        )
    return shapes


def _resolve_shape_provider(shape_type: str) -> Callable[..., Shape]:
    try:
        return SHAPES[shape_type]
    except KeyError as exc:
        available = ", ".join(_available_shapes())
        msg = f"Unknown lineshape '{shape_type}'. Available: {available}"
        raise ValueError(msg) from exc


__all__ = ["create_shapes"]
