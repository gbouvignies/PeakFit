"""Construct built-in lineshape model instances."""

from typing import TYPE_CHECKING

from peakfit.engine.lineshapes.models import (
    SP1,
    SP2,
    Gaussian,
    GaussianDoublet,
    Lorentzian,
    LorentzianDoublet,
    NoApod,
    NoApodDoublet,
    PseudoVoigt,
    PseudoVoigtDoublet,
    SP1Doublet,
    SP2Doublet,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from peakfit.engine.domain.spectrum import AxisLineshape, Spectra
    from peakfit.engine.types import Shape


# Lineshape family -> (singlet, doublet) constructors.
SHAPES: dict[str, tuple[Callable[..., Shape], Callable[..., Shape]]] = {
    "gaussian": (Gaussian, GaussianDoublet),
    "lorentzian": (Lorentzian, LorentzianDoublet),
    "no_apod": (NoApod, NoApodDoublet),
    "pvoigt": (PseudoVoigt, PseudoVoigtDoublet),
    "sp1": (SP1, SP1Doublet),
    "sp2": (SP2, SP2Doublet),
}


def create_shapes(
    spectra: Spectra,
    lineshapes: Sequence[AxisLineshape],
    peak_name: str,
    positions: Sequence[float],
) -> list[Shape]:
    """One shape per spectral axis of a peak, from each axis's resolved model."""
    if len(positions) != len(lineshapes):
        msg = "Number of positions and axis lineshapes must match"
        raise ValueError(msg)
    shapes: list[Shape] = []
    for dim, (center, lineshape) in enumerate(zip(positions, lineshapes, strict=True), start=1):
        try:
            singlet, doublet = SHAPES[lineshape.family]
        except KeyError as exc:
            available = ", ".join(sorted(SHAPES))
            msg = f"Unknown lineshape family '{lineshape.family}'. Available: {available}"
            raise ValueError(msg) from exc
        provider = doublet if lineshape.doublet is not None else singlet
        shapes.append(provider(peak_name, center, spectra, dim, lineshape))
    return shapes


__all__ = ["SHAPES", "create_shapes"]
