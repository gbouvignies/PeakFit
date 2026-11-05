"""Spectral noise correlation from the recorded processing.

See ``docs/correlated-noise.md`` §2. Thermal noise is white and circular complex in
the time domain. Apodization, zero filling, the Fourier transform and the first-order
phase correlate the real points of each processed axis, with lag correlation

    ρ(k) = Re[C(k) e^{−iφk/M}] / C(0),  C(k) = Σ_t w_t² e^{−2πikt/M},

where φ = P1 − 360°·g is the first-order phase NMRPipe applied, including the
Bruker digital-filter delay g it corrects during the Fourier transform. The sign is
NMRPipe's: it stores frequency in decreasing order, and the formula matches spectra
it processed (``tests/data/nmrpipe``).

Axes are processed independently, so a cluster's correlation R is the elementwise
product of each axis's ρ at the lags between its fitting samples. It is exact under
these assumptions, not necessarily a description of the actual spectrum's noise.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from peakfit.engine.domain.spectrum import get_dimension_label

if TYPE_CHECKING:
    from collections.abc import Sequence

    from peakfit.shared.typing import FloatArray, IntArray


@dataclass(frozen=True)
class AxisProcessing:
    """How one spectral axis was processed, as its header records it."""

    acquired: int  # N, complex points (TDSIZE)
    transform: int  # M, Fourier-transform size (FTSIZE)
    window: Literal["none", "sine_bell"]
    offset: float = 0.0  # sine bell: sin^power(π·offset + π(end − offset)·t/(N − 1))
    end: float = 1.0
    power: float = 1.0
    first_point: float = 1.0  # scale applied to the first time-domain point
    phase: float = 0.0  # first-order phase P1, degrees
    # Bruker digital-filter group delay (FDDMXVAL) NMRPipe corrected during the
    # Fourier transform, in points; 0 when no correction was applied.
    filter_delay: float = 0.0

    def __post_init__(self) -> None:
        """Refuse processing outside the supported noise-model domain."""
        values = (
            self.offset,
            self.end,
            self.power,
            self.first_point,
            self.phase,
            self.filter_delay,
        )
        if (
            self.window not in {"none", "sine_bell"}
            or self.acquired < 1
            or self.transform < self.acquired
            or not all(np.isfinite(v) for v in values)
            or self.power <= 0
        ):
            msg = f"Invalid axis processing: {self}"
            raise ValueError(msg)

    def weights(self) -> FloatArray:
        """The apodization over the acquired points, first point scaled."""
        if self.window == "sine_bell":
            t = np.arange(self.acquired) / max(self.acquired - 1, 1)
            angle = np.pi * self.offset + np.pi * (self.end - self.offset) * t
            w = np.sin(angle) ** self.power
        else:
            w = np.ones(self.acquired)
        w[0] *= self.first_point
        return w

    def correlation(self) -> FloatArray:
        """ρ(k) for every lag k = 0 … M − 1; ρ(0) = 1 and ρ(−k) = ρ(k)."""
        w2 = self.weights() ** 2
        lags = np.arange(self.transform)
        spectrum = np.fft.fft(w2, n=self.transform)  # C(k) = Σ_t w_t² e^{−2πikt/M}
        phase = np.deg2rad(self.phase - 360.0 * self.filter_delay)
        rotated = spectrum * np.exp(-1j * phase * lags / self.transform)
        return np.real(rotated) / np.real(spectrum[0])


@dataclass(frozen=True)
class NoiseModel:
    """A spectrum's noise correlation: modelled per axis, or not modelled and why."""

    axes: tuple[AxisProcessing, ...] = ()  # spectral axes in data order; empty if not
    reason: str | None = None  # why the correlation is not modelled

    def __post_init__(self) -> None:
        """Modelled with axes, or not modelled with a reason: never both, never neither."""
        if bool(self.axes) == bool(self.reason) or not all(
            isinstance(a, AxisProcessing) for a in self.axes
        ):
            msg = "A noise model is modelled with axes, or not modelled with a reason."
            raise ValueError(msg)

    @property
    def modelled(self) -> bool:
        """Whether every spectral axis has a recorded, supported processing."""
        return self.reason is None

    @property
    def description(self) -> str:
        """One sentence for every report: what the published errors assume."""
        if not self.modelled:
            return (
                f"Noise correlation not modelled ({self.reason}): uncertainties assume "
                "independent spectral points and can be too small for zero-filled or "
                "apodized data."
            )
        axes = "; ".join(
            f"{get_dimension_label(i)}: "
            f"{'sine bell' if a.window == 'sine_bell' else 'no apodization'}, "
            f"{a.acquired} → {a.transform} points"
            + (f", digital filter {a.filter_delay:g} points" if a.filter_delay else "")
            for i, a in enumerate(self.axes)
        )
        return f"Noise correlation modelled from the recorded processing ({axes})."


def not_modelled(reason: str) -> NoiseModel:
    """A noise model that falls back to independent points, recording why."""
    return NoiseModel(reason=reason)


def cluster_correlation(model: NoiseModel, grid_indices: Sequence[IntArray]) -> FloatArray:
    """R between a cluster's fitting samples: the product of each axis's ρ at its lags.

    ``grid_indices`` holds one index array per spectral axis, aligned point by point,
    as ``Cluster.grid_indices``. Planes are independent, so R is the same for each.
    """
    if not model.modelled:
        raise ValueError(f"Noise correlation is not modelled: {model.reason}")
    if len(grid_indices) != len(model.axes):
        raise ValueError("One index array per spectral axis is required.")
    n_points = len(grid_indices[0])
    correlation = np.ones((n_points, n_points))
    for axis, indices in zip(model.axes, grid_indices, strict=True):
        index = np.asarray(indices)
        lags = np.abs(index[:, None] - index[None, :])
        correlation *= axis.correlation()[lags]
    return correlation


def modelled_correlation(model: NoiseModel, grid_indices: Sequence[IntArray]) -> FloatArray | None:
    """R for a cluster when the correlation is modelled; None means independent points."""
    return cluster_correlation(model, grid_indices) if model.modelled else None


__all__ = [
    "AxisProcessing",
    "NoiseModel",
    "cluster_correlation",
    "modelled_correlation",
    "not_modelled",
]
