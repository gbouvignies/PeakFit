"""Spectral noise correlation from the recorded processing.

See ``docs/correlated-noise.md`` §2. Thermal noise is white and circular complex in
the time domain. Apodization, zero filling, the Fourier transform and the first-order
phase correlate the real points of each processed axis, with lag correlation

    ρ(k) = Re[C(k) e^{iφ₁k/M}] / C(0),  C(k) = Σ_t w_t² e^{−2πikt/M}.

Axes are processed independently, so a cluster's correlation R is the elementwise
product of each axis's ρ at the lags between its fitting samples. It is exact under
these assumptions, not necessarily a description of the actual spectrum's noise.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

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
        rotated = spectrum * np.exp(1j * np.deg2rad(self.phase) * lags / self.transform)
        return np.real(rotated) / np.real(spectrum[0])


@dataclass(frozen=True)
class NoiseModel:
    """A spectrum's noise correlation: modelled per axis, or not modelled and why."""

    axes: tuple[AxisProcessing, ...] = ()  # spectral axes in data order; empty if not
    reason: str | None = None  # why the correlation is not modelled

    @property
    def modelled(self) -> bool:
        """Whether every spectral axis has a recorded, supported processing."""
        return self.reason is None


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


__all__ = ["AxisProcessing", "NoiseModel", "cluster_correlation", "not_modelled"]
