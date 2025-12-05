"""No-Apod lineshape model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from peakfit.core.lineshapes.apodization import ApodizationEvaluator, ApodShape
from peakfit.core.lineshapes.registry import register_shape

if TYPE_CHECKING:
    from peakfit.core.shared.typing import FloatArray


class NoApodEvaluator(ApodizationEvaluator):
    """Non-apodized lineshape evaluator.

    Complex lineshape from pure FID Fourier transform without window function::

        F(dx) = aq * (1 - exp(-z)) / z  where z = aq * (i*dx + r2)

    Returns real part after optional phase correction.
    """

    def _compute_complex(
        self, dx: FloatArray, r2: float, calc_derivs: bool
    ) -> tuple[FloatArray, FloatArray | None, FloatArray | None]:
        """Compute complex lineshape for single component."""
        aq = self.aq
        z1 = aq * (1j * dx + r2)
        emz1 = np.exp(-z1)
        val = aq * (1.0 - emz1) / z1

        if not calc_derivs:
            return val, None, None  # type: ignore[return-value]

        dval_dz1 = aq * (emz1 * (z1 + 1) - 1) / z1**2
        d_dx = 1j * aq * dval_dz1
        d_r2 = aq * dval_dz1
        return val, d_dx, d_r2  # type: ignore[return-value]

    def height(self, r2: float, j_hz: float = 0.0) -> float:
        """Compute height at peak position."""
        del j_hz  # Unused but part of interface
        z1 = self.aq * r2
        height = np.real(self.aq * (1.0 - np.exp(-z1)) / z1)
        return float(height)


@register_shape("no_apod")
class NoApod(ApodShape):
    """Non-apodized lineshape."""

    shape_name = "no_apod"

    def _create_evaluator(self) -> Any:
        """Create NoApod evaluator."""
        return NoApodEvaluator(aq=self.spec_params.aq_time)


def no_apod(
    dx: FloatArray, r2: float, aq: float, phase: float = 0.0, j_hz: float = 0.0
) -> FloatArray:
    """Evaluate non-apodized lineshape.

    Convenience function that creates a temporary NoApodEvaluator.
    For repeated evaluations with the same aq, use NoApodEvaluator directly.

    Args:
        dx: Frequency offset array (radians/s)
        r2: R2 relaxation rate (Hz)
        aq: Acquisition time in seconds
        phase: Phase correction in degrees, default 0.0
        j_hz: J-coupling constant (Hz), default 0.0

    Returns
    -------
        Lineshape values at given offsets
    """
    evaluator = NoApodEvaluator(aq)
    return evaluator.evaluate(np.asarray(dx), r2, phase, j_hz)
