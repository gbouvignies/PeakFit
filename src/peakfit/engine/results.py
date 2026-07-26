"""Numerical fit result primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from peakfit.engine.algorithms.mcmc import UncertaintyResult
    from peakfit.engine.domain.params_scalar import Parameters
    from peakfit.shared.typing import FloatArray


def compute_chi_squared(residuals: FloatArray) -> float:
    """Compute chi-squared as the sum of squared residuals."""
    return float(np.sum(residuals**2))


def compute_degrees_of_freedom(n_data: int, n_params: int) -> int:
    """Compute degrees of freedom, capped at 1 to avoid division by zero."""
    return max(1, n_data - n_params)


def compute_reduced_chi_squared(
    chi_squared: float,
    n_data: int,
    n_params: int,
) -> float:
    """Compute reduced chi-squared with the standard fit degrees of freedom."""
    return chi_squared / compute_degrees_of_freedom(n_data, n_params)


@dataclass
class FitResult:
    """Result of optimization for one cluster."""

    cluster_id: int
    params: Parameters
    residual: FloatArray
    cost: float
    nfev: int = 0
    njev: int = 0
    success: bool = False
    message: str = ""
    optimality: float = 0.0
    n_amplitude_params: int = 0
    uncertainty: UncertaintyResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chisqr(self) -> float:
        """Chi-squared value."""
        return compute_chi_squared(self.residual)

    @property
    def redchi(self) -> float:
        """Reduced chi-squared including nonlinear and amplitude parameters."""
        ndata = len(self.residual)
        nvarys = len(self.params.get_vary_names())
        n_total_fitted = nvarys + self.n_amplitude_params
        return compute_reduced_chi_squared(self.chisqr, ndata, n_total_fitted)


__all__ = [
    "FitResult",
    "compute_chi_squared",
    "compute_degrees_of_freedom",
    "compute_reduced_chi_squared",
]
