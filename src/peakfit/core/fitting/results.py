"""Fitting result classes and utilities."""

from dataclasses import dataclass

import numpy as np

from peakfit.core.fitting.parameters import Parameters
from peakfit.core.results.statistics import compute_chi_squared, compute_reduced_chi_squared


@dataclass
class FitResult:
    """Result of optimization."""

    params: Parameters
    residual: np.ndarray
    cost: float
    nfev: int
    njev: int
    success: bool
    message: str
    optimality: float = 0.0
    n_amplitude_params: int = 0  # Number of analytically computed amplitude parameters

    @property
    def chisqr(self) -> float:
        """Chi-squared value."""
        return compute_chi_squared(self.residual)

    @property
    def redchi(self) -> float:
        """Reduced chi-squared.

        Degrees of freedom includes both nonlinearly optimized parameters
        (vary=True) and analytically computed amplitude parameters.
        """
        ndata = len(self.residual)
        nvarys = len(self.params.get_vary_names())
        n_total_fitted = nvarys + self.n_amplitude_params
        return compute_reduced_chi_squared(self.chisqr, ndata, n_total_fitted)
