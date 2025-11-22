"""Fitting result classes and utilities."""

from dataclasses import dataclass

import numpy as np

from peakfit.fitting.parameters import Parameters


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

    @property
    def chisqr(self) -> float:
        """Chi-squared value."""
        return float(np.sum(self.residual**2))

    @property
    def redchi(self) -> float:
        """Reduced chi-squared."""
        ndata = len(self.residual)
        nvarys = len(self.params.get_vary_names())
        if ndata > nvarys:
            return self.chisqr / (ndata - nvarys)
        return self.chisqr
