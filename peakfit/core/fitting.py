"""Optimized fitting engine using direct scipy.optimize.

This module provides a faster alternative to lmfit by using scipy.optimize.least_squares
directly, reducing overhead and improving performance.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from peakfit.clustering import Cluster
from peakfit.typing import FloatArray


@dataclass
class Parameter:
    """Single fitting parameter with bounds."""

    name: str
    value: float
    min: float = -np.inf
    max: float = np.inf
    vary: bool = True

    def __post_init__(self) -> None:
        """Validate parameter bounds."""
        if self.min > self.max:
            msg = f"Parameter {self.name}: min ({self.min}) > max ({self.max})"
            raise ValueError(msg)
        if not self.min <= self.value <= self.max:
            msg = f"Parameter {self.name}: value ({self.value}) outside bounds [{self.min}, {self.max}]"
            raise ValueError(msg)


@dataclass
class Parameters:
    """Collection of fitting parameters."""

    _params: dict[str, Parameter] = field(default_factory=dict)

    def add(
        self,
        name: str,
        value: float = 0.0,
        min: float = -np.inf,  # noqa: A002
        max: float = np.inf,  # noqa: A002
        vary: bool = True,
    ) -> None:
        """Add a parameter."""
        self._params[name] = Parameter(name, value, min, max, vary)

    def __getitem__(self, key: str) -> Parameter:
        """Get parameter by name."""
        return self._params[key]

    def __setitem__(self, key: str, value: Parameter) -> None:
        """Set parameter."""
        self._params[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if parameter exists."""
        return key in self._params

    def __iter__(self):
        """Iterate over parameter names."""
        return iter(self._params)

    def keys(self):
        """Get parameter names."""
        return self._params.keys()

    def values(self):
        """Get parameter objects."""
        return self._params.values()

    def items(self):
        """Get parameter name-value pairs."""
        return self._params.items()

    def valuesdict(self) -> dict[str, float]:
        """Get dictionary of parameter values."""
        return {name: param.value for name, param in self._params.items()}

    def update(self, other: "Parameters") -> None:
        """Update parameters from another Parameters object."""
        for name, param in other.items():
            self._params[name] = param

    def copy(self) -> "Parameters":
        """Create a copy of parameters."""
        new_params = Parameters()
        for name, param in self._params.items():
            new_params.add(name, param.value, param.min, param.max, param.vary)
        return new_params

    def get_vary_names(self) -> list[str]:
        """Get names of parameters that vary."""
        return [name for name, param in self._params.items() if param.vary]

    def get_vary_values(self) -> np.ndarray:
        """Get values of varying parameters as array."""
        return np.array([self._params[name].value for name in self.get_vary_names()])

    def get_vary_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get bounds for varying parameters."""
        names = self.get_vary_names()
        lower = np.array([self._params[name].min for name in names])
        upper = np.array([self._params[name].max for name in names])
        return lower, upper

    def set_vary_values(self, values: np.ndarray) -> None:
        """Set values of varying parameters from array."""
        names = self.get_vary_names()
        for name, value in zip(names, values, strict=True):
            self._params[name].value = value


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


def calculate_shapes_fast(params: Parameters, cluster: Cluster) -> FloatArray:
    """Calculate shapes for all peaks in cluster (optimized version)."""
    return np.array([peak.evaluate(cluster.positions, params) for peak in cluster.peaks])


def calculate_amplitudes_fast(shapes: FloatArray, data: FloatArray) -> FloatArray:
    """Calculate amplitudes via linear least squares."""
    return np.linalg.lstsq(shapes.T, data, rcond=None)[0]


def residuals_fast(x: np.ndarray, params: Parameters, cluster: Cluster, noise: float) -> np.ndarray:
    """Compute residuals for least_squares optimizer.

    Args:
        x: Array of varying parameter values
        params: Parameters object (will be updated in-place)
        cluster: Cluster being fitted
        noise: Noise level for normalization

    Returns:
        Flattened residual array normalized by noise
    """
    # Update parameters with current values
    params.set_vary_values(x)

    # Calculate shapes and amplitudes
    shapes = calculate_shapes_fast(params, cluster)
    amplitudes = calculate_amplitudes_fast(shapes, cluster.corrected_data)

    # Compute residuals
    fitted = shapes.T @ amplitudes
    residual = (cluster.corrected_data - fitted).ravel() / noise

    return residual


def fit_cluster_fast(
    params: Parameters,
    cluster: Cluster,
    noise: float,
    max_nfev: int = 1000,
    ftol: float = 1e-8,
    xtol: float = 1e-8,
    gtol: float = 1e-8,
    verbose: int = 0,
) -> FitResult:
    """Fit a single cluster using scipy.optimize.least_squares.

    Args:
        params: Initial parameters
        cluster: Cluster to fit
        noise: Noise level
        max_nfev: Maximum number of function evaluations
        ftol: Tolerance for termination by change of cost function
        xtol: Tolerance for termination by change of variables
        gtol: Tolerance for termination by gradient norm
        verbose: Verbosity level (0=silent, 1=termination, 2=iteration)

    Returns:
        FitResult containing optimized parameters and fit statistics
    """
    # Get initial values and bounds for varying parameters
    x0 = params.get_vary_values()
    lower, upper = params.get_vary_bounds()

    # Run optimization
    result = least_squares(
        residuals_fast,
        x0,
        args=(params, cluster, noise),
        bounds=(lower, upper),
        method="trf",  # Trust Region Reflective - good for bounded problems
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        max_nfev=max_nfev,
        verbose=verbose,
    )

    # Update parameters with final values
    params.set_vary_values(result.x)

    return FitResult(
        params=params,
        residual=result.fun,
        cost=result.cost,
        nfev=result.nfev,
        njev=result.njev if hasattr(result, "njev") else 0,
        success=result.success,
        message=result.message,
        optimality=result.optimality if hasattr(result, "optimality") else 0.0,
    )


def fit_clusters_sequential(
    clusters: Sequence[Cluster],
    params_all: Parameters,
    noise: float,
    refine_iterations: int = 1,
    fixed: bool = False,
    verbose: int = 0,
) -> Parameters:
    """Fit all clusters sequentially with refinement.

    Args:
        clusters: List of clusters to fit
        params_all: Global parameters (updated in place)
        noise: Noise level
        refine_iterations: Number of refinement passes
        fixed: Whether to fix peak positions
        verbose: Verbosity level

    Returns:
        Updated global parameters
    """
    from peakfit.computing import update_cluster_corrections
    from peakfit.peak import create_params

    for iteration in range(refine_iterations + 1):
        if iteration > 0:
            # Update corrections for cross-talk
            update_cluster_corrections(params_all, clusters)

        for cluster in clusters:
            # Create parameters for this cluster
            cluster_params = create_params(cluster.peaks, fixed=fixed)

            # Merge with global parameters
            for key in cluster_params:
                if key in params_all:
                    cluster_params[key] = params_all[key]

            # Fit cluster
            result = fit_cluster_fast(cluster_params, cluster, noise, verbose=verbose)

            # Update global parameters
            params_all.update(result.params)

    return params_all
