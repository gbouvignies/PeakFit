"""Optimized fitting engine using direct scipy.optimize.

This module provides a faster alternative to lmfit by using scipy.optimize.least_squares
directly, reducing overhead and improving performance.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import least_squares

from peakfit.typing import FloatArray

if TYPE_CHECKING:
    from peakfit.clustering import Cluster


class ParameterType(str, Enum):
    """Types of NMR fitting parameters."""

    POSITION = "position"  # Peak center in points
    FWHM = "fwhm"  # Full width at half maximum (Hz)
    FRACTION = "fraction"  # Mixing parameter (0-1)
    PHASE = "phase"  # Phase correction (degrees)
    JCOUPLING = "jcoupling"  # J-coupling constant (Hz)
    AMPLITUDE = "amplitude"  # Peak amplitude
    GENERIC = "generic"  # Other parameters


# Default bounds for NMR parameter types
_DEFAULT_BOUNDS: dict[ParameterType, tuple[float, float]] = {
    ParameterType.POSITION: (-np.inf, np.inf),  # Set dynamically from spectrum
    ParameterType.FWHM: (0.1, 200.0),  # Typical NMR linewidths
    ParameterType.FRACTION: (0.0, 1.0),  # Mixing fractions
    ParameterType.PHASE: (-180.0, 180.0),  # Phase in degrees
    ParameterType.JCOUPLING: (0.0, 20.0),  # Typical J-couplings
    ParameterType.AMPLITUDE: (0.0, np.inf),  # Positive amplitudes
    ParameterType.GENERIC: (-np.inf, np.inf),
}


@dataclass
class Parameter:
    """Single NMR fitting parameter with bounds and metadata.

    Designed specifically for NMR lineshape fitting with support for
    different parameter types (position, FWHM, phase, etc.).
    """

    name: str
    value: float
    min: float = -np.inf
    max: float = np.inf
    vary: bool = True
    param_type: ParameterType = ParameterType.GENERIC
    unit: str = ""  # Optional unit string (e.g., "Hz", "ppm", "deg")

    def __post_init__(self) -> None:
        """Validate parameter bounds and apply type-specific defaults."""
        # Apply type-specific defaults if bounds not explicitly set
        if self.min == -np.inf and self.max == np.inf and self.param_type != ParameterType.GENERIC:
            default_min, default_max = _DEFAULT_BOUNDS[self.param_type]
            self.min = default_min
            self.max = default_max

        if self.min > self.max:
            msg = f"Parameter {self.name}: min ({self.min}) > max ({self.max})"
            raise ValueError(msg)
        if not self.min <= self.value <= self.max:
            msg = f"Parameter {self.name}: value ({self.value}) outside bounds [{self.min}, {self.max}]"
            raise ValueError(msg)

    def __repr__(self) -> str:
        """String representation of parameter."""
        vary_str = "vary" if self.vary else "fixed"
        min_str = f"{self.min:.4g}" if self.min > -1e10 else "-inf"
        max_str = f"{self.max:.4g}" if self.max < 1e10 else "inf"
        unit_str = f" {self.unit}" if self.unit else ""
        return f"<Parameter {self.name}={self.value:.6g}{unit_str} [{min_str}, {max_str}] ({vary_str})>"

    def is_at_boundary(self, tol: float = 1e-6) -> bool:
        """Check if parameter is at or near its boundary.

        Args:
            tol: Tolerance for boundary check

        Returns:
            True if at boundary
        """
        at_min = abs(self.value - self.min) < tol * (1 + abs(self.value))
        at_max = abs(self.value - self.max) < tol * (1 + abs(self.value))
        return at_min or at_max

    def relative_position(self) -> float:
        """Get the relative position of value within bounds (0 to 1).

        Returns:
            0.0 if at min, 1.0 if at max, 0.5 if centered
        """
        if self.max == self.min:
            return 0.5
        if np.isinf(self.min) or np.isinf(self.max):
            return 0.5
        return (self.value - self.min) / (self.max - self.min)


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
        param_type: ParameterType = ParameterType.GENERIC,
        unit: str = "",
    ) -> None:
        """Add a parameter.

        Args:
            name: Parameter name
            value: Initial value
            min: Lower bound
            max: Upper bound
            vary: Whether parameter varies during fitting
            param_type: Type of NMR parameter (affects default bounds)
            unit: Unit string (e.g., "Hz", "ppm", "deg")
        """
        self._params[name] = Parameter(name, value, min, max, vary, param_type, unit)

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
            new_params.add(
                name, param.value, param.min, param.max, param.vary,
                param.param_type, param.unit
            )
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

    def __len__(self) -> int:
        """Return number of parameters."""
        return len(self._params)

    def __repr__(self) -> str:
        """String representation of parameters collection."""
        n_total = len(self._params)
        n_vary = len(self.get_vary_names())
        return f"<Parameters: {n_total} total, {n_vary} varying>"

    def summary(self) -> str:
        """Get a formatted summary of all parameters.

        Returns:
            Multi-line string with parameter details
        """
        lines = ["Parameters:", "=" * 60]
        for name in self._params:
            param = self._params[name]
            vary_str = "vary" if param.vary else "fixed"
            min_str = f"{param.min:.4g}" if param.min > -1e10 else "-inf"
            max_str = f"{param.max:.4g}" if param.max < 1e10 else "inf"
            lines.append(f"  {name:20s} = {param.value:12.6g} [{min_str:>10s}, {max_str:<10s}] ({vary_str})")
        lines.append("=" * 60)
        return "\n".join(lines)

    def get_boundary_params(self) -> list[str]:
        """Get names of parameters that are at their boundaries.

        Returns:
            List of parameter names at boundaries
        """
        return [name for name, param in self._params.items() if param.vary and param.is_at_boundary()]

    def freeze(self, names: list[str] | None = None) -> None:
        """Set parameters to not vary (freeze them).

        Args:
            names: List of parameter names to freeze. If None, freeze all.
        """
        if names is None:
            names = list(self._params.keys())
        for name in names:
            if name in self._params:
                self._params[name].vary = False

    def unfreeze(self, names: list[str] | None = None) -> None:
        """Set parameters to vary (unfreeze them).

        Args:
            names: List of parameter names to unfreeze. If None, unfreeze all.
        """
        if names is None:
            names = list(self._params.keys())
        for name in names:
            if name in self._params:
                self._params[name].vary = True


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


def calculate_shapes_fast(params: Parameters, cluster: "Cluster") -> FloatArray:
    """Calculate shapes for all peaks in cluster (optimized version)."""
    return np.array([peak.evaluate(cluster.positions, params) for peak in cluster.peaks])


def calculate_amplitudes_fast(shapes: FloatArray, data: FloatArray) -> FloatArray:
    """Calculate amplitudes via linear least squares."""
    return np.linalg.lstsq(shapes.T, data, rcond=None)[0]


def residuals_fast(x: np.ndarray, params: Parameters, cluster: "Cluster", noise: float) -> np.ndarray:
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
    cluster: "Cluster",
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
    clusters: "Sequence[Cluster]",
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
