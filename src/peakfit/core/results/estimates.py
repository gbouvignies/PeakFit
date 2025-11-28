"""Parameter and amplitude estimate models.

This module defines dataclasses for representing fitted parameter values
with their uncertainties, supporting both symmetric (least-squares) and
asymmetric (MCMC posterior) error estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from peakfit.core.shared.typing import FloatArray


class ParameterCategory(str, Enum):
    """Categories of fitting parameters for grouping and formatting.

    Attributes
    ----------
        LINESHAPE: Shape parameters (position, FWHM, fraction, phase)
        AMPLITUDE: Peak intensities per plane
        EXCHANGE: Exchange dynamics parameters (kex, pb, dw)
        RELAXATION: Relaxation rates (R1, R2, R1rho)
        GLOBAL: Shared parameters across residues
    """

    LINESHAPE = "lineshape"
    AMPLITUDE = "amplitude"
    EXCHANGE = "exchange"
    RELAXATION = "relaxation"
    GLOBAL = "global"


@dataclass(slots=True)
class ParameterEstimate:
    """A fitted parameter value with uncertainties.

    This dataclass represents a single parameter estimate with support for:
    - Point estimate (value) with symmetric uncertainty (std_error)
    - Asymmetric confidence intervals from MCMC posteriors
    - Bounds information for detecting boundary issues
    - Full posterior samples for custom analysis

    Attributes
    ----------
        name: Parameter identifier (e.g., "G23N_x0", "peak1_fwhm")
        value: Best-fit or MAP estimate
        std_error: Standard deviation (symmetric uncertainty)
        unit: Physical unit string (e.g., "Hz", "ppm", "s^-1")
        category: Parameter category for grouping

        ci_68_lower: Lower bound of 68% credible interval (1 sigma)
        ci_68_upper: Upper bound of 68% credible interval (1 sigma)
        ci_95_lower: Lower bound of 95% credible interval (2 sigma)
        ci_95_upper: Upper bound of 95% credible interval (2 sigma)

        min_bound: Lower fitting bound (for boundary detection)
        max_bound: Upper fitting bound (for boundary detection)
        is_fixed: Whether parameter was held fixed during fitting
        is_global: Whether parameter is shared across residues/clusters

        posterior_samples: Full MCMC samples if available (flattened)

    Example:
        >>> param = ParameterEstimate(
        ...     name="G23N_fwhm",
        ...     value=25.3,
        ...     std_error=1.2,
        ...     unit="Hz",
        ...     category=ParameterCategory.LINESHAPE,
        ...     ci_68_lower=24.1,
        ...     ci_68_upper=26.5,
        ... )
        >>> param.is_at_boundary()  # Check for boundary issues
        False
        >>> param.relative_error  # Get relative uncertainty
        0.0474
    """

    # Core identification
    name: str
    value: float
    std_error: float
    unit: str = ""
    category: ParameterCategory = ParameterCategory.LINESHAPE

    # Asymmetric confidence intervals (from MCMC)
    ci_68_lower: float | None = None
    ci_68_upper: float | None = None
    ci_95_lower: float | None = None
    ci_95_upper: float | None = None

    # Bounds and constraints
    min_bound: float = field(default_factory=lambda: -np.inf)
    max_bound: float = field(default_factory=lambda: np.inf)
    is_fixed: bool = False
    is_global: bool = False

    # Full posterior (optional, for detailed analysis)
    posterior_samples: FloatArray | None = None

    @property
    def has_asymmetric_error(self) -> bool:
        """Check if asymmetric confidence intervals are available."""
        return self.ci_68_lower is not None and self.ci_68_upper is not None

    @property
    def error_lower(self) -> float:
        """Lower error bar (value - ci_68_lower, or std_error)."""
        if self.ci_68_lower is not None:
            return self.value - self.ci_68_lower
        return self.std_error

    @property
    def error_upper(self) -> float:
        """Upper error bar (ci_68_upper - value, or std_error)."""
        if self.ci_68_upper is not None:
            return self.ci_68_upper - self.value
        return self.std_error

    @property
    def relative_error(self) -> float | None:
        """Relative uncertainty (std_error / |value|), or None if value is zero."""
        if abs(self.value) < 1e-15:
            return None
        return abs(self.std_error / self.value)

    def is_at_boundary(self, tolerance: float = 1e-6) -> bool:
        """Check if value is at or near fitting bounds."""
        if np.isinf(self.min_bound) and np.isinf(self.max_bound):
            return False
        at_min = abs(self.value - self.min_bound) < tolerance
        at_max = abs(self.value - self.max_bound) < tolerance
        return at_min or at_max

    @property
    def is_problematic(self) -> bool:
        """Check if parameter has potential issues.

        A parameter is flagged as problematic if:
        - It is at a fitting boundary
        - Relative error exceeds 50%
        - Standard error is zero or negative (not computed)
        """
        if self.is_fixed:
            return False
        if self.is_at_boundary():
            return True
        if self.std_error <= 0:
            return True
        rel_err = self.relative_error
        return bool(rel_err is not None and rel_err > 0.5)

    def format_value(self, precision: int = 6) -> str:
        """Format value with uncertainty for display.

        Args:
            precision: Number of decimal places

        Returns
        -------
            Formatted string like "25.300 ± 1.200" or "25.300 +1.200/-1.100"
        """
        if self.has_asymmetric_error:
            return (
                f"{self.value:.{precision}f} "
                f"+{self.error_upper:.{precision}f}/-{self.error_lower:.{precision}f}"
            )
        return f"{self.value:.{precision}f} ± {self.std_error:.{precision}f}"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "name": self.name,
            "value": self.value,
            "std_error": self.std_error,
            "unit": self.unit,
            "category": self.category.value,
            "is_fixed": self.is_fixed,
            "is_global": self.is_global,
        }

        # Add optional fields if present
        if self.ci_68_lower is not None:
            result["ci_68"] = [self.ci_68_lower, self.ci_68_upper]
        if self.ci_95_lower is not None:
            result["ci_95"] = [self.ci_95_lower, self.ci_95_upper]
        if not np.isinf(self.min_bound):
            result["min_bound"] = self.min_bound
        if not np.isinf(self.max_bound):
            result["max_bound"] = self.max_bound

        # Don't include posterior_samples in JSON (too large)
        return result


@dataclass(slots=True)
class AmplitudeEstimate:
    """Fitted amplitude (intensity) for a single peak at a single plane.

    Amplitudes are separated from lineshape parameters because:
    - They are computed via linear least-squares given lineshape params
    - There are typically many amplitudes (n_peaks × n_planes)
    - They have different display/export requirements

    Attributes
    ----------
        peak_name: Peak identifier
        plane_index: Index in the Z-dimension (0-based)
        z_value: Physical value in Z-dimension (e.g., relaxation delay, B1 offset)
        value: Fitted amplitude
        std_error: Standard error from linear propagation or MCMC
        ci_68_lower: Lower 68% CI from MCMC
        ci_68_upper: Upper 68% CI from MCMC
    """

    peak_name: str
    plane_index: int
    z_value: float | None
    value: float
    std_error: float
    ci_68_lower: float | None = None
    ci_68_upper: float | None = None

    @property
    def has_asymmetric_error(self) -> bool:
        """Check if asymmetric confidence intervals are available."""
        return self.ci_68_lower is not None and self.ci_68_upper is not None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "peak_name": self.peak_name,
            "plane_index": self.plane_index,
            "value": self.value,
            "std_error": self.std_error,
        }
        if self.z_value is not None:
            result["z_value"] = self.z_value
        if self.ci_68_lower is not None:
            result["ci_68"] = [self.ci_68_lower, self.ci_68_upper]
        return result


@dataclass(slots=True)
class ClusterEstimates:
    """Collection of parameter estimates for a single cluster.

    A cluster is a group of overlapping peaks fitted together.
    This dataclass groups all parameters and amplitudes for one cluster.

    Attributes
    ----------
        cluster_id: Unique cluster identifier (0-based index)
        peak_names: List of peak names in this cluster
        lineshape_params: Lineshape parameter estimates
        amplitudes: Amplitude estimates per peak per plane
        correlation_matrix: Parameter correlation matrix (lineshape only)
        correlation_param_names: Names corresponding to correlation matrix rows/cols
    """

    cluster_id: int
    peak_names: list[str]
    lineshape_params: list[ParameterEstimate]
    amplitudes: list[AmplitudeEstimate] = field(default_factory=list)
    correlation_matrix: FloatArray | None = None
    correlation_param_names: list[str] = field(default_factory=list)

    @property
    def n_peaks(self) -> int:
        """Number of peaks in cluster."""
        return len(self.peak_names)

    @property
    def n_lineshape_params(self) -> int:
        """Number of lineshape parameters."""
        return len(self.lineshape_params)

    @property
    def n_planes(self) -> int:
        """Number of planes (inferred from amplitudes)."""
        if not self.amplitudes:
            return 0
        return max(a.plane_index for a in self.amplitudes) + 1

    def get_amplitudes_for_peak(self, peak_name: str) -> list[AmplitudeEstimate]:
        """Get all amplitudes for a specific peak."""
        return [a for a in self.amplitudes if a.peak_name == peak_name]

    def get_strong_correlations(self, threshold: float = 0.7) -> list[tuple[str, str, float]]:
        """Find pairs of strongly correlated parameters.

        Args:
            threshold: Minimum absolute correlation to report

        Returns
        -------
            List of (param1, param2, correlation) tuples
        """
        if self.correlation_matrix is None or len(self.correlation_param_names) < 2:
            return []

        pairs = []
        n = len(self.correlation_param_names)
        for i in range(n):
            for j in range(i + 1, n):
                corr = float(self.correlation_matrix[i, j])
                if abs(corr) >= threshold:
                    pairs.append(
                        (
                            self.correlation_param_names[i],
                            self.correlation_param_names[j],
                            corr,
                        )
                    )
        return pairs

    def get_problematic_params(self) -> list[ParameterEstimate]:
        """Get list of parameters flagged as problematic."""
        return [p for p in self.lineshape_params if p.is_problematic]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "cluster_id": self.cluster_id,
            "peak_names": self.peak_names,
            "lineshape_parameters": [p.to_dict() for p in self.lineshape_params],
            "amplitudes": [a.to_dict() for a in self.amplitudes],
        }

        if self.correlation_matrix is not None:
            result["correlation"] = {
                "parameter_names": self.correlation_param_names,
                "matrix": self.correlation_matrix.tolist(),
            }

        return result
