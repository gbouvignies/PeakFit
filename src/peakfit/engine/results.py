"""Result models for PeakFit output system.

This module provides unified data models for representing fitting results,
statistics, and diagnostics in a serialization-friendly format.

Design Principles:
    - All numeric arrays use numpy for computation, but serialize to lists/base64
    - Units are always explicitly specified in field metadata
    - Asymmetric uncertainties (from MCMC posteriors) are first-class citizens
    - Clear separation between point estimates and full distributions
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from importlib import metadata
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    __version__ = metadata.version("peakfit")
except metadata.PackageNotFoundError:
    __version__ = "unknown"

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.engine.algorithms.mcmc import UncertaintyResult
    from peakfit.engine.domain.cluster import Cluster
    from peakfit.engine.domain.param_id import ParameterId
    from peakfit.engine.domain.params_scalar import Parameters
    from peakfit.engine.domain.peaks import Peak
    from peakfit.shared.typing import FloatArray


# =============================================================================
# Constants
# =============================================================================

_ZERO_VALUE_THRESHOLD = 1e-15
_POOR_RELATIVE_ERROR_THRESHOLD = 0.5

_RHAT_EXCELLENT_THRESHOLD = 1.01
_RHAT_ACCEPTABLE_THRESHOLD = 1.05

_ESS_EXCELLENT_THRESHOLD = 10000
_ESS_GOOD_THRESHOLD = 1000
_ESS_ACCEPTABLE_THRESHOLD = 400
_ESS_MARGINAL_THRESHOLD = 100

_RECOMMENDED_ESS_PER_CHAIN = 100
_LOW_ESS_PER_CHAIN = 10


# =============================================================================
# Enums
# =============================================================================


class ParameterCategory(StrEnum):
    """Categories of fitting parameters for grouping and formatting.

    Attributes:
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


class ConvergenceStatus(StrEnum):
    """Convergence status categories.

    Based on recommendations from:
    - Gelman & Rubin (1992) for R-hat
    - Vehtari et al. (2021) for ESS
    - Kruschke (2021) BARG guidelines
    """

    EXCELLENT = "excellent"  # R-hat ≤ 1.01, ESS ≥ 10000
    GOOD = "good"  # R-hat ≤ 1.01, ESS ≥ 1000
    ACCEPTABLE = "acceptable"  # R-hat ≤ 1.05, ESS ≥ 400
    MARGINAL = "marginal"  # R-hat ≤ 1.05, ESS ≥ 100
    POOR = "poor"  # R-hat > 1.05 or ESS < 100
    UNKNOWN = "unknown"  # Diagnostics not computed


# =============================================================================
# Statistics Functions
# =============================================================================


def compute_chi_squared(residuals: FloatArray) -> float:
    """Compute chi-squared (sum of squared residuals).

    This is the single source of truth for chi-squared calculation.

    Args:
        residuals: Normalized residuals (data - model) / noise

    Returns:
    -------
        Chi-squared value (sum of residuals squared)
    """
    return float(np.sum(residuals**2))


def compute_degrees_of_freedom(n_data: int, n_params: int) -> int:
    """Compute degrees of freedom for statistical calculations.

    This is the single source of truth for DOF calculation.

    Args:
        n_data: Number of data points
        n_params: Total number of fitted parameters (lineshape + amplitudes)

    Returns:
    -------
        Degrees of freedom, minimum of 1 to avoid division by zero
    """
    return max(1, n_data - n_params)


def compute_reduced_chi_squared(
    chi_squared: float,
    n_data: int,
    n_params: int,
) -> float:
    """Compute reduced chi-squared with proper degrees of freedom.

    This is the single source of truth for reduced chi-squared calculation.
    The degrees of freedom is n_data - n_params, where n_params should include
    all fitted parameters (both nonlinearly optimized lineshape parameters
    and analytically computed amplitude parameters).

    Args:
        chi_squared: Sum of squared normalized residuals
        n_data: Number of data points
        n_params: Total number of fitted parameters (lineshape + amplitudes)

    Returns:
    -------
        Reduced chi-squared value (chi_squared / dof)
    """
    dof = compute_degrees_of_freedom(n_data, n_params)
    return chi_squared / dof


# =============================================================================
# Statistics Dataclasses
# =============================================================================


@dataclass(slots=True)
class ResidualStatistics:
    """Statistics computed from fit residuals.

    Attributes:
    ----------
        raw_residuals: Unweighted residuals (data - model)
        normalized_residuals: Residuals divided by noise level
        n_points: Number of data points
        n_params: Number of varying parameters
        noise_level: Noise estimate used for normalization
    """

    raw_residuals: FloatArray | None = None
    normalized_residuals: FloatArray | None = None
    n_points: int = 0
    n_params: int = 0
    noise_level: float = 1.0

    @property
    def dof(self) -> int:
        """Degrees of freedom (n_points - n_params)."""
        return compute_degrees_of_freedom(self.n_points, self.n_params)

    @property
    def sum_squared(self) -> float:
        """Sum of squared normalized residuals (chi-squared)."""
        if self.normalized_residuals is None:
            return 0.0
        return compute_chi_squared(self.normalized_residuals)

    @property
    def rms(self) -> float:
        """Root mean square of raw residuals."""
        if self.raw_residuals is None or len(self.raw_residuals) == 0:
            return 0.0
        return float(np.sqrt(np.mean(self.raw_residuals**2)))

    @property
    def mean(self) -> float:
        """Mean of raw residuals (should be ~0 for good fit)."""
        if self.raw_residuals is None or len(self.raw_residuals) == 0:
            return 0.0
        return float(np.mean(self.raw_residuals))

    @property
    def std(self) -> float:
        """Standard deviation of raw residuals."""
        if self.raw_residuals is None or len(self.raw_residuals) == 0:
            return 0.0
        return float(np.std(self.raw_residuals))


@dataclass(slots=True)
class FitStatistics:
    """Comprehensive fit quality statistics.

    Attributes:
    ----------
        chi_squared: Chi-squared value (sum of squared normalized residuals)
        reduced_chi_squared: Chi-squared divided by degrees of freedom
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        log_likelihood: Log-likelihood value
        n_data: Number of data points
        n_params: Number of varying parameters
        residuals: Detailed residual statistics
        fit_converged: Whether optimizer reported convergence
        n_function_evals: Number of objective function evaluations
        fit_message: Optimizer status message
    """

    chi_squared: float = 0.0
    reduced_chi_squared: float = 0.0
    aic: float | None = None
    bic: float | None = None
    log_likelihood: float | None = None
    n_data: int = 0
    n_params: int = 0
    residuals: ResidualStatistics = field(default_factory=ResidualStatistics)
    fit_converged: bool = True
    n_function_evals: int = 0
    fit_message: str = ""

    @property
    def dof(self) -> int:
        """Degrees of freedom."""
        return compute_degrees_of_freedom(self.n_data, self.n_params)


@dataclass(slots=True)
class ParameterEstimate:
    """A fitted parameter value with uncertainties.

    This dataclass represents a single parameter estimate with support for:
    - Point estimate (value) with symmetric uncertainty (std_error)
    - Asymmetric confidence intervals from MCMC posteriors
    - Bounds information for detecting boundary issues
    - Full posterior samples for custom analysis

    Attributes:
    ----------
        name: Canonical parameter identifier (e.g., "G23N.F2.cs", "peak1.F1.lw")
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
        ...     name="G23N.F2.lw",
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

    # Structured parameter identifier (for consistent naming)
    param_id: ParameterId | None = None

    @property
    def has_asymmetric_error(self) -> bool:
        """Check if asymmetric confidence intervals are available."""
        return self.ci_68_lower is not None and self.ci_68_upper is not None

    @property
    def relative_error(self) -> float | None:
        """Relative uncertainty (std_error / |value|), or None if value is zero."""
        if abs(self.value) < _ZERO_VALUE_THRESHOLD:
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
        return bool(rel_err is not None and rel_err > _POOR_RELATIVE_ERROR_THRESHOLD)


@dataclass(slots=True)
class AmplitudeEstimate:
    """Fitted amplitude (intensity) for a single peak at a single plane.

    Amplitudes are separated from lineshape parameters because:
    - They are computed via linear least-squares given lineshape params
    - There are typically many amplitudes (n_peaks × n_series)
    - They have different display/export requirements

    Attributes:
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


@dataclass(slots=True)
class ClusterEstimates:
    """Collection of parameter estimates for a single cluster.

    A cluster is a group of overlapping peaks fitted together.
    This dataclass groups all parameters and amplitudes for one cluster.

    Attributes:
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


# =============================================================================
# MCMC Diagnostics
# =============================================================================


@dataclass(slots=True)
class ParameterDiagnostic:
    """Convergence diagnostics for a single MCMC parameter.

    Attributes:
    ----------
        name: Parameter name
        rhat: R-hat (potential scale reduction factor)
        ess_bulk: Effective sample size for bulk of distribution
        ess_tail: Effective sample size for tails
        status: Overall convergence status
        warnings: List of specific warnings for this parameter
    """

    name: str
    rhat: float | None = None
    ess_bulk: float | None = None
    ess_tail: float | None = None
    status: ConvergenceStatus = ConvergenceStatus.UNKNOWN
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_values(
        cls,
        name: str,
        rhat: float | None,
        ess_bulk: float | None,
        ess_tail: float | None = None,
        n_chains: int = 4,
    ) -> ParameterDiagnostic:
        """Create diagnostic with automatic status determination.

        Args:
            name: Parameter name
            rhat: R-hat value
            ess_bulk: Bulk ESS
            ess_tail: Tail ESS (optional)
            n_chains: Number of chains (for ESS thresholds)

        Returns:
        -------
            ParameterDiagnostic with computed status
        """
        warnings = []
        status = ConvergenceStatus.UNKNOWN

        if rhat is None or ess_bulk is None:
            return cls(name=name, rhat=rhat, ess_bulk=ess_bulk, ess_tail=ess_tail)

        # Determine status based on BARG guidelines
        if rhat <= _RHAT_EXCELLENT_THRESHOLD and ess_bulk >= _ESS_EXCELLENT_THRESHOLD:
            status = ConvergenceStatus.EXCELLENT
        elif rhat <= _RHAT_EXCELLENT_THRESHOLD and ess_bulk >= _ESS_GOOD_THRESHOLD:
            status = ConvergenceStatus.GOOD
        elif rhat <= _RHAT_ACCEPTABLE_THRESHOLD and ess_bulk >= _ESS_ACCEPTABLE_THRESHOLD:
            status = ConvergenceStatus.ACCEPTABLE
        elif rhat <= _RHAT_ACCEPTABLE_THRESHOLD and ess_bulk >= _ESS_MARGINAL_THRESHOLD:
            status = ConvergenceStatus.MARGINAL
        else:
            status = ConvergenceStatus.POOR

        # Generate warnings
        if rhat > _RHAT_EXCELLENT_THRESHOLD:
            warnings.append(
                f"R-hat = {rhat:.4f} (should be ≤ {_RHAT_EXCELLENT_THRESHOLD:.2f}). "
                "Chains have not mixed well."
            )
        if rhat > _RHAT_ACCEPTABLE_THRESHOLD:
            warnings.append(
                f"R-hat = {rhat:.4f} is very high (> {_RHAT_ACCEPTABLE_THRESHOLD:.2f}). "
                "Results should not be trusted."
            )

        recommended_ess = _RECOMMENDED_ESS_PER_CHAIN * n_chains
        if ess_bulk < recommended_ess:
            warnings.append(
                f"ESS_bulk = {ess_bulk:.0f} (recommended ≥ {recommended_ess:.0f}). "
                "Consider more samples."
            )
        if ess_bulk < _LOW_ESS_PER_CHAIN * n_chains:
            warnings.append(
                f"ESS_bulk = {ess_bulk:.0f} is very low. Posterior estimates are highly uncertain."
            )

        return cls(
            name=name,
            rhat=rhat,
            ess_bulk=ess_bulk,
            ess_tail=ess_tail,
            status=status,
            warnings=warnings,
        )


@dataclass(slots=True)
class MCMCDiagnostics:
    """Complete MCMC diagnostics for a cluster or analysis.

    Attributes:
    ----------
        n_chains: Number of MCMC chains
        n_samples: Number of samples per chain (after burn-in)
        burn_in: Number of burn-in samples discarded
        parameter_diagnostics: Per-parameter diagnostics
        overall_status: Worst status among all parameters
        total_samples: Total effective samples (n_chains * n_samples)
        burn_in_method: How burn-in was determined
        burn_in_details: Additional burn-in determination info
    """

    n_chains: int
    n_samples: int
    burn_in: int
    parameter_diagnostics: list[ParameterDiagnostic] = field(default_factory=list)
    overall_status: ConvergenceStatus = ConvergenceStatus.UNKNOWN
    burn_in_method: str = "manual"  # "manual", "auto", "geweke", "ess"
    burn_in_details: dict[str, Any] = field(default_factory=dict)

    @property
    def total_samples(self) -> int:
        """Total number of post-burn-in samples across all chains."""
        return self.n_chains * self.n_samples

    @property
    def converged(self) -> bool:
        """Check if MCMC has converged (at least ACCEPTABLE status)."""
        return self.overall_status in (
            ConvergenceStatus.EXCELLENT,
            ConvergenceStatus.GOOD,
            ConvergenceStatus.ACCEPTABLE,
        )

    @property
    def all_warnings(self) -> list[str]:
        """Collect all warnings from all parameters."""
        warnings = []
        for diag in self.parameter_diagnostics:
            warnings.extend(diag.warnings)
        return warnings

    def update_overall_status(self) -> None:
        """Recompute overall status from parameter diagnostics."""
        if not self.parameter_diagnostics:
            self.overall_status = ConvergenceStatus.UNKNOWN
            return

        # Overall status is the worst among all parameters
        status_order = [
            ConvergenceStatus.EXCELLENT,
            ConvergenceStatus.GOOD,
            ConvergenceStatus.ACCEPTABLE,
            ConvergenceStatus.MARGINAL,
            ConvergenceStatus.POOR,
        ]

        worst_idx = 0
        for diag in self.parameter_diagnostics:
            if diag.status in status_order:
                idx = status_order.index(diag.status)
                worst_idx = max(worst_idx, idx)

        self.overall_status = status_order[worst_idx]

    def get_problematic_parameters(self) -> list[ParameterDiagnostic]:
        """Get parameters with POOR or MARGINAL convergence."""
        return [
            d
            for d in self.parameter_diagnostics
            if d.status in (ConvergenceStatus.POOR, ConvergenceStatus.MARGINAL)
        ]


# =============================================================================
# Fit Results
# =============================================================================


@dataclass
class RunMetadata:
    """Metadata about the fitting run for reproducibility.

    Attributes:
    ----------
        timestamp: When the analysis was run (ISO 8601)
        software_version: PeakFit version string
        git_commit: Git commit hash if in repository
        python_version: Python interpreter version
        platform: Operating system platform
        input_files: Dictionary of input file paths and checksums
        configuration: Complete configuration used for fitting
        command_line: Command line arguments if available
        run_duration_seconds: Total run time
    """

    timestamp: str = ""
    software_version: str = ""
    git_commit: str | None = None
    python_version: str = ""
    platform: str = ""
    input_files: dict[str, dict[str, str]] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)
    command_line: str = ""
    run_duration_seconds: float | None = None

    @classmethod
    def capture(cls, config: dict[str, Any] | None = None) -> RunMetadata:
        """Capture current environment metadata.

        Args:
            config: Configuration dictionary to include

        Returns:
        -------
            RunMetadata with populated fields
        """
        # Try to get git commit
        git_commit = None
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if result.returncode == 0:
                git_commit = result.stdout.strip()[:12]
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            pass

        return cls(
            timestamp=datetime.now(UTC).isoformat(),
            software_version=__version__,
            git_commit=git_commit,
            python_version=sys.version,
            platform=platform.platform(),
            configuration=config or {},
        )

    def add_input_file(self, name: str, path: Path) -> None:
        """Add an input file with its checksum.

        Args:
            name: Descriptive name for the file
            path: Path to the file
        """
        if path.exists():
            checksum = _compute_file_checksum(path)
            self.input_files[name] = {
                "path": str(path.name),  # Relative, not absolute
                "checksum_sha256": checksum,
            }


@dataclass
class FitResult:
    """Result of optimization for a single cluster.

    Encapsulates the optimizer output used by fit orchestration.
    """

    params: Parameters
    residual: FloatArray
    cost: float
    nfev: int = 0
    njev: int = 0
    success: bool = False
    message: str = ""
    optimality: float = 0.0
    n_amplitude_params: int = 0

    # Optional MCMC results
    uncertainty: UncertaintyResult | None = None

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

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
        # Avoid circular dependencies by checking vary names dynamically if needed,
        # but params is a full Parameters object.
        nvarys = len(self.params.get_vary_names())
        n_total_fitted = nvarys + self.n_amplitude_params
        return compute_reduced_chi_squared(self.chisqr, ndata, n_total_fitted)


@dataclass
class FitResults:
    """Complete results from a fitting run.

    This is the top-level result object that aggregates all outputs
    from fitting: parameter estimates, statistics, and diagnostics
    for all clusters.

    Attributes:
    ----------
        metadata: Run metadata for reproducibility
        clusters: Per-cluster parameter estimates
        statistics: Per-cluster fit statistics
        global_statistics: Overall fit statistics (if applicable)
        mcmc_diagnostics: Per-cluster MCMC diagnostics (if MCMC used)
        z_values: Z-dimension values (e.g., relaxation delays)
    """

    metadata: RunMetadata = field(default_factory=RunMetadata)
    method: str = "least_squares"
    clusters: list[ClusterEstimates] = field(default_factory=list)
    statistics: list[FitStatistics] = field(default_factory=list)
    global_statistics: FitStatistics | None = None
    mcmc_diagnostics: list[MCMCDiagnostics] = field(default_factory=list)
    z_values: FloatArray | None = None

    @property
    def n_clusters(self) -> int:
        """Number of fitted clusters."""
        return len(self.clusters)

    @property
    def n_peaks(self) -> int:
        """Total number of peaks across all clusters."""
        return sum(c.n_peaks for c in self.clusters)

    @property
    def has_converged(self) -> bool:
        """Check if all MCMC analyses converged."""
        if not self.mcmc_diagnostics:
            return True  # Non-MCMC assumed converged
        return all(d.converged for d in self.mcmc_diagnostics)


# =============================================================================
# Utility Functions
# =============================================================================


def _compute_file_checksum(path: Path, algorithm: str = "sha256") -> str:
    """Compute checksum of a file.

    Args:
        path: Path to file
        algorithm: Hash algorithm (default sha256)

    Returns:
    -------
        Hex digest of file contents
    """
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# MCMC Analysis Result Types
# =============================================================================


@dataclass(slots=True, frozen=True)
class ClusterMCMCResult:
    """Container for the uncertainty result of a single cluster."""

    cluster: Cluster
    result: UncertaintyResult


@dataclass(slots=True, frozen=True)
class MCMCAnalysisResult:
    """Aggregate result for an MCMC uncertainty run."""

    clusters: list[Cluster]
    params: Parameters
    noise: float
    peaks: list[Peak]
    cluster_results: list[ClusterMCMCResult]


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "AmplitudeEstimate",
    "ClusterEstimates",
    "ClusterMCMCResult",
    "ConvergenceStatus",
    "FitResult",
    "FitResults",
    "FitStatistics",
    "MCMCAnalysisResult",
    "MCMCDiagnostics",
    "ParameterCategory",
    "ParameterDiagnostic",
    "ParameterEstimate",
    "ResidualStatistics",
    "RunMetadata",
    "compute_chi_squared",
    "compute_degrees_of_freedom",
    "compute_reduced_chi_squared",
]
