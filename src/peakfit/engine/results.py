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
    from peakfit.engine.diagnostics.convergence import ConvergenceDiagnostics
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
_MIN_CORRELATION_PARAMS = 2

_RHAT_EXCELLENT_THRESHOLD = 1.01
_RHAT_ACCEPTABLE_THRESHOLD = 1.05

_ESS_EXCELLENT_THRESHOLD = 10000
_ESS_GOOD_THRESHOLD = 1000
_ESS_ACCEPTABLE_THRESHOLD = 400
_ESS_MARGINAL_THRESHOLD = 100

_RECOMMENDED_ESS_PER_CHAIN = 100
_LOW_ESS_PER_CHAIN = 10

_REDUCED_CHI2_GOOD_MIN = 0.5
_REDUCED_CHI2_GOOD_MAX = 2.0

_AIC_STRONG_EVIDENCE = 10
_AIC_MODERATE_EVIDENCE = 4
_AIC_WEAK_EVIDENCE = 2


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


class FitMethod(StrEnum):
    """Fitting method used."""

    LEAST_SQUARES = "least_squares"
    BASIN_HOPPING = "basin_hopping"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    MCMC = "mcmc"
    PROFILE_LIKELIHOOD = "profile_likelihood"


class OutputVerbosity(StrEnum):
    """Output verbosity levels.

    Controls which outputs are generated:
    - MINIMAL: Essential outputs only (parameters CSV, summary JSON)
    - STANDARD: Default outputs (+ diagnostics, figures)
    - FULL: All outputs including posteriors and debug info
    """

    MINIMAL = "minimal"
    STANDARD = "standard"
    FULL = "full"


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

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary (excludes large arrays)."""
        return {
            "n_points": self.n_points,
            "n_params": self.n_params,
            "dof": self.dof,
            "noise_level": self.noise_level,
            "sum_squared": self.sum_squared,
            "rms": self.rms,
            "mean": self.mean,
            "std": self.std,
        }


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

    @property
    def is_good_fit(self) -> bool:
        """Check if fit quality is acceptable.

        A fit is considered good if:
        - Reduced chi-squared is between 0.5 and 2.0
        - The fit converged
        """
        return (
            self.fit_converged
            and _REDUCED_CHI2_GOOD_MIN <= self.reduced_chi_squared <= _REDUCED_CHI2_GOOD_MAX
        )

    @classmethod
    def from_residuals(
        cls,
        residuals: FloatArray,
        noise: float,
        n_params: int,
    ) -> FitStatistics:
        """Compute statistics from fit residuals.

        Args:
            residuals: Raw residuals (data - model)
            noise: Noise level for normalization
            n_params: Total number of fitted parameters (lineshape + amplitudes)

        Returns:
        -------
            FitStatistics with computed values
        """
        n_data = len(residuals)
        normalized = residuals / noise
        chi2 = compute_chi_squared(normalized)
        red_chi2 = compute_reduced_chi_squared(chi2, n_data, n_params)

        # Compute information criteria
        # Formula for AIC: -2 * log_likelihood + 2 * k
        # Formula for BIC: -2 * log_likelihood + k * log(n)
        # Formula for log_likelihood: -0.5 * chi2 - n * log(noise) - 0.5 * n * log(2*pi)
        log_like = -0.5 * chi2 - n_data * np.log(noise) - 0.5 * n_data * np.log(2 * np.pi)
        aic = -2 * log_like + 2 * n_params
        bic = -2 * log_like + n_params * np.log(n_data) if n_data > 0 else None

        residual_stats = ResidualStatistics(
            raw_residuals=residuals,
            normalized_residuals=normalized,
            n_points=n_data,
            n_params=n_params,
            noise_level=noise,
        )

        return cls(
            chi_squared=chi2,
            reduced_chi_squared=red_chi2,
            aic=aic,
            bic=bic,
            log_likelihood=log_like,
            n_data=n_data,
            n_params=n_params,
            residuals=residual_stats,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, object] = {
            "chi_squared": self.chi_squared,
            "reduced_chi_squared": self.reduced_chi_squared,
            "n_data": self.n_data,
            "n_params": self.n_params,
            "dof": self.dof,
            "fit_converged": self.fit_converged,
            "n_function_evals": self.n_function_evals,
        }

        if self.aic is not None:
            result["aic"] = self.aic
        if self.bic is not None:
            result["bic"] = self.bic
        if self.log_likelihood is not None:
            result["log_likelihood"] = self.log_likelihood
        if self.fit_message:
            result["fit_message"] = self.fit_message

        result["residuals"] = self.residuals.to_dict()

        return result


@dataclass(slots=True)
class ModelComparison:
    """Comparison between two fitted models.

    Used for model selection (e.g., one-site vs two-site exchange).

    Attributes:
    ----------
        model_a_name: Name/description of first model
        model_b_name: Name/description of second model
        delta_aic: AIC(model_b) - AIC(model_a), negative favors model_b
        delta_bic: BIC(model_b) - BIC(model_a), negative favors model_b
        likelihood_ratio: Ratio of likelihoods
        p_value: P-value for likelihood ratio test (if nested models)
        preferred_model: Name of preferred model based on criteria
        evidence_strength: Qualitative assessment of evidence
    """

    model_a_name: str
    model_b_name: str
    delta_aic: float | None = None
    delta_bic: float | None = None
    likelihood_ratio: float | None = None
    p_value: float | None = None
    preferred_model: str = ""
    evidence_strength: str = ""  # "strong", "moderate", "weak", "inconclusive"

    @classmethod
    def compare(
        cls,
        stats_a: FitStatistics,
        stats_b: FitStatistics,
        name_a: str = "Model A",
        name_b: str = "Model B",
    ) -> ModelComparison:
        """Compare two models using their fit statistics.

        Args:
            stats_a: Statistics for model A
            stats_b: Statistics for model B
            name_a: Name for model A
            name_b: Name for model B

        Returns:
        -------
            ModelComparison with computed metrics
        """
        delta_aic = None
        delta_bic = None

        if stats_a.aic is not None and stats_b.aic is not None:
            delta_aic = stats_b.aic - stats_a.aic

        if stats_a.bic is not None and stats_b.bic is not None:
            delta_bic = stats_b.bic - stats_a.bic

        # Determine preferred model based on AIC
        preferred = ""
        evidence = "inconclusive"
        if delta_aic is not None:
            if delta_aic < -_AIC_STRONG_EVIDENCE:
                preferred = name_b
                evidence = "strong"
            elif delta_aic < -_AIC_MODERATE_EVIDENCE:
                preferred = name_b
                evidence = "moderate"
            elif delta_aic < -_AIC_WEAK_EVIDENCE:
                preferred = name_b
                evidence = "weak"
            elif delta_aic > _AIC_STRONG_EVIDENCE:
                preferred = name_a
                evidence = "strong"
            elif delta_aic > _AIC_MODERATE_EVIDENCE:
                preferred = name_a
                evidence = "moderate"
            elif delta_aic > _AIC_WEAK_EVIDENCE:
                preferred = name_a
                evidence = "weak"

        return cls(
            model_a_name=name_a,
            model_b_name=name_b,
            delta_aic=delta_aic,
            delta_bic=delta_bic,
            preferred_model=preferred,
            evidence_strength=evidence,
        )

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, object] = {
            "model_a": self.model_a_name,
            "model_b": self.model_b_name,
            "preferred_model": self.preferred_model,
            "evidence_strength": self.evidence_strength,
        }
        if self.delta_aic is not None:
            result["delta_aic"] = self.delta_aic
        if self.delta_bic is not None:
            result["delta_bic"] = self.delta_bic
        if self.likelihood_ratio is not None:
            result["likelihood_ratio"] = self.likelihood_ratio
        if self.p_value is not None:
            result["p_value"] = self.p_value
        return result


# =============================================================================
# Parameter Estimates
# =============================================================================


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

    # Structured parameter identifier (for consistent naming)
    param_id: ParameterId | None = None

    @property
    def user_name(self) -> str:
        """User-friendly parameter name.

        Returns the user_name from param_id if available,
        otherwise falls back to the raw name.
        """
        if self.param_id is not None:
            return self.param_id.user_name
        return self.name

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

    def format_value(self, precision: int = 6) -> str:
        """Format value with uncertainty for display.

        Args:
            precision: Number of decimal places

        Returns:
        -------
            Formatted string like "25.300 ± 1.200" or "25.300 +1.200/-1.100"
        """
        if self.has_asymmetric_error:
            return (
                f"{self.value:.{precision}f} "
                f"+{self.error_upper:.{precision}f}/-{self.error_lower:.{precision}f}"
            )
        return f"{self.value:.{precision}f} ± {self.std_error:.{precision}f}"

    def to_dict(self) -> dict[str, Any]:
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

    @property
    def has_asymmetric_error(self) -> bool:
        """Check if asymmetric confidence intervals are available."""
        return self.ci_68_lower is not None and self.ci_68_upper is not None

    def to_dict(self) -> dict[str, Any]:
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

    @property
    def n_lineshape_params(self) -> int:
        """Number of lineshape parameters."""
        return len(self.lineshape_params)

    @property
    def n_series(self) -> int:
        """Number of spectra in the pseudo dimension (inferred from amplitudes)."""
        if not self.amplitudes:
            return 0
        return max(a.plane_index for a in self.amplitudes) + 1

    def get_amplitudes_for_peak(self, peak_name: str) -> list[AmplitudeEstimate]:
        """Get all amplitudes for a specific peak."""
        return [a for a in self.amplitudes if a.peak_name == peak_name]

    def get_strong_correlations(
        self,
        threshold: float = 0.7,
    ) -> list[tuple[str, str, float]]:
        """Find pairs of strongly correlated parameters.

        Args:
            threshold: Minimum absolute correlation to report

        Returns:
        -------
            List of (param1, param2, correlation) tuples
        """
        if (
            self.correlation_matrix is None
            or len(self.correlation_param_names) < _MIN_CORRELATION_PARAMS
        ):
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

    def to_dict(self) -> dict[str, Any]:
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

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, object] = {
            "name": self.name,
            "status": self.status.value,
        }
        if self.rhat is not None:
            result["rhat"] = self.rhat
        if self.ess_bulk is not None:
            result["ess_bulk"] = self.ess_bulk
        if self.ess_tail is not None:
            result["ess_tail"] = self.ess_tail
        if self.warnings:
            result["warnings"] = self.warnings
        return result


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

    def get_rhat_values(self) -> dict[str, float]:
        """Get dictionary of R-hat values by parameter name."""
        return {d.name: d.rhat for d in self.parameter_diagnostics if d.rhat is not None}

    def get_ess_values(self) -> dict[str, float]:
        """Get dictionary of bulk ESS values by parameter name."""
        return {d.name: d.ess_bulk for d in self.parameter_diagnostics if d.ess_bulk is not None}

    @classmethod
    def from_convergence_diagnostics(
        cls,
        conv_diag: ConvergenceDiagnostics,
        burn_in: int = 0,
        burn_in_method: str = "manual",
        burn_in_details: dict[str, Any] | None = None,
    ) -> MCMCDiagnostics:
        """Create from existing ConvergenceDiagnostics object.

        Args:
            conv_diag: Convergence diagnostics from core module
            burn_in: Number of burn-in samples
            burn_in_method: How burn-in was determined
            burn_in_details: Additional info about burn-in

        Returns:
        -------
            MCMCDiagnostics instance
        """
        param_diagnostics = []
        for i, name in enumerate(conv_diag.parameter_names):
            rhat = float(conv_diag.rhat[i]) if i < len(conv_diag.rhat) else None
            ess_bulk = float(conv_diag.ess_bulk[i]) if i < len(conv_diag.ess_bulk) else None
            ess_tail = float(conv_diag.ess_tail[i]) if i < len(conv_diag.ess_tail) else None

            diag = ParameterDiagnostic.from_values(
                name=name,
                rhat=rhat,
                ess_bulk=ess_bulk,
                ess_tail=ess_tail,
                n_chains=conv_diag.n_chains,
            )
            param_diagnostics.append(diag)

        result = cls(
            n_chains=conv_diag.n_chains,
            n_samples=conv_diag.n_samples,
            burn_in=burn_in,
            parameter_diagnostics=param_diagnostics,
            burn_in_method=burn_in_method,
            burn_in_details=burn_in_details or {},
        )
        result.update_overall_status()

        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_chains": self.n_chains,
            "n_samples": self.n_samples,
            "burn_in": self.burn_in,
            "burn_in_method": self.burn_in_method,
            "total_samples": self.total_samples,
            "overall_status": self.overall_status.value,
            "converged": self.converged,
            "parameters": [d.to_dict() for d in self.parameter_diagnostics],
            "burn_in_details": self.burn_in_details,
        }


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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "timestamp": self.timestamp,
            "software_version": self.software_version,
            "python_version": self.python_version,
            "platform": self.platform,
        }

        if self.git_commit:
            result["git_commit"] = self.git_commit
        if self.input_files:
            result["input_files"] = self.input_files
        if self.configuration:
            result["configuration"] = self.configuration
        if self.command_line:
            result["command_line"] = self.command_line
        if self.run_duration_seconds is not None:
            result["run_duration_seconds"] = self.run_duration_seconds

        return result


@dataclass
class FitResult:
    """Result of optimization for a single cluster.

    Encapsulates all outputs from an optimization strategy, whether it be
    Iterative Least Squares, VarPro, or MCMC.
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
        method: Fitting method used
        clusters: Per-cluster parameter estimates
        statistics: Per-cluster fit statistics
        global_statistics: Overall fit statistics (if applicable)
        mcmc_diagnostics: Per-cluster MCMC diagnostics (if MCMC used)
        model_comparisons: Model comparison results (if multiple models)
        z_values: Z-dimension values (e.g., relaxation delays)
    """

    metadata: RunMetadata = field(default_factory=RunMetadata)
    method: FitMethod = FitMethod.LEAST_SQUARES
    clusters: list[ClusterEstimates] = field(default_factory=list)
    statistics: list[FitStatistics] = field(default_factory=list)
    global_statistics: FitStatistics | None = None
    mcmc_diagnostics: list[MCMCDiagnostics] = field(default_factory=list)
    model_comparisons: list[ModelComparison] = field(default_factory=list)
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
    def all_peak_names(self) -> list[str]:
        """List of all peak names across clusters."""
        names = []
        for cluster in self.clusters:
            names.extend(cluster.peak_names)
        return names

    @property
    def is_mcmc(self) -> bool:
        """Check if MCMC was used."""
        return self.method == FitMethod.MCMC

    @property
    def has_converged(self) -> bool:
        """Check if all MCMC analyses converged."""
        if not self.mcmc_diagnostics:
            return True  # Non-MCMC assumed converged
        return all(d.converged for d in self.mcmc_diagnostics)

    def get_cluster_by_peak(self, peak_name: str) -> ClusterEstimates | None:
        """Find cluster containing a specific peak."""
        for cluster in self.clusters:
            if peak_name in cluster.peak_names:
                return cluster
        return None

    def get_all_problematic_params(self) -> list[tuple[str, str]]:
        """Get all problematic parameters across clusters.

        Returns:
        -------
            List of (cluster_label, param_name) tuples
        """
        problems: list[tuple[str, str]] = []
        for cluster in self.clusters:
            label = ", ".join(cluster.peak_names)
            problems.extend((label, param.name) for param in cluster.get_problematic_params())
        return problems

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "metadata": self.metadata.to_dict(),
            "method": self.method.value,
            "n_clusters": self.n_clusters,
            "n_peaks": self.n_peaks,
            "clusters": [c.to_dict() for c in self.clusters],
        }

        if self.statistics:
            result["statistics"] = [s.to_dict() for s in self.statistics]

        if self.global_statistics:
            result["global_statistics"] = self.global_statistics.to_dict()

        if self.mcmc_diagnostics:
            result["mcmc_diagnostics"] = [d.to_dict() for d in self.mcmc_diagnostics]

        if self.model_comparisons:
            result["model_comparisons"] = [m.to_dict() for m in self.model_comparisons]

        if self.z_values is not None:
            result["z_values"] = self.z_values.tolist()

        return result

    def summary_dict(self) -> dict[str, Any]:
        """Generate a summary dictionary for quick inspection.

        This is a condensed version suitable for the executive summary.
        """
        summary: dict[str, object] = {
            "timestamp": self.metadata.timestamp,
            "method": self.method.value,
            "n_clusters": self.n_clusters,
            "n_peaks": self.n_peaks,
        }

        if self.global_statistics:
            summary["reduced_chi_squared"] = self.global_statistics.reduced_chi_squared
            summary["fit_converged"] = self.global_statistics.fit_converged

        if self.mcmc_diagnostics:
            summary["mcmc_converged"] = self.has_converged
            n_problematic = sum(len(d.get_problematic_parameters()) for d in self.mcmc_diagnostics)
            summary["n_problematic_params"] = n_problematic

        problems: list[tuple[str, str]] = self.get_all_problematic_params()
        summary["n_problematic_total"] = len(problems)
        if problems:
            summary["problematic_params"] = problems[:10]  # First 10 only

        return summary


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
    "FitMethod",
    "FitResult",
    "FitResults",
    "FitStatistics",
    "MCMCAnalysisResult",
    "MCMCDiagnostics",
    "ModelComparison",
    "OutputVerbosity",
    "ParameterCategory",
    "ParameterDiagnostic",
    "ParameterEstimate",
    "ResidualStatistics",
    "RunMetadata",
    "compute_chi_squared",
    "compute_degrees_of_freedom",
    "compute_reduced_chi_squared",
]
