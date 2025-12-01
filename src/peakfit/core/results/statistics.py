"""Fit statistics and model comparison metrics.

This module defines dataclasses for representing fit quality metrics,
residual statistics, and model comparison information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from peakfit.core.shared.typing import FloatArray


def compute_chi_squared(residuals: FloatArray) -> float:
    """Compute chi-squared (sum of squared residuals).

    This is the single source of truth for chi-squared calculation.

    Args:
        residuals: Normalized residuals (data - model) / noise

    Returns
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

    Returns
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

    Returns
    -------
        Reduced chi-squared value (chi_squared / dof)
    """
    dof = compute_degrees_of_freedom(n_data, n_params)
    return chi_squared / dof


@dataclass(slots=True)
class ResidualStatistics:
    """Statistics computed from fit residuals.

    Attributes
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

    Attributes
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
        return self.fit_converged and 0.5 <= self.reduced_chi_squared <= 2.0

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

        Returns
        -------
            FitStatistics with computed values
        """
        n_data = len(residuals)
        normalized = residuals / noise
        chi2 = compute_chi_squared(normalized)
        red_chi2 = compute_reduced_chi_squared(chi2, n_data, n_params)

        # Compute information criteria
        # AIC = -2 * log_likelihood + 2 * k
        # BIC = -2 * log_likelihood + k * log(n)
        # For Gaussian errors: log_likelihood = -0.5 * chi2 - n * log(noise) - 0.5 * n * log(2*pi)
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

    def to_dict(self) -> dict:
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

    Attributes
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

        Returns
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
            if delta_aic < -10:
                preferred = name_b
                evidence = "strong"
            elif delta_aic < -4:
                preferred = name_b
                evidence = "moderate"
            elif delta_aic < -2:
                preferred = name_b
                evidence = "weak"
            elif delta_aic > 10:
                preferred = name_a
                evidence = "strong"
            elif delta_aic > 4:
                preferred = name_a
                evidence = "moderate"
            elif delta_aic > 2:
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
