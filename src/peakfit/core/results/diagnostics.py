"""MCMC convergence diagnostics models.

This module defines dataclasses for representing MCMC convergence
diagnostics, including R-hat, ESS, and convergence status assessments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from peakfit.core.diagnostics.convergence import ConvergenceDiagnostics


class ConvergenceStatus(str, Enum):
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


@dataclass(slots=True)
class ParameterDiagnostic:
    """Convergence diagnostics for a single MCMC parameter.

    Attributes:
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
            ParameterDiagnostic with computed status
        """
        warnings = []
        status = ConvergenceStatus.UNKNOWN

        if rhat is None or ess_bulk is None:
            return cls(name=name, rhat=rhat, ess_bulk=ess_bulk, ess_tail=ess_tail)

        # Determine status based on BARG guidelines
        if rhat <= 1.01 and ess_bulk >= 10000:
            status = ConvergenceStatus.EXCELLENT
        elif rhat <= 1.01 and ess_bulk >= 1000:
            status = ConvergenceStatus.GOOD
        elif rhat <= 1.05 and ess_bulk >= 400:
            status = ConvergenceStatus.ACCEPTABLE
        elif rhat <= 1.05 and ess_bulk >= 100:
            status = ConvergenceStatus.MARGINAL
        else:
            status = ConvergenceStatus.POOR

        # Generate warnings
        if rhat > 1.01:
            warnings.append(f"R-hat = {rhat:.4f} (should be ≤ 1.01). Chains have not mixed well.")
        if rhat > 1.05:
            warnings.append(
                f"R-hat = {rhat:.4f} is very high (> 1.05). Results should not be trusted."
            )

        recommended_ess = 100 * n_chains
        if ess_bulk < recommended_ess:
            warnings.append(
                f"ESS_bulk = {ess_bulk:.0f} (recommended ≥ {recommended_ess:.0f}). "
                "Consider more samples."
            )
        if ess_bulk < 10 * n_chains:
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
    burn_in_details: dict = field(default_factory=dict)

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
        burn_in_details: dict | None = None,
    ) -> MCMCDiagnostics:
        """Create from existing ConvergenceDiagnostics object.

        Args:
            conv_diag: Convergence diagnostics from core module
            burn_in: Number of burn-in samples
            burn_in_method: How burn-in was determined
            burn_in_details: Additional info about burn-in

        Returns:
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

    def to_dict(self) -> dict:
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
