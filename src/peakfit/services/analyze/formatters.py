"""Result formatting for analyze services.

These formatters convert service results into display-ready structures.
They don't do Rich/console output - that stays in CLI layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

if TYPE_CHECKING:
    from peakfit.core.diagnostics.convergence import ConvergenceDiagnostics
    from peakfit.core.shared.typing import FloatArray
    from peakfit.services.analyze.mcmc_service import ClusterMCMCResult


@dataclass
class MCMCParameterSummary:
    """Summary of a single MCMC parameter result for display."""

    name: str
    value: float
    std_error: float
    ci_68_lower: float
    ci_68_upper: float
    ci_95_lower: float
    ci_95_upper: float
    rhat: float | None = None
    ess_bulk: float | None = None
    ess_tail: float | None = None

    @property
    def converged(self) -> bool:
        """Check if parameter has converged based on R-hat."""
        if self.rhat is None:
            return True
        return self.rhat <= 1.05

    @property
    def convergence_status(self) -> str:
        """Get convergence status string."""
        if self.rhat is None or self.ess_bulk is None:
            return "unknown"
        if self.rhat <= 1.01 and self.ess_bulk >= 10000:
            return "excellent"
        if self.rhat <= 1.01 and self.ess_bulk >= 100:
            return "good"
        if self.rhat <= 1.05 and self.ess_bulk >= 100:
            return "acceptable"
        if self.rhat <= 1.05 and self.ess_bulk >= 10:
            return "marginal"
        return "poor"


@dataclass
class MCMCAmplitudeSummary:
    """Summary of MCMC amplitude (intensity) results for a single peak."""

    peak_name: str
    plane_index: int
    value: float
    std_error: float
    ci_68_lower: float
    ci_68_upper: float
    ci_95_lower: float
    ci_95_upper: float
    z_value: float | None = None  # Corresponding Z-dimension value if available


@dataclass
class MCMCClusterSummary:
    """Summary of MCMC results for a single cluster."""

    peak_names: list[str]
    parameter_summaries: list[MCMCParameterSummary]
    correlation_matrix: np.ndarray | None
    burn_in_used: int | None
    n_chains: int
    n_samples: int
    # Amplitude (intensity) summaries
    amplitude_summaries: list[MCMCAmplitudeSummary] = field(default_factory=list)

    @property
    def cluster_label(self) -> str:
        """Get formatted cluster label."""
        return ", ".join(self.peak_names)

    def get_strong_correlations(self, threshold: float = 0.7) -> list[tuple[str, str, float]]:
        """Get pairs of strongly correlated parameters.

        Returns
        -------
            List of (param1, param2, correlation) tuples
        """
        if self.correlation_matrix is None:
            return []

        pairs = []
        n_params = len(self.parameter_summaries)
        for i in range(n_params):
            for j in range(i + 1, n_params):
                corr = self.correlation_matrix[i, j]
                if abs(corr) >= threshold:
                    pairs.append((
                        self.parameter_summaries[i].name,
                        self.parameter_summaries[j].name,
                        corr,
                    ))
        return pairs

    def get_amplitudes_by_peak(self) -> dict[str, list[MCMCAmplitudeSummary]]:
        """Group amplitude summaries by peak name.

        Returns
        -------
            Dictionary mapping peak name to list of amplitude summaries across planes
        """
        by_peak: dict[str, list[MCMCAmplitudeSummary]] = {}
        for amp in self.amplitude_summaries:
            if amp.peak_name not in by_peak:
                by_peak[amp.peak_name] = []
            by_peak[amp.peak_name].append(amp)
        return by_peak


def format_mcmc_cluster_result(
    cluster_result: ClusterMCMCResult,
    diagnostics: ConvergenceDiagnostics | None = None,
    z_values: FloatArray | None = None,
) -> MCMCClusterSummary:
    """Convert MCMCClusterResult to display-friendly summary.

    Args:
        cluster_result: Result from MCMC analysis service
        diagnostics: Optional convergence diagnostics
        z_values: Optional Z-dimension values for amplitude labeling

    Returns
    -------
        MCMCClusterSummary ready for display
    """
    result = cluster_result.result
    cluster = cluster_result.cluster
    peak_names = [p.name for p in cluster.peaks]

    # Get diagnostics values
    if diagnostics is None and result.mcmc_diagnostics is not None:
        diagnostics = result.mcmc_diagnostics

    # Get metadata for distinguishing lineshape vs amplitude parameters
    n_lineshape = getattr(result, "n_lineshape_params", len(result.parameter_names))
    n_planes = getattr(result, "n_planes", 1)
    amp_peak_names = result.amplitude_names or []

    parameter_summaries = []
    # Only include lineshape parameters in the main parameter summaries
    for i in range(n_lineshape):
        name = result.parameter_names[i]
        ci_68 = result.confidence_intervals_68[i]
        ci_95 = result.confidence_intervals_95[i]

        summary = MCMCParameterSummary(
            name=name,
            value=result.values[i],
            std_error=result.std_errors[i],
            ci_68_lower=ci_68[0],
            ci_68_upper=ci_68[1],
            ci_95_lower=ci_95[0],
            ci_95_upper=ci_95[1],
            rhat=diagnostics.rhat[i] if diagnostics and i < len(diagnostics.rhat) else None,
            ess_bulk=diagnostics.ess_bulk[i]
            if diagnostics and i < len(diagnostics.ess_bulk)
            else None,
            ess_tail=diagnostics.ess_tail[i]
            if diagnostics and i < len(diagnostics.ess_tail)
            else None,
        )
        parameter_summaries.append(summary)

    # Build amplitude summaries from unified parameter arrays
    amplitude_summaries: list[MCMCAmplitudeSummary] = []
    if amp_peak_names and n_lineshape < len(result.parameter_names):
        # Amplitude parameters start after lineshape parameters
        amp_start = n_lineshape

        for i_peak, peak_name in enumerate(amp_peak_names):
            for i_plane in range(n_planes):
                idx = amp_start + i_peak * n_planes + i_plane
                ci_68 = result.confidence_intervals_68[idx]
                ci_95 = result.confidence_intervals_95[idx]

                z_val = None
                if z_values is not None and i_plane < len(z_values):
                    z_val = float(z_values[i_plane])

                amp_summary = MCMCAmplitudeSummary(
                    peak_name=peak_name,
                    plane_index=i_plane,
                    value=float(result.values[idx]),
                    std_error=float(result.std_errors[idx]),
                    ci_68_lower=float(ci_68[0]),
                    ci_68_upper=float(ci_68[1]),
                    ci_95_lower=float(ci_95[0]),
                    ci_95_upper=float(ci_95[1]),
                    z_value=z_val,
                )
                amplitude_summaries.append(amp_summary)

    return MCMCClusterSummary(
        peak_names=peak_names,
        parameter_summaries=parameter_summaries,
        correlation_matrix=result.correlation_matrix,
        burn_in_used=result.burn_in_info.get("burn_in") if result.burn_in_info else None,
        n_chains=diagnostics.n_chains if diagnostics else 0,
        n_samples=diagnostics.n_samples if diagnostics else 0,
        amplitude_summaries=amplitude_summaries,
    )


@dataclass
class ProfileParameterSummary:
    """Summary of profile likelihood results for a parameter."""

    name: str
    best_fit: float
    lower_bound: float
    upper_bound: float
    ci_lower: float
    ci_upper: float
    converged: bool


def format_profile_results(
    parameter_name: str,
    best_fit: float,
    ci_lower: float,
    ci_upper: float,
    converged: bool = True,
) -> ProfileParameterSummary:
    """Format profile likelihood result for display.

    Args:
        parameter_name: Name of the parameter
        best_fit: Best-fit value
        ci_lower: Lower confidence interval bound
        ci_upper: Upper confidence interval bound
        converged: Whether the profile converged

    Returns
    -------
        ProfileParameterSummary ready for display
    """
    return ProfileParameterSummary(
        name=parameter_name,
        best_fit=best_fit,
        lower_bound=ci_lower,
        upper_bound=ci_upper,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        converged=converged,
    )
