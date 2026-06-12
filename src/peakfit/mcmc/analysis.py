"""MCMC workflow helpers for post-fit uncertainty estimation.

This module consolidates MCMC analysis and formatting utilities for CLI use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from peakfit.engine.algorithms.mcmc import estimate_uncertainties_mcmc
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.results import ClusterMCMCResult, MCMCAnalysisResult
from peakfit.io.readers.results import ResultsLoader
from peakfit.io.state import default_state_path, load_state
from peakfit.shared.paths import format_path

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from peakfit.engine.diagnostics.convergence import ConvergenceDiagnostics
    from peakfit.engine.domain.cluster import Cluster
    from peakfit.shared.typing import FloatArray


# =============================================================================
# Constants
# =============================================================================

_RHAT_CONVERGED_THRESHOLD = 1.05
_RHAT_EXCELLENT_THRESHOLD = 1.01
_ESS_EXCELLENT_THRESHOLD = 10_000
_ESS_GOOD_THRESHOLD = 100
_ESS_MARGINAL_THRESHOLD = 10


class PeaksNotFoundError(ValueError):
    """Raised when requested peaks cannot be matched to clusters."""

    def __init__(self, peaks: list[str]) -> None:
        super().__init__(f"No clusters found for peaks: {peaks}")
        self.peaks = peaks


_DEFAULT_NOISE = 1.0


def run_mcmc_analysis(
    results_dir: Path,
    *,
    target_peaks: list[str] | None = None,
    n_walkers: int = 32,
    n_steps: int = 1000,
    burn_in: int | None = None,
    auto_burnin: bool = True,
    workers: int = 1,
    progress_callback: Callable[[int, int, str, float | None], None] | None = None,
) -> MCMCAnalysisResult:
    """Run MCMC sampling for the results in the given directory."""
    results_dir = Path(results_dir)
    loader = ResultsLoader(results_dir)
    state_path = default_state_path(results_dir)
    state = load_state(state_path) if state_path.exists() else loader.load_fitting_state()
    summary = loader.load_summary()

    spec_path, list_path = _resolve_input_paths(results_dir, summary.metadata.input_files)
    if not spec_path.exists():
        raise FileNotFoundError(f"Spectrum file not found: {format_path(spec_path)}")
    if not list_path.exists():
        raise FileNotFoundError(f"Peak list file not found: {format_path(list_path)}")

    noise = state.noise if state.noise is not None else _DEFAULT_NOISE
    target_clusters = _filter_clusters(state.clusters, target_peaks)
    if target_peaks and not target_clusters:
        raise PeaksNotFoundError(target_peaks)

    results: list[ClusterMCMCResult] = []
    burn_in_arg = None if auto_burnin else burn_in

    def _progress_callback(i: int, n: int, msg: str, acceptance: float | None) -> None:
        if progress_callback is not None:
            progress_callback(i, n, msg, acceptance)

    callback = _progress_callback if progress_callback is not None else None
    n_clusters = len(target_clusters)
    for i_cluster, cluster in enumerate(target_clusters, start=1):
        if callback is not None:
            callback(
                0,
                n_steps,
                f"Preparing cluster {i_cluster}/{n_clusters} ({len(cluster.peaks)} peaks)...",
                None,
            )

        cluster_params = _create_cluster_params(cluster, state.scalar_params)
        if callback is not None:
            callback(
                0,
                n_steps,
                f"Prepared {len(cluster_params)} parameters for cluster {i_cluster}/{n_clusters}",
                None,
            )

        result = estimate_uncertainties_mcmc(
            cluster_params,
            cluster,
            noise,
            n_walkers=n_walkers,
            n_steps=n_steps,
            burn_in=burn_in_arg,
            workers=workers,
            progress_callback=callback,
        )
        results.append(ClusterMCMCResult(cluster=cluster, result=result))

    return MCMCAnalysisResult(
        clusters=target_clusters,
        params=state.scalar_params,
        noise=state.noise or 0.0,
        peaks=state.peaks,
        cluster_results=results,
    )


def _resolve_input_paths(results_dir: Path, input_files: dict[str, Any]) -> tuple[Path, Path]:
    """Resolve spectrum and peaklist paths, handling potential relocation."""
    spectrum_info = input_files.get("spectrum")
    peaklist_info = input_files.get("peaklist")

    if spectrum_info is None or peaklist_info is None:
        raise ValueError("Input paths missing from results metadata.")

    if hasattr(spectrum_info, "path"):
        spec_path_str = spectrum_info.path
    else:
        spec_path_str = spectrum_info.get("path") if isinstance(spectrum_info, dict) else None

    if hasattr(peaklist_info, "path"):
        list_path_str = peaklist_info.path
    else:
        list_path_str = peaklist_info.get("path") if isinstance(peaklist_info, dict) else None

    if not spec_path_str or not list_path_str:
        raise ValueError("Input paths missing from results metadata.")

    spec_path = _find_input_file(Path(spec_path_str), results_dir)
    list_path = _find_input_file(Path(list_path_str), results_dir)

    if not spec_path.exists():
        raise FileNotFoundError(f"Spectrum file not found: {format_path(spec_path_str)}")
    if not list_path.exists():
        raise FileNotFoundError(f"Peak list file not found: {format_path(list_path_str)}")

    return spec_path, list_path


def _find_input_file(path: Path, results_dir: Path) -> Path:
    """Find an input file after a results directory has moved."""
    parent = results_dir.parent
    grandparent = parent.parent
    candidates = [
        path,
        results_dir / path.name,
        parent / path.name,
        grandparent / path.name,
        grandparent / "data" / path.name,
        grandparent / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def _filter_clusters(clusters: list[Cluster], peaks: list[str] | None) -> list[Cluster]:
    if not peaks:
        return list(clusters)
    peak_set = set(peaks)
    return [cluster for cluster in clusters if any(p.name in peak_set for p in cluster.peaks)]


def _create_cluster_params(cluster: Cluster, params_all: Parameters) -> Parameters:
    """Extract parameters for peaks in the cluster from the full parameter set."""
    peak_names = {p.name for p in cluster.peaks}
    prefixes = tuple(f"{peak_name}." for peak_name in peak_names)

    cluster_params = Parameters()
    for key, param in params_all.items():
        if key.startswith(prefixes):
            cluster_params[key] = param.model_copy(deep=False)

    return cluster_params


# =============================================================================
# MCMC Formatters
# =============================================================================


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
        return self.rhat <= _RHAT_CONVERGED_THRESHOLD if self.rhat is not None else True

    @property
    def convergence_status(self) -> str:
        """Get convergence status string."""
        if self.rhat is None or self.ess_bulk is None:
            return "unknown"
        if self.rhat <= _RHAT_EXCELLENT_THRESHOLD:
            if self.ess_bulk >= _ESS_EXCELLENT_THRESHOLD:
                return "excellent"
            if self.ess_bulk >= _ESS_GOOD_THRESHOLD:
                return "good"
        if self.rhat <= _RHAT_CONVERGED_THRESHOLD:
            if self.ess_bulk >= _ESS_GOOD_THRESHOLD:
                return "acceptable"
            if self.ess_bulk >= _ESS_MARGINAL_THRESHOLD:
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
    z_value: float | None = None


@dataclass
class MCMCClusterSummary:
    """Summary of MCMC results for a single cluster."""

    peak_names: list[str]
    parameter_summaries: list[MCMCParameterSummary]
    correlation_matrix: np.ndarray | None
    burn_in_used: int | None
    n_chains: int
    n_samples: int
    amplitude_summaries: list[MCMCAmplitudeSummary] = field(default_factory=list)

    @property
    def cluster_label(self) -> str:
        """Get formatted cluster label."""
        return ", ".join(self.peak_names)

    def get_strong_correlations(self, threshold: float = 0.7) -> list[tuple[str, str, float]]:
        """Get pairs of strongly correlated parameters.

        Returns:
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
                    pairs.append(
                        (
                            self.parameter_summaries[i].name,
                            self.parameter_summaries[j].name,
                            corr,
                        )
                    )
        return pairs

    def get_amplitudes_by_peak(self) -> dict[str, list[MCMCAmplitudeSummary]]:
        """Group amplitude summaries by peak name."""
        by_peak: dict[str, list[MCMCAmplitudeSummary]] = {}
        for amp in self.amplitude_summaries:
            if amp.peak_name not in by_peak:
                by_peak[amp.peak_name] = []
            by_peak[amp.peak_name].append(amp)
        return by_peak


def _extract_lineshape_summaries(
    result: Any, diagnostics: ConvergenceDiagnostics | None
) -> list[MCMCParameterSummary]:
    """Extract lineshape parameter summaries from MCMC result."""
    param_summaries = []
    for i in range(result.n_lineshape_params):
        name = result.parameter_names[i]
        val = float(result.values[i])
        err = float(result.std_errors[i])

        # Confidence Intervals
        ci_68 = (
            tuple(result.confidence_intervals_68[i])
            if result.confidence_intervals_68 is not None
            else None
        )
        ci_95 = (
            tuple(result.confidence_intervals_95[i])
            if result.confidence_intervals_95 is not None
            else None
        )

        rhat = diagnostics.rhat[i] if diagnostics and i < len(diagnostics.rhat) else None
        ess_bulk = (
            diagnostics.ess_bulk[i] if diagnostics and i < len(diagnostics.ess_bulk) else None
        )
        ess_tail = (
            diagnostics.ess_tail[i] if diagnostics and i < len(diagnostics.ess_tail) else None
        )

        param_summaries.append(
            MCMCParameterSummary(
                name=name,
                value=val,
                std_error=err,
                ci_68_lower=ci_68[0] if ci_68 else 0.0,
                ci_68_upper=ci_68[1] if ci_68 else 0.0,
                ci_95_lower=ci_95[0] if ci_95 else 0.0,
                ci_95_upper=ci_95[1] if ci_95 else 0.0,
                rhat=float(rhat) if rhat is not None else None,
                ess_bulk=float(ess_bulk) if ess_bulk is not None else None,
                ess_tail=float(ess_tail) if ess_tail is not None else None,
            )
        )
    return param_summaries


def _extract_amplitude_summaries(
    result: Any, peak_names: list[str], z_values: FloatArray | None
) -> list[MCMCAmplitudeSummary]:
    """Extract amplitude summaries from MCMC result."""
    amp_summaries = []

    n_lines = result.n_lineshape_params
    n_series = result.n_series
    amp_idx_base = n_lines

    for i_p, p_name in enumerate(peak_names):
        for i_series in range(n_series):
            idx = amp_idx_base + i_p * n_series + i_series
            if idx >= len(result.values):
                break

            val = float(result.values[idx])
            err = float(result.std_errors[idx])

            ci_68 = (
                tuple(result.confidence_intervals_68[idx])
                if result.confidence_intervals_68 is not None
                else (0.0, 0.0)
            )
            ci_95 = (
                tuple(result.confidence_intervals_95[idx])
                if result.confidence_intervals_95 is not None
                else (0.0, 0.0)
            )

            z_val = (
                float(z_values[i_series])
                if z_values is not None and i_series < len(z_values)
                else None
            )

            amp_summaries.append(
                MCMCAmplitudeSummary(
                    peak_name=p_name,
                    plane_index=i_series,
                    value=val,
                    std_error=err,
                    ci_68_lower=float(ci_68[0]),
                    ci_68_upper=float(ci_68[1]),
                    ci_95_lower=float(ci_95[0]),
                    ci_95_upper=float(ci_95[1]),
                    z_value=z_val,
                )
            )

    return amp_summaries


def format_mcmc_cluster_result(
    cluster_result: ClusterMCMCResult,
    diagnostics: ConvergenceDiagnostics | None = None,
    z_values: FloatArray | None = None,
) -> MCMCClusterSummary:
    """Convert MCMCClusterResult to display-friendly summary."""
    result = cluster_result.result
    cluster = cluster_result.cluster
    peak_names = [p.name for p in cluster.peaks]

    if diagnostics is None and result.mcmc_diagnostics is not None:
        diagnostics = result.mcmc_diagnostics

    param_summaries = _extract_lineshape_summaries(result, diagnostics)
    amp_summaries = _extract_amplitude_summaries(result, peak_names, z_values)

    cor_matrix = result.correlation_matrix if result.correlation_matrix is not None else None

    return MCMCClusterSummary(
        peak_names=peak_names,
        parameter_summaries=param_summaries,
        amplitude_summaries=amp_summaries,
        correlation_matrix=cor_matrix,
        burn_in_used=result.burn_in_info.get("burn_in", 0) if result.burn_in_info else 0,
        n_chains=diagnostics.n_chains if diagnostics else 0,
        n_samples=diagnostics.n_samples if diagnostics else 0,
    )


__all__ = [
    "MCMCAmplitudeSummary",
    "MCMCClusterSummary",
    "MCMCParameterSummary",
    "PeaksNotFoundError",
    "format_mcmc_cluster_result",
    "run_mcmc_analysis",
]
