"""Services for orchestrating MCMC uncertainty analysis."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from peakfit.core.domain.cluster import Cluster
from peakfit.core.domain.peaks import Peak, create_params
from peakfit.core.domain.state import FittingState
from peakfit.core.fitting.advanced import UncertaintyResult, estimate_uncertainties_mcmc
from peakfit.core.fitting.parameters import Parameters


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


class PeaksNotFoundError(ValueError):
    """Raised when requested peaks cannot be matched to clusters."""

    def __init__(self, peaks: Iterable[str]):
        peaks_list = list(peaks)
        super().__init__(f"No clusters found for peaks: {peaks_list}")
        self.peaks = peaks_list


class MCMCAnalysisService:
    """High-level service for running MCMC uncertainty estimation."""

    @staticmethod
    def run(
        state: FittingState,
        *,
        peaks: list[str] | None,
        n_walkers: int,
        n_steps: int,
        burn_in: int | None,
        auto_burnin: bool,
    ) -> MCMCAnalysisResult:
        """Run MCMC analysis for the provided fitting state."""
        clusters = _filter_clusters(state.clusters, peaks)
        if peaks and not clusters:
            raise PeaksNotFoundError(peaks)

        results: list[ClusterMCMCResult] = []
        params = state.params
        burn_in_arg = None if auto_burnin else burn_in

        for cluster in clusters:
            cluster_params = _create_cluster_params(cluster, params)
            result = estimate_uncertainties_mcmc(
                cluster_params,
                cluster,
                state.noise,
                n_walkers=n_walkers,
                n_steps=n_steps,
                burn_in=burn_in_arg,
            )
            _update_global_errors(params, result)
            results.append(ClusterMCMCResult(cluster=cluster, result=result))

        return MCMCAnalysisResult(
            clusters=clusters,
            params=params,
            noise=state.noise,
            peaks=state.peaks,
            cluster_results=results,
        )


def _filter_clusters(clusters: list[Cluster], peaks: list[str] | None) -> list[Cluster]:
    if not peaks:
        return list(clusters)
    peak_set = set(peaks)
    return [cluster for cluster in clusters if any(p.name in peak_set for p in cluster.peaks)]


def _create_cluster_params(cluster: Cluster, params_all: Parameters) -> Parameters:
    cluster_params = create_params(cluster.peaks)
    for key in cluster_params:
        if key in params_all:
            cluster_params[key].value = params_all[key].value
            cluster_params[key].stderr = params_all[key].stderr
    return cluster_params


def _update_global_errors(params_all: Parameters, result: UncertaintyResult) -> None:
    for idx, name in enumerate(result.parameter_names):
        if name in params_all:
            params_all[name].stderr = result.std_errors[idx]
