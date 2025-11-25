"""Services for computing profile likelihood confidence intervals."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from scipy.stats import chi2  # type: ignore[import-not-found]

from peakfit.core.domain.cluster import Cluster
from peakfit.core.domain.peaks import create_params
from peakfit.core.domain.state import FittingState
from peakfit.core.fitting.advanced import compute_profile_likelihood
from peakfit.core.fitting.parameters import Parameters
from peakfit.core.shared.typing import FloatArray


@dataclass(slots=True, frozen=True)
class ProfileParameterResult:
    """Profile likelihood output for a single parameter."""

    parameter_name: str
    cluster: Cluster
    parameter_values: FloatArray
    chi2_values: FloatArray
    ci_low: float
    ci_high: float
    best_value: float
    covariance_stderr: float


@dataclass(slots=True, frozen=True)
class ProfileLikelihoodAnalysisResult:
    """Aggregate result for a profile likelihood computation."""

    target_parameters: list[str]
    confidence_level: float
    delta_chi2: float
    results: list[ProfileParameterResult]
    missing_parameters: list[str]


class NoVaryingParametersError(RuntimeError):
    """Raised when the fitting state does not contain varying parameters."""


class ParameterMatchError(ValueError):
    """Raised when no parameters match the requested pattern."""

    def __init__(self, pattern: str, available: Iterable[str]):
        available_list = list(available)
        super().__init__(f"No parameters matching '{pattern}' found")
        self.pattern = pattern
        self.available = available_list


class ProfileLikelihoodService:
    """Service responsible for running profile likelihood analysis."""

    @staticmethod
    def run(
        state: FittingState,
        *,
        param_name: str | None,
        n_points: int,
        confidence_level: float,
    ) -> ProfileLikelihoodAnalysisResult:
        params = state.params
        all_param_names = params.get_vary_names()
        if not all_param_names:
            raise NoVaryingParametersError("No varying parameters found in fitting state")

        if param_name is None:
            target_parameters = list(all_param_names)
        else:
            target_parameters = _match_parameters(param_name, all_param_names)
            if not target_parameters:
                raise ParameterMatchError(param_name, all_param_names)

        delta_chi2 = float(chi2.ppf(confidence_level, df=1))
        cache: dict[int, Parameters] = {}
        results: list[ProfileParameterResult] = []
        missing_parameters: list[str] = []

        for target in target_parameters:
            cluster, cluster_params = _locate_cluster_params(state.clusters, params, target, cache)
            if cluster is None or cluster_params is None:
                missing_parameters.append(target)
                continue

            param_vals, chi2_vals, (ci_low, ci_high) = compute_profile_likelihood(
                cluster_params,
                cluster,
                state.noise,
                param_name=target,
                n_points=n_points,
                delta_chi2=delta_chi2,
            )

            best_value = cluster_params[target].value
            covariance_stderr = cluster_params[target].stderr

            results.append(
                ProfileParameterResult(
                    parameter_name=target,
                    cluster=cluster,
                    parameter_values=param_vals,
                    chi2_values=chi2_vals,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    best_value=best_value,
                    covariance_stderr=covariance_stderr,
                )
            )

        return ProfileLikelihoodAnalysisResult(
            target_parameters=target_parameters,
            confidence_level=confidence_level,
            delta_chi2=delta_chi2,
            results=results,
            missing_parameters=missing_parameters,
        )


def _locate_cluster_params(
    clusters: list[Cluster],
    global_params: Parameters,
    target_param: str,
    cache: dict[int, Parameters],
) -> tuple[Cluster | None, Parameters | None]:
    for cluster in clusters:
        cluster_params = _get_cluster_params(cluster, global_params, cache)
        if target_param in cluster_params:
            return cluster, cluster_params
    return None, None


def _get_cluster_params(
    cluster: Cluster,
    global_params: Parameters,
    cache: dict[int, Parameters],
) -> Parameters:
    cache_key = id(cluster)
    if cache_key not in cache:
        cluster_params = create_params(cluster.peaks)
        for key in cluster_params:
            if key in global_params:
                cluster_params[key].value = global_params[key].value
                cluster_params[key].stderr = global_params[key].stderr
        cache[cache_key] = cluster_params
    return cache[cache_key]


def _match_parameters(pattern: str, all_params: list[str]) -> list[str]:
    if pattern in all_params:
        return [pattern]

    matches: list[str] = []
    pattern_lower = pattern.lower()
    for param in all_params:
        param_lower = param.lower()

        if "_" in param:
            peak_name, param_type = param.rsplit("_", 1)
            if (
                pattern_lower == peak_name.lower()
                or pattern_lower == param_type.lower()
                or pattern_lower in param_lower
            ):
                matches.append(param)
        else:
            if pattern_lower in param_lower:
                matches.append(param)

    return matches


__all__ = [
    "NoVaryingParametersError",
    "ParameterMatchError",
    "ProfileLikelihoodAnalysisResult",
    "ProfileLikelihoodService",
    "ProfileParameterResult",
]
