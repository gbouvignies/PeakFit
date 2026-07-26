"""Assemble structured fit outputs from pipeline state."""

from __future__ import annotations

import hashlib
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from peakfit.engine.algorithms.common import residuals
from peakfit.engine.algorithms.linear_algebra import calculate_amplitudes_with_uncertainty
from peakfit.engine.results import (
    compute_chi_squared,
    compute_reduced_chi_squared,
)
from peakfit.fit.result_models import (
    AmplitudeEstimate,
    ClusterEstimates,
    ConvergenceStatus,
    FitResults,
    FitStatistics,
    MCMCDiagnostics,
    ParameterCategory,
    ParameterDiagnostic,
    ParameterEstimate,
    ResidualStatistics,
    RunMetadata,
)

if TYPE_CHECKING:
    from peakfit.engine.domain.cluster import Cluster
    from peakfit.engine.domain.params_scalar import Parameter, Parameters
    from peakfit.engine.domain.spectrum import Spectra
    from peakfit.engine.domain.state import FittingState

_AMPLITUDE_PARAM_PATTERN = re.compile(r"\.I\d+$")

try:
    __version__ = metadata.version("peakfit")
except metadata.PackageNotFoundError:
    __version__ = "unknown"


def build_fit_results(
    state: FittingState,
    spectra: Spectra,
    config: dict[str, Any],
    input_files: dict[str, Path],
) -> FitResults:
    """Build structured fit results for output writers."""
    metadata = capture_output_metadata(config, input_files)
    noise = state.noise or 1.0
    z_values = spectra.z_values

    clusters: list[ClusterEstimates] = []
    statistics: list[FitStatistics] = []
    for cluster in state.clusters:
        cluster_estimate, cluster_stats = _build_cluster_output(
            cluster=cluster,
            params=state.scalar_params,
            noise=noise,
            z_values=z_values,
        )
        clusters.append(cluster_estimate)
        statistics.append(cluster_stats)

    if not clusters:
        msg = "No cluster estimates available for output."
        raise ValueError(msg)

    return FitResults(
        metadata=metadata,
        clusters=clusters,
        statistics=statistics,
        global_statistics=_build_global_statistics(statistics),
        z_values=z_values,
    )


def capture_output_metadata(config: dict[str, Any], input_files: dict[str, Path]) -> RunMetadata:
    """Capture operational metadata without evaluating any fitted model."""
    run_metadata = RunMetadata(
        timestamp=datetime.now(UTC).isoformat(),
        software_version=__version__,
        git_commit=_current_git_commit(),
        python_version=sys.version,
        platform=platform.platform(),
        configuration=config,
    )
    for name, path in input_files.items():
        if isinstance(path, Path) and path.exists():
            run_metadata.input_files[name] = {
                "path": str(path.name),
                "checksum_sha256": _compute_file_checksum(path),
            }
    return run_metadata


def _current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None

    if result.returncode != 0:
        return None
    return result.stdout.strip()[:12]


def _compute_file_checksum(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_cluster_output(
    cluster: Cluster,
    params: Parameters,
    noise: float,
    z_values: np.ndarray | None,
) -> tuple[ClusterEstimates, FitStatistics]:
    """Build parameter, amplitude, and fit-statistics output for one cluster."""
    shapes = cluster.evaluate(params)
    amplitudes, amplitude_errors, _covariance = calculate_amplitudes_with_uncertainty(
        shapes, cluster.corrected_data.real.astype(float), noise
    )

    cluster_stats = _build_cluster_statistics(cluster, params, noise)
    scale_factor = (
        np.sqrt(cluster_stats.reduced_chi_squared)
        if cluster_stats.reduced_chi_squared > 1.0
        else 1.0
    )
    series_z_values = z_values if z_values is not None else np.arange(amplitudes.shape[1])

    lineshape_params: list[ParameterEstimate] = []
    amplitudes_out: list[AmplitudeEstimate] = []
    for peak_index, peak in enumerate(cluster.peaks):
        lineshape_params.extend(_extract_peak_parameters(peak.name, params))

        for plane_index in range(amplitudes.shape[1]):
            z_value = (
                float(series_z_values[plane_index])
                if plane_index < len(series_z_values)
                else float(plane_index)
            )
            amplitudes_out.append(
                AmplitudeEstimate(
                    peak_name=peak.name,
                    plane_index=plane_index,
                    z_value=z_value,
                    value=float(amplitudes[peak_index, plane_index]),
                    std_error=float(amplitude_errors[peak_index]) * float(scale_factor),
                )
            )

    lineshape_params.extend(_extract_cluster_parameters(cluster.cluster_id, params))

    return (
        ClusterEstimates(
            cluster_id=cluster.cluster_id,
            peak_names=[p.name for p in cluster.peaks],
            lineshape_params=lineshape_params,
            amplitudes=amplitudes_out,
        ),
        cluster_stats,
    )


def _extract_peak_parameters(peak_name: str, params: Parameters) -> list[ParameterEstimate]:
    estimates: list[ParameterEstimate] = []
    for param_name, param in params.items():
        if not param_name.startswith(peak_name + ".") or _is_amplitude_parameter(param_name, param):
            continue
        estimates.append(_to_parameter_estimate(param_name, param))
    return estimates


def _extract_cluster_parameters(cluster_id: int, params: Parameters) -> list[ParameterEstimate]:
    estimates: list[ParameterEstimate] = []
    cluster_prefix = f"cluster_{cluster_id}."
    for param_name, param in params.items():
        if param_name.startswith(cluster_prefix):
            estimates.append(_to_parameter_estimate(param_name, param))
    return estimates


def _to_parameter_estimate(param_name: str, param: Parameter) -> ParameterEstimate:
    return ParameterEstimate(
        name=param_name,
        value=param.value,
        std_error=param.stderr,
        unit=param.unit,
        category=ParameterCategory.LINESHAPE,
        min_bound=param.min,
        max_bound=param.max,
        is_fixed=not param.vary,
        is_global=_is_global_parameter(param_name, param.param_id),
        param_id=param.param_id,
    )


def _is_amplitude_parameter(param_name: str, param: Parameter) -> bool:
    return (param.param_id is not None and param.param_id.label == "I") or bool(
        _AMPLITUDE_PARAM_PATTERN.search(param_name)
    )


def _is_global_parameter(param_name: str, param_id: Any | None) -> bool:
    if param_id is not None and getattr(param_id, "cluster_id", None) is not None:
        return True
    return param_name.startswith("cluster_")


def _build_cluster_statistics(
    cluster: Cluster,
    params: Parameters,
    noise: float,
) -> FitStatistics:
    """Build statistics for a cluster."""
    n_lineshape_params = _count_varying_lineshape_params(cluster, params)
    n_params = n_lineshape_params + cluster.n_amplitude_params
    n_data = cluster.n_observations

    normalized_residuals: np.ndarray | None = None
    try:
        normalized_residuals = residuals(params, cluster, noise)
    except (ValueError, KeyError, AttributeError):
        normalized_residuals = None

    chi_squared = (
        compute_chi_squared(normalized_residuals) if normalized_residuals is not None else 0.0
    )
    aic, bic, log_likelihood = _compute_information_criteria(
        chi_squared=chi_squared,
        n_data=n_data,
        n_params=n_params,
        noise=noise,
    )

    residual_stats = ResidualStatistics(
        raw_residuals=(normalized_residuals * noise) if normalized_residuals is not None else None,
        normalized_residuals=normalized_residuals,
        n_points=n_data,
        n_params=n_params,
        noise_level=noise,
    )

    return FitStatistics(
        chi_squared=chi_squared,
        reduced_chi_squared=compute_reduced_chi_squared(chi_squared, n_data, n_params),
        aic=aic,
        bic=bic,
        log_likelihood=log_likelihood,
        n_data=n_data,
        n_params=n_params,
        residuals=residual_stats,
        fit_converged=True,
        n_function_evals=0,
        fit_message="Statistics computed from fitted model",
    )


def _count_varying_lineshape_params(cluster: Cluster, params: Parameters) -> int:
    cluster_peak_names = {p.name for p in cluster.peaks}
    n_params = 0
    for param_name, param in params.items():
        if not param.vary or _is_amplitude_parameter(param_name, param):
            continue
        belongs_to_peak = any(param_name.startswith(f"{name}.") for name in cluster_peak_names)
        belongs_to_cluster = param_name.startswith(f"cluster_{cluster.cluster_id}.")
        if belongs_to_peak or belongs_to_cluster:
            n_params += 1
    return n_params


def _build_global_statistics(statistics: list[FitStatistics]) -> FitStatistics:
    total_chi_sq = sum(stats.chi_squared for stats in statistics)
    total_params = sum(stats.n_params for stats in statistics)
    total_data = sum(stats.n_data for stats in statistics)
    total_nfev = sum(stats.n_function_evals for stats in statistics)
    all_converged = all(stats.fit_converged for stats in statistics)

    total_log_likelihood: float | None = None
    total_aic: float | None = None
    total_bic: float | None = None
    if statistics and all(stats.log_likelihood is not None for stats in statistics):
        total_log_likelihood = float(
            sum(stats.log_likelihood for stats in statistics if stats.log_likelihood is not None)
        )
        total_aic = -2.0 * total_log_likelihood + 2.0 * total_params
        if total_data > 0:
            total_bic = -2.0 * total_log_likelihood + total_params * float(np.log(total_data))

    return FitStatistics(
        chi_squared=total_chi_sq,
        reduced_chi_squared=compute_reduced_chi_squared(total_chi_sq, total_data, total_params),
        aic=total_aic,
        bic=total_bic,
        log_likelihood=total_log_likelihood,
        n_data=total_data,
        n_params=total_params,
        fit_converged=all_converged,
        n_function_evals=total_nfev,
    )


def _compute_information_criteria(
    chi_squared: float,
    n_data: int,
    n_params: int,
    noise: float,
) -> tuple[float | None, float | None, float | None]:
    """Compute information criteria under Gaussian residual assumptions."""
    if n_data <= 0 or n_params < 0 or noise <= 0:
        return None, None, None

    log_likelihood = -0.5 * chi_squared - n_data * np.log(noise) - 0.5 * n_data * np.log(2 * np.pi)
    aic = -2.0 * log_likelihood + 2.0 * n_params
    bic = -2.0 * log_likelihood + n_params * np.log(n_data)
    return float(aic), float(bic), float(log_likelihood)


__all__ = [
    "AmplitudeEstimate",
    "ClusterEstimates",
    "ConvergenceStatus",
    "FitResults",
    "FitStatistics",
    "MCMCDiagnostics",
    "ParameterCategory",
    "ParameterDiagnostic",
    "ParameterEstimate",
    "ResidualStatistics",
    "RunMetadata",
    "build_fit_results",
    "capture_output_metadata",
]
