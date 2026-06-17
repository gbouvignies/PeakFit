"""JSON summary writer."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import numpy as np

from peakfit.io.schemas import OUTPUT_SCHEMA_VERSION, FitSummarySchema
from peakfit.io.writers.config import WriterConfig
from peakfit.io.writers.utils import JsonValue, NumpyEncoder, canonical_parameter_name

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.fit.results import (
        ClusterEstimates,
        FitResults,
        FitStatistics,
        MCMCDiagnostics,
        ParameterDiagnostic,
        ParameterEstimate,
        RunMetadata,
    )


def write_summary(
    results: FitResults,
    path: Path,
    config: WriterConfig | None = None,
) -> Path:
    """Write the canonical machine-readable fit summary."""
    cfg = config or WriterConfig()
    path.parent.mkdir(parents=True, exist_ok=True)

    output: dict[str, JsonValue] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "metadata": _serialize_metadata(results.metadata),
        "method": results.method,
        "n_clusters": results.n_clusters,
        "n_peaks": results.n_peaks,
        "clusters": [_serialize_cluster(cluster, cfg) for cluster in results.clusters],
    }

    if results.statistics:
        output["statistics"] = [_serialize_statistics(stats, cfg) for stats in results.statistics]

    if results.global_statistics:
        output["global_statistics"] = _serialize_statistics(results.global_statistics, cfg)

    if results.mcmc_diagnostics:
        output["mcmc_diagnostics"] = [
            _serialize_mcmc_diagnostics(diagnostic, cfg) for diagnostic in results.mcmc_diagnostics
        ]

    if results.z_values is not None:
        output["z_axis"] = {"values": results.z_values.tolist()}

    FitSummarySchema.model_validate(output)
    _write_json(output, path)
    return path


def _serialize_metadata(metadata: RunMetadata) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {
        "timestamp": metadata.timestamp,
        "software_version": metadata.software_version,
        "python_version": metadata.python_version,
        "platform": metadata.platform,
    }
    if metadata.git_commit:
        result["git_commit"] = metadata.git_commit
    if metadata.command_line:
        result["command_line"] = metadata.command_line
    if metadata.input_files:
        result["input_files"] = cast("JsonValue", metadata.input_files)
    if metadata.configuration:
        result["configuration"] = metadata.configuration
    if metadata.run_duration_seconds is not None:
        result["run_duration_seconds"] = metadata.run_duration_seconds
    return result


def _serialize_cluster(
    cluster: ClusterEstimates,
    config: WriterConfig,
) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {
        "cluster_id": cluster.cluster_id,
        "peak_names": cast("JsonValue", cluster.peak_names),
        "parameters": [
            _serialize_parameter(parameter, config) for parameter in cluster.lineshape_params
        ],
    }
    if cluster.correlation_matrix is not None:
        result["correlation"] = {
            "parameter_names": cast("JsonValue", cluster.correlation_param_names),
            "matrix": cluster.correlation_matrix.tolist(),
        }
    return result


def _serialize_parameter(
    param: ParameterEstimate,
    config: WriterConfig,
) -> dict[str, JsonValue]:
    precision = config.precision
    threshold = config.scientific_notation_threshold

    result: dict[str, JsonValue] = {
        "name": canonical_parameter_name(param),
        "category": param.category.value,
        "value": _format_value(param.value, precision, threshold),
        "std_error": _format_value(param.std_error, precision, threshold),
        "unit": param.unit,
        "is_fixed": param.is_fixed,
        "is_global": param.is_global,
    }
    if param.param_id is not None:
        result["label"] = param.param_id.label
        result["axis"] = param.param_id.axis or "F0"

    if param.ci_68_lower is not None:
        result["ci_68"] = {
            "lower": _format_value(param.ci_68_lower, precision, threshold),
            "upper": _format_value(param.ci_68_upper, precision, threshold),
        }
    if param.ci_95_lower is not None:
        result["ci_95"] = {
            "lower": _format_value(param.ci_95_lower, precision, threshold),
            "upper": _format_value(param.ci_95_upper, precision, threshold),
        }

    if not np.isinf(param.min_bound):
        result["min_bound"] = _format_value(param.min_bound, precision, threshold)
    if not np.isinf(param.max_bound):
        result["max_bound"] = _format_value(param.max_bound, precision, threshold)

    return result


def _serialize_statistics(
    stats: FitStatistics,
    config: WriterConfig,
) -> dict[str, JsonValue]:
    precision = config.precision
    threshold = config.scientific_notation_threshold

    result: dict[str, JsonValue] = {
        "chi_squared": _format_value(stats.chi_squared, precision, threshold),
        "reduced_chi_squared": _format_value(stats.reduced_chi_squared, precision, threshold),
        "degrees_of_freedom": stats.dof,
        "n_data": stats.n_data,
        "n_params": stats.n_params,
        "fit_converged": stats.fit_converged,
    }
    if stats.aic is not None:
        result["aic"] = _format_value(stats.aic, precision, threshold)
    if stats.bic is not None:
        result["bic"] = _format_value(stats.bic, precision, threshold)
    if stats.log_likelihood is not None:
        result["log_likelihood"] = _format_value(stats.log_likelihood, precision, threshold)
    return result


def _serialize_mcmc_diagnostics(
    diag: MCMCDiagnostics,
    config: WriterConfig,
) -> dict[str, JsonValue]:
    return {
        "overall_status": diag.overall_status.value,
        "converged": diag.converged,
        "n_chains": diag.n_chains,
        "n_samples": diag.n_samples,
        "burn_in": diag.burn_in,
        "burn_in_method": diag.burn_in_method,
        "total_samples": diag.total_samples,
        "warnings": cast("JsonValue", diag.all_warnings),
        "parameters": [
            _serialize_parameter_diagnostic(param, config) for param in diag.parameter_diagnostics
        ],
    }


def _serialize_parameter_diagnostic(
    diagnostic: ParameterDiagnostic,
    config: WriterConfig,
) -> dict[str, JsonValue]:
    precision = config.precision
    return {
        "name": diagnostic.name,
        "rhat": round(diagnostic.rhat, precision) if diagnostic.rhat is not None else None,
        "ess_bulk": diagnostic.ess_bulk,
        "ess_tail": diagnostic.ess_tail,
        "status": diagnostic.status.value,
        "warnings": cast("JsonValue", diagnostic.warnings),
    }


def _format_value(value: float | None, precision: int, threshold: float) -> float | None:
    if value is None:
        return None

    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        return None

    scientific_cutoff = 10**threshold
    if numeric_value != 0 and (
        abs(numeric_value) < 1 / scientific_cutoff or abs(numeric_value) >= scientific_cutoff
    ):
        return float(f"{numeric_value:.{precision}e}")
    return round(numeric_value, precision)


def _write_json(data: dict[str, JsonValue], path: Path) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
        f.write("\n")


__all__ = ["write_summary"]
