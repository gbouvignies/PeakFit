"""JSON 4.0.0 projection of immutable completed-fit outcomes."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from peakfit.io.schemas import OUTPUT_SCHEMA_VERSION, FitSummarySchema

if TYPE_CHECKING:
    from peakfit.fit.final_outcome import (
        FinalAnalyticalEvaluation,
        FinalClusterOutcome,
        FinalFitOutcome,
        FinalFitStatistics,
        FinalParameter,
        OptimizerProvenance,
    )
    from peakfit.fit.output_metadata import RunMetadata


def write_final_outcome_summary(
    outcome: FinalFitOutcome,
    *,
    metadata: RunMetadata,
    z_values: np.ndarray | None,
    path: Path,
) -> Path:
    """Write a completed-fit JSON document without consulting mutable state.

    This is intentionally a narrow adapter. JSON independently projects the
    same authoritative outcome as the human and tabular writers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    output = _summary_document(outcome, metadata=metadata, z_values=z_values)
    FitSummarySchema.model_validate(output)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return path


def _summary_document(
    outcome: FinalFitOutcome,
    *,
    metadata: RunMetadata,
    z_values: np.ndarray | None,
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "metadata": _serialize_metadata(metadata),
        "terminal_correction_revision": outcome.terminal_correction_revision,
        "noise": _finite_value(outcome.noise),
        "final_nonlinear_parameters": [
            _serialize_parameter(parameter) for parameter in outcome.final_nonlinear_parameters
        ],
        "clusters": [_serialize_cluster(cluster) for cluster in outcome.clusters],
        "statistics": _serialize_final_statistics(outcome.statistics),
    }
    if z_values is not None:
        document["z_axis"] = {"values": _json_value(np.asarray(z_values))}
    return document


def _serialize_metadata(metadata: RunMetadata) -> dict[str, Any]:
    return {
        "timestamp": metadata.timestamp,
        "software_version": metadata.software_version,
        "git_commit": metadata.git_commit,
        "python_version": metadata.python_version,
        "platform": metadata.platform,
        "input_files": _json_value(metadata.input_files),
        "configuration": _json_value(metadata.configuration),
        "command_line": metadata.command_line,
        "run_duration_seconds": _optional_finite_value(metadata.run_duration_seconds),
    }


def _serialize_cluster(cluster: FinalClusterOutcome) -> dict[str, Any]:
    return {
        "cluster_id": cluster.cluster_id,
        "peak_names": list(cluster.peak_names),
        "classification": cluster.classification.value,
        "unusable_reason": cluster.unusable_reason,
        "correction_revision": cluster.correction_revision,
        "optimizer_provenance": _serialize_provenance(cluster.optimizer_provenance),
        "final_nonlinear_parameters": [
            _serialize_parameter(parameter) for parameter in cluster.final_nonlinear_parameters
        ],
        "analytical_evaluation": (
            _serialize_evaluation(cluster.analytical_evaluation)
            if cluster.analytical_evaluation is not None
            else None
        ),
    }


def _serialize_parameter(parameter: FinalParameter) -> dict[str, Any]:
    return {
        "name": parameter.name,
        "value": _finite_value(parameter.value),
        "min_bound": _optional_finite_value(parameter.min),
        "max_bound": _optional_finite_value(parameter.max),
        "vary": parameter.vary,
        "unit": parameter.unit,
        "standard_error": _optional_finite_value(parameter.standard_error),
    }


def _serialize_evaluation(evaluation: FinalAnalyticalEvaluation) -> dict[str, Any]:
    return {
        "shapes": _json_value(evaluation.shapes),
        "amplitudes": _json_value(evaluation.amplitudes),
        "amplitude_standard_errors": _json_value(evaluation.amplitude_standard_errors),
        "amplitude_covariance": _json_value(evaluation.amplitude_covariance),
        "scaled_amplitude_standard_errors": _json_value(
            evaluation.scaled_amplitude_standard_errors
        ),
        "model_values": _json_value(evaluation.model_values),
        "raw_residuals": _json_value(evaluation.raw_residuals),
        "normalized_residuals": _json_value(evaluation.normalized_residuals),
        "statistics": {
            "chi_squared": _finite_value(evaluation.statistics.chi_squared),
            "n_observations": evaluation.statistics.n_observations,
            "n_nonlinear_parameters": evaluation.statistics.n_nonlinear_parameters,
            "n_amplitude_parameters": evaluation.statistics.n_amplitude_parameters,
            "n_fitted_parameters": evaluation.statistics.n_fitted_parameters,
            "degrees_of_freedom": evaluation.statistics.degrees_of_freedom,
            "reduced_chi_squared": _finite_value(evaluation.statistics.reduced_chi_squared),
            "amplitude_uncertainty_scale": _finite_value(
                evaluation.statistics.amplitude_uncertainty_scale
            ),
            "aic": _finite_value(evaluation.statistics.aic),
            "bic": _finite_value(evaluation.statistics.bic),
            "log_likelihood": _finite_value(evaluation.statistics.log_likelihood),
        },
    }


def _serialize_provenance(provenance: OptimizerProvenance) -> dict[str, Any]:
    result: dict[str, Any] = {
        "success": provenance.converged,
        "correction_revision": provenance.correction_revision,
        "metadata": _json_value(provenance.metadata),
    }
    optional_values = {
        "optimizer_kind": provenance.optimizer_kind,
        "termination_message": provenance.termination_message,
        "function_evaluations": provenance.function_evaluations,
        "jacobian_evaluations": provenance.jacobian_evaluations,
        "iterations": provenance.iterations,
        "optimality": _optional_finite_value(provenance.optimality),
        "final_cost": _optional_finite_value(provenance.final_cost),
    }
    result.update({key: value for key, value in optional_values.items() if value is not None})
    return result


def _serialize_final_statistics(statistics: FinalFitStatistics) -> dict[str, Any]:
    return {
        "chi_squared": _finite_value(statistics.chi_squared),
        "reduced_chi_squared": _finite_value(statistics.reduced_chi_squared),
        "n_observations": statistics.n_observations,
        "n_fitted_parameters": statistics.n_fitted_parameters,
        "degrees_of_freedom": statistics.degrees_of_freedom,
        "aic": _optional_finite_value(statistics.aic),
        "bic": _optional_finite_value(statistics.bic),
        "log_likelihood": _optional_finite_value(statistics.log_likelihood),
        "function_evaluations": statistics.function_evaluations,
    }


def _finite_value(value: float) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("Completed-fit JSON cannot represent a non-finite scientific value.")
    return converted


def _optional_finite_value(value: float | None) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _json_value(value: Any) -> Any:
    """Copy immutable and NumPy values into JSON primitives without mutation."""
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        value = sorted(value, key=repr)
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        return _finite_value(value)
    if isinstance(value, (str, Path, int, bool)) or value is None:
        return str(value) if isinstance(value, Path) else value
    raise TypeError(f"Cannot serialize {type(value).__name__} as JSON provenance metadata.")


__all__ = ["write_final_outcome_summary"]
