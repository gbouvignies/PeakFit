"""JSON format output writer."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, cast

import numpy as np

from peakfit.io.writers.config import WriterConfig
from peakfit.io.writers.utils import NumpyEncoder

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.engine.results import (
        AmplitudeEstimate,
        ClusterEstimates,
        FitResults,
        FitStatistics,
        MCMCDiagnostics,
        ParameterDiagnostic,
        ParameterEstimate,
        ResidualStatistics,
        RunMetadata,
    )
    from peakfit.engine.types import JsonValue


class JSONWriter:
    """Writer for JSON output files.

    Produces structured JSON output including:
    - fit_summary.json: Complete results with all parameters
    - run_metadata.json: Reproducibility information
    - mcmc_diagnostics.json: MCMC-specific diagnostics (when applicable)
    """

    def __init__(self, config: WriterConfig | None = None) -> None:
        """Initialize JSON writer.

        Args:
            config: Writer configuration. Defaults to standard settings.
        """
        self.config = config or WriterConfig()

    def write_results(self, results: FitResults, path: Path) -> None:
        """Write complete fit results to JSON.

        This is the main output file containing all fit results
        in a structured, machine-readable format.

        Args:
            results: FitResults object
            path: Output file path (e.g., results/summary/fit_summary.json)
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        output: dict[str, JsonValue] = {
            "schema_version": "1.0.0",
            "metadata": self._serialize_metadata(results.metadata),
            "method": results.method.value,
            "n_clusters": results.n_clusters,
            "n_peaks": results.n_peaks,
            "clusters": [self._serialize_cluster(cluster) for cluster in results.clusters],
        }

        # Statistics (list, one per cluster)
        if results.statistics:
            output["statistics"] = [self._serialize_statistics(s) for s in results.statistics]

        # Global statistics
        if results.global_statistics:
            output["global_statistics"] = self._serialize_statistics(results.global_statistics)

        # Include MCMC diagnostics if available
        if results.mcmc_diagnostics:
            output["mcmc_diagnostics"] = [
                self._serialize_mcmc_diagnostics(d) for d in results.mcmc_diagnostics
            ]

        # Model comparisons
        if results.model_comparisons:
            output["model_comparisons"] = [m.to_dict() for m in results.model_comparisons]

        # Add z-axis information
        if results.z_values is not None:
            output["z_axis"] = {
                "values": results.z_values.tolist(),
            }

        self._write_json(output, path)

    def write_metadata(self, results: FitResults, path: Path) -> None:
        """Write run metadata to separate JSON file.

        This file contains only reproducibility information,
        useful for tracking provenance.

        Args:
            results: FitResults object
            path: Output file path (e.g., results/metadata/run_metadata.json)
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            "schema_version": "1.0.0",
            "metadata": self._serialize_metadata(results.metadata),
            "fit_configuration": {
                "method": results.method.value,
                "n_clusters": len(results.clusters),
                "total_parameters": sum(
                    len(c.lineshape_params) + len(c.amplitudes) for c in results.clusters
                ),
            },
        }

        self._write_json(output, path)

    def write_statistics(self, results: FitResults, path: Path) -> None:
        """Write fit statistics to JSON.

        Args:
            results: FitResults object
            path: Output file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        output: dict[str, JsonValue] = {
            "schema_version": "1.0.0",
            "generated": datetime.now().isoformat(),
        }

        if results.global_statistics:
            output["global_statistics"] = self._serialize_statistics(results.global_statistics)

        if results.statistics:
            output["cluster_statistics"] = [
                {
                    "cluster_id": results.clusters[i].cluster_id,
                    **self._serialize_statistics(s),
                }
                for i, s in enumerate(results.statistics)
                if i < len(results.clusters)
            ]

        self._write_json(output, path)

    def write_diagnostics(self, results: FitResults, path: Path) -> None:
        """Write MCMC diagnostics to separate JSON file.

        Only written when MCMC diagnostics are available.

        Args:
            results: FitResults object
            path: Output file path (e.g., results/diagnostics/mcmc_diagnostics.json)
        """
        if not results.mcmc_diagnostics:
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize all diagnostics (one per cluster)
        all_diagnostics = [self._serialize_mcmc_diagnostics(d) for d in results.mcmc_diagnostics]

        # Generate interpretation based on overall status
        overall_converged = results.has_converged
        interpretation = {
            "summary": "All clusters converged."
            if overall_converged
            else "Some clusters have convergence issues.",
            "n_clusters_analyzed": len(results.mcmc_diagnostics),
        }

        output = {
            "schema_version": "1.0.0",
            "generated": datetime.now().isoformat(),
            "overall_converged": overall_converged,
            "diagnostics": all_diagnostics,
            "interpretation": interpretation,
        }

        self._write_json(output, path)

    def write_parameters_json(self, results: FitResults, path: Path) -> None:
        """Write parameters in JSON format (alternative to CSV).

        Useful for programmatic access when CSV parsing is inconvenient.

        Args:
            results: FitResults object
            path: Output file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        parameters = [
            {
                "cluster_id": cluster.cluster_id,
                "peak_names": cast("JsonValue", cluster.peak_names),
                **self._serialize_parameter(param),
            }
            for cluster in results.clusters
            for param in cluster.lineshape_params
        ]

        output = {
            "schema_version": "1.0.0",
            "generated": datetime.now().isoformat(),
            "total_parameters": len(parameters),
            "parameters": parameters,
        }

        self._write_json(output, path)

    def write_amplitudes_json(self, results: FitResults, path: Path) -> None:
        """Write amplitudes in JSON format (alternative to CSV).

        Args:
            results: FitResults object
            path: Output file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        amplitudes = [
            {
                "cluster_id": cluster.cluster_id,
                **self._serialize_amplitude(amp),
            }
            for cluster in results.clusters
            for amp in cluster.amplitudes
        ]

        output = {
            "schema_version": "1.0.0",
            "generated": datetime.now().isoformat(),
            "total_amplitudes": len(amplitudes),
            "amplitudes": amplitudes,
        }

        self._write_json(output, path)

    # ----------------------------------------------------------------
    # Serialization helpers
    # ----------------------------------------------------------------

    def _serialize_metadata(self, metadata: RunMetadata) -> dict[str, JsonValue]:
        """Serialize run metadata to dict."""
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

    def _serialize_cluster(self, cluster: ClusterEstimates) -> dict[str, JsonValue]:
        """Serialize a cluster's results."""
        result: dict[str, JsonValue] = {
            "cluster_id": cluster.cluster_id,
            "peak_names": cast("JsonValue", cluster.peak_names),
            "parameters": [self._serialize_parameter(p) for p in cluster.lineshape_params],
        }
        if self.config.include_amplitudes_in_summary:
            result["amplitudes"] = [self._serialize_amplitude(a) for a in cluster.amplitudes]
        # Add correlation matrix if available
        if cluster.correlation_matrix is not None:
            result["correlation"] = {
                "parameter_names": cast("JsonValue", cluster.correlation_param_names),
                "matrix": cluster.correlation_matrix.tolist(),
            }
        return result

    def _serialize_parameter(self, param: ParameterEstimate) -> dict[str, JsonValue]:
        """Serialize a parameter estimate."""
        prec = self.config.precision
        thresh = self.config.scientific_notation_threshold

        if param.param_id is not None and not param.param_id.axis:
            entity = (
                f"cluster_{param.param_id.cluster_id}"
                if param.param_id.cluster_id is not None
                else param.param_id.peak_name
            )
            suffix = (
                f"{param.param_id.label}{param.param_id.index}"
                if param.param_id.index is not None
                else param.param_id.label
            )
            canonical_name = f"{entity}.F0.{suffix}"
        else:
            canonical_name = param.param_id.name if param.param_id is not None else param.name

        result: dict[str, JsonValue] = {
            "name": canonical_name,
            "category": param.category.value,
            "value": self._format_value(param.value, prec, thresh),
            "std_error": self._format_value(param.std_error, prec, thresh),
            "unit": param.unit,
            "is_fixed": param.is_fixed,
            "is_global": param.is_global,
        }
        if param.param_id is not None:
            result["label"] = param.param_id.label
            result["axis"] = param.param_id.axis or "F0"

        # Add credible intervals if available
        if param.ci_68_lower is not None:
            result["ci_68"] = {
                "lower": self._format_value(param.ci_68_lower, prec, thresh),
                "upper": self._format_value(param.ci_68_upper, prec, thresh),
            }
        if param.ci_95_lower is not None:
            result["ci_95"] = {
                "lower": self._format_value(param.ci_95_lower, prec, thresh),
                "upper": self._format_value(param.ci_95_upper, prec, thresh),
            }

        # Add bounds (only if finite)
        if not np.isinf(param.min_bound):
            result["min_bound"] = self._format_value(param.min_bound, prec, thresh)
        if not np.isinf(param.max_bound):
            result["max_bound"] = self._format_value(param.max_bound, prec, thresh)

        return result

    def _serialize_amplitude(self, amp: AmplitudeEstimate) -> dict[str, JsonValue]:
        """Serialize an amplitude estimate."""
        prec = self.config.precision
        thresh = self.config.scientific_notation_threshold

        result: dict[str, JsonValue] = {
            "peak_name": amp.peak_name,
            "plane_index": amp.plane_index,
            "value": self._format_value(amp.value, prec, thresh),
            "std_error": self._format_value(amp.std_error, prec, thresh),
        }

        if amp.z_value is not None:
            result["z_value"] = self._format_value(amp.z_value, prec, thresh)

        if amp.ci_68_lower is not None:
            result["ci_68"] = {
                "lower": self._format_value(amp.ci_68_lower, prec, thresh),
                "upper": self._format_value(amp.ci_68_upper, prec, thresh),
            }

        return result

    def _serialize_statistics(self, stats: FitStatistics) -> dict[str, JsonValue]:
        """Serialize fit statistics."""
        prec = self.config.precision
        thresh = self.config.scientific_notation_threshold

        result: dict[str, JsonValue] = {
            "chi_squared": self._format_value(stats.chi_squared, prec, thresh),
            "reduced_chi_squared": self._format_value(stats.reduced_chi_squared, prec, thresh),
            "degrees_of_freedom": stats.dof,
            "n_data": stats.n_data,
            "n_params": stats.n_params,
            "fit_converged": stats.fit_converged,
        }
        if stats.aic is not None:
            result["aic"] = self._format_value(stats.aic, prec, thresh)
        if stats.bic is not None:
            result["bic"] = self._format_value(stats.bic, prec, thresh)
        if stats.log_likelihood is not None:
            result["log_likelihood"] = self._format_value(stats.log_likelihood, prec, thresh)
        return result

    def _serialize_residuals(
        self, residuals: ResidualStatistics | None
    ) -> dict[str, JsonValue] | None:
        """Serialize residual statistics."""
        if residuals is None:
            return None

        prec = self.config.precision
        thresh = self.config.scientific_notation_threshold

        return {
            "n_points": residuals.n_points,
            "n_params": residuals.n_params,
            "degrees_of_freedom": residuals.dof,
            "noise_level": self._format_value(residuals.noise_level, prec, thresh),
            "sum_squared": self._format_value(residuals.sum_squared, prec, thresh),
            "rms": self._format_value(residuals.rms, prec, thresh),
            "mean": self._format_value(residuals.mean, prec, thresh),
            "std": self._format_value(residuals.std, prec, thresh),
        }

    def _serialize_mcmc_diagnostics(self, diag: MCMCDiagnostics) -> dict[str, JsonValue]:
        """Serialize MCMC diagnostics."""
        return {
            "overall_status": diag.overall_status.value,
            "converged": diag.converged,
            "n_chains": diag.n_chains,
            "n_samples": diag.n_samples,
            "burn_in": diag.burn_in,
            "burn_in_method": diag.burn_in_method,
            "total_samples": diag.total_samples,
            "warnings": cast("JsonValue", diag.all_warnings),
            "parameter_diagnostics": [
                self._serialize_parameter_diagnostic(pd) for pd in diag.parameter_diagnostics
            ],
        }

    def _serialize_parameter_diagnostic(self, pd: ParameterDiagnostic) -> dict[str, JsonValue]:
        """Serialize a single parameter's MCMC diagnostics."""
        prec = self.config.precision

        return {
            "name": pd.name,
            "rhat": round(pd.rhat, prec) if pd.rhat is not None else None,
            "ess_bulk": pd.ess_bulk,
            "ess_tail": pd.ess_tail,
            "status": pd.status.value,
            "warnings": cast("JsonValue", pd.warnings),
        }

    def _format_value(self, value: float | None, precision: int, threshold: float) -> float | None:
        """Format a numeric value, returning None for None input."""
        if value is None:
            return None

        numeric_value = float(value)
        if not np.isfinite(numeric_value):
            return None

        # Keep numeric JSON values while preserving readability for extreme scales.
        scientific_cutoff = 10**threshold
        if numeric_value != 0 and (
            abs(numeric_value) < 1 / scientific_cutoff or abs(numeric_value) >= scientific_cutoff
        ):
            return float(f"{numeric_value:.{precision}e}")
        return round(numeric_value, precision)

    def _write_json(self, data: dict[str, JsonValue], path: Path) -> None:
        """Write data to JSON file with proper formatting."""
        with path.open("w") as f:
            json.dump(
                data,
                f,
                indent=2,
                cls=NumpyEncoder,
                ensure_ascii=False,
            )
            f.write("\n")  # Trailing newline


__all__ = [
    "JSONWriter",
]
