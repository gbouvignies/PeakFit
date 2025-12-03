"""YAML output writer for PeakFit results.

Produces human-readable YAML files with full metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import yaml

from peakfit.io.writers.json_writer import JSONWriter

if TYPE_CHECKING:
    from peakfit.core.results.fit_results import FitResults


class YAMLWriter(JSONWriter):
    """Writer for YAML output files.

    Inherits serialization logic from JSONWriter but outputs YAML.
    """

    def write_results(self, results: FitResults, path: Path) -> None:
        """Write complete fit results to YAML.

        Args:
            results: FitResults object
            path: Output file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Reuse JSON serialization logic to get a dict
        output = {
            "schema_version": "1.0.0",
            "metadata": self._serialize_metadata(results.metadata),
            "method": results.method.value,
            "experiment_type": results.experiment_type,
            "n_clusters": results.n_clusters,
            "n_peaks": results.n_peaks,
            "clusters": [self._serialize_cluster(cluster) for cluster in results.clusters],
        }

        if results.statistics:
            output["statistics"] = [self._serialize_statistics(s) for s in results.statistics]

        if results.global_statistics:
            output["global_statistics"] = self._serialize_statistics(results.global_statistics)

        if results.mcmc_diagnostics:
            output["mcmc_diagnostics"] = [
                self._serialize_mcmc_diagnostics(d) for d in results.mcmc_diagnostics
            ]

        if results.model_comparisons:
            output["model_comparisons"] = [m.to_dict() for m in results.model_comparisons]

        if results.z_values is not None:
            output["z_axis"] = {
                "values": results.z_values.tolist(),
                "unit": results.z_unit or "",
            }

        self._write_yaml(output, path)

    def _write_yaml(self, data: dict[str, Any], path: Path) -> None:
        """Write data to YAML file."""
        with path.open("w") as f:
            yaml.dump(data, f, sort_keys=False, default_flow_style=False)
