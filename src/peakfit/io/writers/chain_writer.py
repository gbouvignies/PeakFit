"""MCMC chain storage writer.

This module provides functionality for storing MCMC chains in a
compact, efficient format that can be loaded for post-hoc analysis.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np

if TYPE_CHECKING:
    from peakfit.core.fitting.advanced import UncertaintyResult
    from peakfit.core.results.fit_results import FitResults


class ChainWriter:
    """Writer for MCMC chain data.

    Stores chains in compressed numpy format (.npz) with metadata
    in accompanying JSON files for efficient storage and retrieval.

    File Structure:
        chains/
        ├── cluster_0_chains.npz     # Compressed chain data
        ├── cluster_0_metadata.json  # Parameter names, shapes, etc.
        ├── cluster_1_chains.npz
        ├── cluster_1_metadata.json
        └── summary.json             # Overall chain storage summary
    """

    def write_chains_from_results(
        self,
        results: FitResults,
        output_dir: Path,
    ) -> dict[str, Path]:
        """Write all MCMC chains from FitResults.

        Args:
            results: FitResults containing MCMC data
            output_dir: Base output directory (chains will go in chains/ subdir)

        Returns
        -------
            Dictionary mapping cluster IDs to written file paths
        """
        chains_dir = output_dir / "chains"
        chains_dir.mkdir(parents=True, exist_ok=True)

        written_files: dict[str, Path] = {}
        cluster_summaries: list[dict] = []

        # Check if we have any MCMC data
        if not results.mcmc_diagnostics:
            return written_files

        # Write summary
        summary = {
            "n_clusters": len(results.clusters),
            "clusters_with_chains": len(cluster_summaries),
            "method": results.method.value,
            "timestamp": results.metadata.timestamp,
        }

        summary_path = chains_dir / "summary.json"
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)
        written_files["summary"] = summary_path

        return written_files

    def write_uncertainty_result(
        self,
        uncertainty_result: UncertaintyResult,
        cluster_id: int,
        output_dir: Path,
    ) -> dict[str, Path]:
        """Write chains from a single UncertaintyResult.

        Args:
            uncertainty_result: UncertaintyResult containing chain data
            cluster_id: Cluster identifier
            output_dir: Output directory for chain files

        Returns
        -------
            Dictionary of written file paths
        """
        chains_dir = output_dir / "chains"
        chains_dir.mkdir(parents=True, exist_ok=True)

        written_files: dict[str, Path] = {}

        # Write chain data if available
        if uncertainty_result.mcmc_chains is not None:
            chains_path = chains_dir / f"cluster_{cluster_id}_chains.npz"
            self._write_chains_npz(
                uncertainty_result.mcmc_chains,
                uncertainty_result.parameter_names,
                chains_path,
            )
            written_files[f"cluster_{cluster_id}_chains"] = chains_path

        # Write flattened samples if no full chains
        elif uncertainty_result.mcmc_samples is not None:
            samples_path = chains_dir / f"cluster_{cluster_id}_samples.npz"
            self._write_samples_npz(
                uncertainty_result.mcmc_samples,
                uncertainty_result.parameter_names,
                samples_path,
            )
            written_files[f"cluster_{cluster_id}_samples"] = samples_path

        # Write metadata
        metadata_path = chains_dir / f"cluster_{cluster_id}_metadata.json"
        metadata = self._build_chain_metadata(uncertainty_result, cluster_id)
        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=2)
        written_files[f"cluster_{cluster_id}_metadata"] = metadata_path

        return written_files

    def _write_chains_npz(
        self,
        chains: np.ndarray,
        parameter_names: list[str],
        path: Path,
    ) -> None:
        """Write chain data to compressed NPZ file.

        Args:
            chains: Chain array of shape (n_walkers, n_steps, n_params)
            parameter_names: List of parameter names
            path: Output path
        """
        # Save with compression
        np.savez_compressed(
            path,
            chains=chains,
            parameter_names=np.array(parameter_names),
        )

    def _write_samples_npz(
        self,
        samples: np.ndarray,
        parameter_names: list[str],
        path: Path,
    ) -> None:
        """Write flattened samples to compressed NPZ file.

        Args:
            samples: Sample array of shape (n_samples, n_params)
            parameter_names: List of parameter names
            path: Output path
        """
        np.savez_compressed(
            path,
            samples=samples,
            parameter_names=np.array(parameter_names),
        )

    def _build_chain_metadata(
        self,
        uncertainty_result: UncertaintyResult,
        cluster_id: int,
    ) -> dict:
        """Build metadata dictionary for chain file.

        Args:
            uncertainty_result: UncertaintyResult object
            cluster_id: Cluster identifier

        Returns
        -------
            Metadata dictionary
        """
        metadata: dict = {
            "cluster_id": cluster_id,
            "parameter_names": uncertainty_result.parameter_names,
            "n_lineshape_params": uncertainty_result.n_lineshape_params,
            "n_planes": uncertainty_result.n_planes,
        }

        # Add shape information
        if uncertainty_result.mcmc_chains is not None:
            chains = uncertainty_result.mcmc_chains
            metadata["chain_shape"] = {
                "n_walkers": chains.shape[0],
                "n_steps": chains.shape[1],
                "n_params": chains.shape[2],
            }
            metadata["storage_type"] = "full_chains"
        elif uncertainty_result.mcmc_samples is not None:
            samples = uncertainty_result.mcmc_samples
            metadata["sample_shape"] = {
                "n_samples": samples.shape[0],
                "n_params": samples.shape[1],
            }
            metadata["storage_type"] = "flattened_samples"

        # Add burn-in info
        if uncertainty_result.burn_in_info:
            metadata["burn_in"] = uncertainty_result.burn_in_info

        # Add diagnostics summary
        if uncertainty_result.mcmc_diagnostics:
            diag = uncertainty_result.mcmc_diagnostics
            metadata["diagnostics"] = {
                "converged": diag.converged,
                "n_chains": diag.n_chains,
                "n_samples": diag.n_samples,
            }

        return metadata


def load_chains(chains_dir: Path, cluster_id: int) -> dict:
    """Load chain data for a specific cluster.

    Args:
        chains_dir: Directory containing chain files
        cluster_id: Cluster identifier

    Returns
    -------
        Dictionary with 'chains' or 'samples' array and 'parameter_names'

    Raises
    ------
        FileNotFoundError: If chain files don't exist
    """
    # Try full chains first
    chains_path = chains_dir / f"cluster_{cluster_id}_chains.npz"
    if chains_path.exists():
        data = np.load(chains_path)
        return {
            "chains": data["chains"],
            "parameter_names": list(data["parameter_names"]),
            "storage_type": "full_chains",
        }

    # Try flattened samples
    samples_path = chains_dir / f"cluster_{cluster_id}_samples.npz"
    if samples_path.exists():
        data = np.load(samples_path)
        return {
            "samples": data["samples"],
            "parameter_names": list(data["parameter_names"]),
            "storage_type": "flattened_samples",
        }

    msg = f"No chain data found for cluster {cluster_id} in {chains_dir}"
    raise FileNotFoundError(msg)


def load_chain_metadata(chains_dir: Path, cluster_id: int) -> dict:
    """Load metadata for a specific cluster's chains.

    Args:
        chains_dir: Directory containing chain files
        cluster_id: Cluster identifier

    Returns
    -------
        Metadata dictionary

    Raises
    ------
        FileNotFoundError: If metadata file doesn't exist
    """
    metadata_path = chains_dir / f"cluster_{cluster_id}_metadata.json"
    if not metadata_path.exists():
        msg = f"No metadata found for cluster {cluster_id} in {chains_dir}"
        raise FileNotFoundError(msg)

    with metadata_path.open() as f:
        return cast("dict[str, Any]", json.load(f))


__all__ = [
    "ChainWriter",
    "load_chain_metadata",
    "load_chains",
]
