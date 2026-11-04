"""Read a fit's scientific archive and MCMC chains from its results directory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from peakfit.io.output_metadata import RunMetadata
from peakfit.io.readers.continuation import load_completed
from peakfit.io.schemas import FitSummarySchema
from peakfit.io.scientific import read_scientific
from peakfit.io.writers.json import _summary_document
from peakfit.shared.paths import format_path

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.engine.scientific.completed import CompletedFit

__all__ = ["ResultsLoader"]
type MCMCChainRecord = tuple[Any, list[str], int, int, int]


class ResultsLoader:
    """Loader for a results directory whose ``scientific.npz`` is the fit's truth.

    Example:
        >>> loader = ResultsLoader(Path("Fits/20260129_120000"))
        >>> summary = loader.load_summary()
        >>> completed = loader.load_completed()
        >>> chains = loader.load_mcmc_chains()
    """

    def __init__(self, directory: Path) -> None:
        """Initialize the loader.

        Args:
            directory: Path to results directory

        Raises:
            FileNotFoundError: If the directory holds no scientific.npz
        """
        self.directory = directory
        self.archive = directory / "scientific.npz"
        if not self.archive.is_file():
            raise FileNotFoundError(
                f"No scientific.npz in {format_path(directory)}; "
                "refit results written by earlier PeakFit versions."
            )

    def load_summary(self) -> FitSummarySchema:
        """Project the archive into the fit JSON summary schema."""
        outcome, _, context = read_scientific(self.archive)
        return FitSummarySchema.model_validate(
            _summary_document(
                outcome,
                metadata=RunMetadata(**context["metadata"]),
                z_values=context["plane_values"],
            )
        )

    def load_completed(self) -> CompletedFit:
        """The validated completed fit, with its observations, for continuation and MCMC."""
        return load_completed(self.archive)

    def load_mcmc_chains(self) -> list[MCMCChainRecord]:
        """Load MCMC chains from HDF5 files for all clusters.

        Returns:
            List of tuples containing:
            - chains: numpy array of shape (n_chains, n_samples, n_params)
            - param_names: list of parameter names
            - cluster_id: cluster identifier
            - burn_in_idx: burn-in index
            - thin: thinning factor
        """
        chains_dir = self.directory / "chains"
        if not chains_dir.exists():
            return []

        return [
            record
            for h5_path in sorted(chains_dir.glob("cluster_*_chains.h5"))
            if (record := _load_mcmc_chain_file(h5_path))
        ]


def _load_mcmc_chain_file(path: Path) -> MCMCChainRecord | None:
    """Load one MCMC chain file, returning None when it is incomplete."""
    try:
        cluster_id = int(path.name.split("_")[1])
        with h5py.File(str(path), "r") as handle:
            group = handle[f"cluster_{cluster_id}"]
            if "chains" not in group:
                return None

            chain = group["chains"][()]
            nonlinear_names = [name.decode() for name in group["nonlinear_names"][:]]
            burn_in, thin = _load_burn_in_metadata(group)
            return chain, nonlinear_names, cluster_id, burn_in, thin
    except (KeyError, OSError, ValueError):
        return None


def _load_burn_in_metadata(group: Any) -> tuple[int, int]:
    """Read burn-in and thinning metadata from an MCMC HDF5 group."""
    if "burn_in" not in group:
        return 0, 1

    burn_in_group = group["burn_in"]
    thin = int(burn_in_group.attrs.get("thin", 1))
    if "burn_in_idx" in burn_in_group.attrs:
        return int(burn_in_group.attrs["burn_in_idx"]), thin
    if "burn_in" in burn_in_group.attrs:
        burn_in = int(np.ceil(int(burn_in_group.attrs["burn_in"]) / thin))
        return burn_in, thin
    return 0, thin
