"""Persistence helpers for serialized fitting state artifacts."""

from __future__ import annotations

import pickle
import warnings
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.domain.state import FittingState

if TYPE_CHECKING:
    from pathlib import Path

# State file location: metadata/fitting_state.pkl
STATE_SUBDIR = "metadata"
STATE_FILENAME = "fitting_state.pkl"


def default_state_path(results_dir: Path) -> Path:
    """Return the conventional state-file path under a results directory.

    Returns: results_dir/metadata/fitting_state.pkl
    """
    return results_dir / STATE_SUBDIR / STATE_FILENAME


def save_state(path: Path, state: FittingState) -> Path:
    """Serialize the fitting state to *path* and return it."""
    payload = state.model_dump()
    payload["terminal_corrections"] = {c.cluster_id: c.corrections for c in state.clusters}
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as fh:
        pickle.dump(payload, fh)

    return path


def load_state(path: Path, *, require_corrections: bool = False) -> FittingState:
    """Load a serialized fitting state from *path*."""
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    corrections = payload.pop("terminal_corrections", None)
    state = FittingState.model_validate(payload)
    if corrections is None:
        if require_corrections:
            raise ValueError(
                "Legacy pickle omitted multi-pass terminal corrections. "
                "Use its scientific.npz evidence or refit before continuation."
            )
        warnings.warn(
            "Legacy pickle omitted terminal corrections. Recorded values cannot be recovered "
            "without scientific evidence; refit before exact continuation.",
            RuntimeWarning,
            stacklevel=2,
        )
    if corrections is not None:
        if set(corrections) != {c.cluster_id for c in state.clusters}:
            raise ValueError("Continuation correction cluster identities disagree.")
        for cluster in state.clusters:
            value = np.asarray(corrections[cluster.cluster_id])
            if value.shape != cluster.data.shape or not np.all(np.isfinite(value)):
                raise ValueError("Invalid continuation corrections.")
            cluster.corrections = value.copy()
    return state
