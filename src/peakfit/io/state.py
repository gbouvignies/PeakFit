"""Persistence helpers for serialized fitting state artifacts."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

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
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as fh:
        pickle.dump(payload, fh)

    return path


def load_state(path: Path) -> FittingState:
    """Load a serialized fitting state from *path*."""
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    return FittingState.model_validate(payload)
