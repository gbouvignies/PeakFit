"""Persistence helpers for serialized fitting state artifacts."""

from __future__ import annotations

import pickle
from pathlib import Path

from peakfit.core.domain.state import FittingState

STATE_FILENAME = ".peakfit_state.pkl"


class StateRepository:
    """Infrastructure boundary for saving/loading fitting state blobs."""

    filename: str = STATE_FILENAME

    @classmethod
    def default_path(cls, results_dir: Path) -> Path:
        """Return the conventional state-file path under a results directory."""
        return results_dir / cls.filename

    @classmethod
    def save(cls, path: Path, state: FittingState) -> Path:
        """Serialize the fitting state to *path* and return it."""
        payload = state.to_payload()
        path.parent.mkdir(parents=True, exist_ok=True)
        cls._write(path, payload)
        return path

    @classmethod
    def load(cls, path: Path) -> FittingState:
        """Load a serialized fitting state from disk."""
        payload = cls._read(path)
        return FittingState.from_payload(payload)

    @staticmethod
    def _write(path: Path, state: dict[str, object]) -> None:
        with path.open("wb") as fh:
            pickle.dump(state, fh)

    @staticmethod
    def _read(path: Path) -> dict[str, object]:
        with path.open("rb") as fh:
            return pickle.load(fh)
