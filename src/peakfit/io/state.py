"""Persistence helpers for serialized fitting state artifacts."""

from __future__ import annotations

import pickle
from pathlib import Path

from peakfit.core.domain.state import FittingState

# State file location: cache/state.pkl (no hidden files)
STATE_SUBDIR = "cache"
STATE_FILENAME = "state.pkl"

# Legacy state filename for backward compatibility
LEGACY_STATE_FILENAME = ".peakfit_state.pkl"


class StateRepository:
    """Infrastructure boundary for saving/loading fitting state blobs."""

    subdir: str = STATE_SUBDIR
    filename: str = STATE_FILENAME

    @classmethod
    def default_path(cls, results_dir: Path) -> Path:
        """Return the conventional state-file path under a results directory.

        Returns: results_dir/cache/state.pkl
        """
        return results_dir / cls.subdir / cls.filename

    @classmethod
    def legacy_path(cls, results_dir: Path) -> Path:
        """Return the legacy state-file path for backward compatibility.

        Returns: results_dir/.peakfit_state.pkl
        """
        return results_dir / LEGACY_STATE_FILENAME

    @classmethod
    def find_state_file(cls, results_dir: Path) -> Path | None:
        """Find state file, checking new location first then legacy.

        Args:
            results_dir: Results directory to search

        Returns:
            Path to state file if found, None otherwise
        """
        new_path = cls.default_path(results_dir)
        if new_path.exists():
            return new_path

        legacy_path = cls.legacy_path(results_dir)
        if legacy_path.exists():
            return legacy_path

        return None

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
            payload: dict[str, object] = pickle.load(fh)
            return payload
