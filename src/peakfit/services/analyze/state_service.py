"""Services for loading serialized fitting state artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from peakfit.core.domain.state import FittingState
from peakfit.io.state import StateRepository


@dataclass(slots=True, frozen=True)
class LoadedFittingState:
    """Wrapper containing a loaded fitting state and the source path."""

    state: FittingState
    path: Path


class StateFileMissingError(FileNotFoundError):
    """Raised when no .peakfit_state.pkl file exists in the results directory."""

    def __init__(self, results_dir: Path, state_path: Path) -> None:
        super().__init__(f"No fitting state found in {results_dir}")
        self.results_dir = results_dir
        self.state_path = state_path


class StateLoadError(RuntimeError):
    """Raised when the fitting state file exists but cannot be loaded."""

    def __init__(self, state_path: Path, original_exc: Exception) -> None:
        super().__init__(f"Failed to load fitting state from {state_path}: {original_exc}")
        self.state_path = state_path
        self.original_exc = original_exc


class FittingStateService:
    """Boundary for retrieving fitting states for downstream analysis."""

    @staticmethod
    def load(results_dir: Path) -> LoadedFittingState:
        """Load the fitting state located within *results_dir*."""
        state_path = StateRepository.default_path(results_dir)
        try:
            state = StateRepository.load(state_path)
        except FileNotFoundError as exc:
            raise StateFileMissingError(results_dir, state_path) from exc
        except Exception as exc:
            raise StateLoadError(state_path, exc) from exc

        return LoadedFittingState(state=state, path=state_path)
