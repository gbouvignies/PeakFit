"""Tests for the analyze-layer fitting state service."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest  # type: ignore[import-not-found]

import numpy as np

from peakfit.core.domain.cluster import Cluster
from peakfit.core.domain.peaks import Peak
from peakfit.core.domain.state import FittingState
from peakfit.core.fitting.parameters import Parameters
from peakfit.io.state import StateRepository
from peakfit.services.analyze import FittingStateService, StateFileMissingError, StateLoadError


def _make_peak(name: str = "peak-a") -> Peak:
    """Create a minimal valid Peak object for testing."""
    return Peak(name=name, positions=np.array([1.0, 2.0]), shapes=[])


def _make_cluster(cluster_id: int = 0, peak_names: list[str] | None = None) -> Cluster:
    """Create a minimal valid Cluster object for testing."""
    if peak_names is None:
        peak_names = ["peak-a"]
    peaks = [_make_peak(name) for name in peak_names]
    return Cluster(
        cluster_id=cluster_id,
        peaks=peaks,
        positions=[np.array([0, 1, 2])],
        data=np.zeros((3,)),
    )


def _make_state() -> FittingState:
    params = Parameters()
    params.add("amp", value=1.0)
    peak = _make_peak("peak-a")
    cluster = _make_cluster(0, ["peak-a"])
    return FittingState(clusters=[cluster], params=params, noise=0.5, peaks=[peak], version="1.0")


def test_service_loads_state(tmp_path: Path) -> None:
    results_dir = tmp_path / "Fits"
    results_dir.mkdir()

    state_path = StateRepository.default_path(results_dir)
    StateRepository.save(state_path, _make_state())

    loaded = FittingStateService.load(results_dir)

    assert loaded.path == state_path
    assert loaded.state.noise == pytest.approx(0.5)
    assert loaded.state.peaks[0].name == "peak-a"


def test_service_missing_state(tmp_path: Path) -> None:
    results_dir = tmp_path / "Fits"
    results_dir.mkdir()

    with pytest.raises(StateFileMissingError) as excinfo:
        FittingStateService.load(results_dir)

    missing_error = excinfo.value
    assert missing_error.results_dir == results_dir
    assert missing_error.state_path == StateRepository.default_path(results_dir)


def test_service_load_error(tmp_path: Path) -> None:
    results_dir = tmp_path / "Fits"
    results_dir.mkdir()

    bad_state = StateRepository.default_path(results_dir)
    bad_state.parent.mkdir(parents=True, exist_ok=True)
    bad_state.write_text("not-a-pickle")

    with pytest.raises(StateLoadError) as excinfo:
        FittingStateService.load(results_dir)

    assert excinfo.value.state_path == bad_state
    assert isinstance(excinfo.value.original_exc, Exception)
