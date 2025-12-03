"""Tests for the infrastructure state repository and fitting state model."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np

from peakfit.core.domain.cluster import Cluster
from peakfit.core.domain.peaks import Peak
from peakfit.core.domain.state import FittingState
from peakfit.core.fitting.parameters import Parameters
from peakfit.io.state import StateRepository


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
    params["amp"].stderr = 0.1
    peak = _make_peak("peak-a")
    cluster = _make_cluster(0, ["peak-a"])
    return FittingState(
        clusters=[cluster],
        params=params,
        noise=0.5,
        peaks=[peak],
        version="1.0",
    )


def test_fitting_state_round_trip() -> None:
    state = _make_state()

    payload = state.to_payload()
    restored = FittingState.from_payload(payload)

    assert restored.noise == state.noise
    assert restored.version == state.version
    # Compare clusters by checking their cluster_id (numpy arrays make direct equality ambiguous)
    assert len(restored.clusters) == len(state.clusters)
    for rc, sc in zip(restored.clusters, state.clusters, strict=True):
        assert rc.cluster_id == sc.cluster_id
    assert list(restored.params.keys()) == list(state.params.keys())


def test_state_repository_persistence(tmp_path: Path) -> None:
    state = _make_state()

    state_path = StateRepository.default_path(tmp_path)
    StateRepository.save(state_path, state)

    loaded = StateRepository.load(state_path)

    assert loaded.noise == state.noise
    # Compare peaks by name since Peak contains numpy arrays
    assert len(loaded.peaks) == len(state.peaks)
    for lp, sp in zip(loaded.peaks, state.peaks, strict=True):
        assert lp.name == sp.name
        np.testing.assert_array_equal(lp.positions, sp.positions)
    assert list(loaded.params.keys()) == list(state.params.keys())
