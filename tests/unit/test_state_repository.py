"""Tests for the infrastructure state repository and fitting state model."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from peakfit.core.domain.cluster import Cluster
from peakfit.core.domain.peaks import Peak
from peakfit.core.domain.state import FittingState
from peakfit.core.fitting.parameters import Parameters
from peakfit.io.state import StateRepository


def _make_state() -> FittingState:
    params = Parameters()
    params.add("amp", value=1.0)
    params["amp"].stderr = 0.1
    clusters = cast(list[Cluster], ["cluster-a"])
    peaks = cast(list[Peak], ["peak-a"])
    return FittingState(
        clusters=clusters,
        params=params,
        noise=0.5,
        peaks=peaks,
        version="1.0",
    )


def test_fitting_state_round_trip() -> None:
    state = _make_state()

    payload = state.to_payload()
    restored = FittingState.from_payload(payload)

    assert restored.noise == state.noise
    assert restored.version == state.version
    assert restored.clusters == state.clusters
    assert list(restored.params.keys()) == list(state.params.keys())


def test_state_repository_persistence(tmp_path: Path) -> None:
    state = _make_state()

    state_path = StateRepository.default_path(tmp_path)
    StateRepository.save(state_path, state)

    loaded = StateRepository.load(state_path)

    assert loaded.noise == state.noise
    assert loaded.peaks == state.peaks
    assert list(loaded.params.keys()) == list(state.params.keys())
