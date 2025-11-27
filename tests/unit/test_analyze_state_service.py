"""Tests for the analyze-layer fitting state service."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest  # type: ignore[import-not-found]

from peakfit.core.domain.cluster import Cluster
from peakfit.core.domain.peaks import Peak
from peakfit.core.domain.state import FittingState
from peakfit.core.fitting.parameters import Parameters
from peakfit.io.state import StateRepository
from peakfit.services.analyze import FittingStateService, StateFileMissingError, StateLoadError


def _make_state() -> FittingState:
    params = Parameters()
    params.add("amp", value=1.0)
    clusters = cast(list[Cluster], ["cluster-a"])
    peaks = cast(list[Peak], ["peak-a"])
    return FittingState(clusters=clusters, params=params, noise=0.5, peaks=peaks, version="1.0")


def test_service_loads_state(tmp_path: Path) -> None:
    results_dir = tmp_path / "Fits"
    results_dir.mkdir()

    state_path = StateRepository.default_path(results_dir)
    StateRepository.save(state_path, _make_state())

    loaded = FittingStateService.load(results_dir)

    assert loaded.path == state_path
    assert loaded.state.noise == pytest.approx(0.5)
    assert loaded.state.peaks == ["peak-a"]


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
