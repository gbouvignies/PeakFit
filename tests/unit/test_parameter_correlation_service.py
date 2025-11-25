"""Tests for the parameter correlation analysis service."""

from __future__ import annotations

from typing import cast

import pytest  # type: ignore[import-not-found]

from peakfit.core.domain.cluster import Cluster
from peakfit.core.domain.peaks import Peak
from peakfit.core.domain.state import FittingState
from peakfit.core.fitting.parameters import Parameters
from peakfit.services.analyze import NotEnoughVaryingParametersError, ParameterCorrelationService


def _make_state(param_defs: list[tuple[str, bool]]) -> FittingState:
    params = Parameters()
    for name, vary in param_defs:
        params.add(name, value=1.0, min=0.0, max=2.0, vary=vary)
        params[name].stderr = 0.1
    clusters = cast(list[Cluster], [])
    peaks = cast(list[Peak], [])
    return FittingState(clusters=clusters, params=params, noise=0.2, peaks=peaks)


def test_analyze_returns_parameter_entries() -> None:
    state = _make_state([("A_x0", True), ("B_x0", True), ("C_x0", False)])
    state.params["A_x0"].value = 0.0  # at lower bound

    result = ParameterCorrelationService.analyze(state)

    assert [entry.name for entry in result.parameters] == ["A_x0", "B_x0"]
    boundary_names = [entry.name for entry in result.boundary_parameters]
    assert boundary_names == ["A_x0"]
    first_entry = result.parameters[0]
    assert first_entry.min_bound == pytest.approx(0.0)
    assert first_entry.max_bound == pytest.approx(2.0)
    assert first_entry.stderr == pytest.approx(0.1)


def test_requires_two_varying_parameters() -> None:
    state = _make_state([("A_x0", True), ("B_x0", False)])

    with pytest.raises(NotEnoughVaryingParametersError) as excinfo:
        ParameterCorrelationService.analyze(state)

    match_error = excinfo.value
    assert isinstance(match_error, NotEnoughVaryingParametersError)
    assert match_error.vary_names == ["A_x0"]
