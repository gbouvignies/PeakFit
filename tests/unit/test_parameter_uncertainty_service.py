"""Tests for the parameter uncertainty service."""

from __future__ import annotations

from typing import Any, cast

import pytest  # type: ignore[import-not-found]

from peakfit.core.domain.cluster import Cluster
from peakfit.core.domain.peaks import Peak
from peakfit.core.domain.state import FittingState
from peakfit.core.fitting.parameters import Parameters
from peakfit.services.analyze import NoVaryingParametersFoundError, ParameterUncertaintyService


def _make_state(param_defs: list[dict[str, Any]]) -> FittingState:
    params = Parameters()
    for definition in param_defs:
        name = definition["name"]
        params.add(
            name,
            value=definition.get("value", 1.0),
            min=definition.get("min", 0.0),
            max=definition.get("max", 2.0),
            vary=definition.get("vary", True),
        )
        params[name].stderr = definition.get("stderr", 0.1)
    clusters = cast(list[Cluster], [])
    peaks = cast(list[Peak], [])
    return FittingState(clusters=clusters, params=params, noise=0.1, peaks=peaks)


def test_analyze_builds_entries_with_metadata() -> None:
    state = _make_state(
        [
            {"name": "A_x0", "value": 1.0, "stderr": 0.2},
            {"name": "B_x0", "value": 0.0, "stderr": 0.0, "min": 0.0, "max": 1.0},
            {"name": "C_x0", "vary": False},
        ]
    )

    result = ParameterUncertaintyService.analyze(state)

    assert [entry.name for entry in result.parameters] == ["A_x0", "B_x0"]
    assert {entry.name for entry in result.boundary_parameters} == {"B_x0"}
    assert {entry.name for entry in result.large_uncertainty_parameters} == {"A_x0"}

    first_entry = result.parameters[0]
    assert first_entry.rel_error_pct == pytest.approx(20.0)
    assert first_entry.min_bound == pytest.approx(0.0)
    assert first_entry.max_bound == pytest.approx(2.0)


def test_raises_when_no_varying_parameters() -> None:
    state = _make_state(
        [
            {"name": "A_x0", "vary": False},
        ]
    )

    with pytest.raises(NoVaryingParametersFoundError):
        ParameterUncertaintyService.analyze(state)
