"""Tests for the profile likelihood analysis service."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np  # type: ignore[import-not-found]
import pytest  # type: ignore[import-not-found]

from peakfit.core.domain.state import FittingState
from peakfit.core.fitting.parameters import Parameters
from peakfit.services.analyze import (
    NoVaryingParametersError,
    ParameterMatchError,
    ProfileLikelihoodService,
)


class FakePeak:
    """Minimal peak object returning a deterministic parameter set."""

    def __init__(self, param_name: str):
        self.name = param_name.split("_", 1)[0]
        self._param_name = param_name

    def create_params(self) -> Parameters:
        params = Parameters()
        params.add(self._param_name, value=1.0)
        params[self._param_name].stderr = 0.2
        return params


def _make_state(param_layout: list[list[str]]) -> FittingState:
    params = Parameters()
    peaks = []
    clusters = []

    for cluster_params in param_layout:
        cluster_peaks = []
        for param_name in cluster_params:
            params.add(param_name, value=1.0)
            params[param_name].stderr = 0.2
            peak = FakePeak(param_name)
            cluster_peaks.append(peak)
            peaks.append(peak)
        clusters.append(SimpleNamespace(peaks=cluster_peaks))  # type: ignore[arg-type]

    return FittingState(
        clusters=clusters,  # type: ignore[arg-type]
        params=params,
        noise=0.1,
        peaks=peaks,  # type: ignore[arg-type]
    )


def test_profiles_all_varying_parameters(monkeypatch):
    state = _make_state([["A_x0"], ["B_x0"]])
    call_count = 0

    def fake_profile(*_, **__):
        nonlocal call_count
        call_count += 1
        return np.array([0.0, 1.0]), np.array([0.1, 0.2]), (0.2, 0.8)

    monkeypatch.setattr(
        "peakfit.services.analyze.profile_service.compute_profile_likelihood",
        fake_profile,
    )

    result = ProfileLikelihoodService.run(
        state,
        param_name=None,
        n_points=10,
        confidence_level=0.95,
    )

    assert result.target_parameters == state.params.get_vary_names()
    assert len(result.results) == 2
    assert call_count == 2


def test_filters_parameters_by_pattern(monkeypatch):
    state = _make_state([["A_x0", "A_fwhm"], ["B_x0"]])

    monkeypatch.setattr(
        "peakfit.services.analyze.profile_service.compute_profile_likelihood",
        lambda *_, **__: (np.array([0.0]), np.array([0.0]), (0.0, 1.0)),
    )

    result = ProfileLikelihoodService.run(
        state,
        param_name="A",
        n_points=5,
        confidence_level=0.95,
    )

    assert result.target_parameters == ["A_x0", "A_fwhm"]
    assert {res.parameter_name for res in result.results} == {"A_x0", "A_fwhm"}


def test_raises_for_missing_pattern():
    state = _make_state([["A_x0"]])

    with pytest.raises(ParameterMatchError) as excinfo:
        ProfileLikelihoodService.run(
            state,
            param_name="missing",
            n_points=5,
            confidence_level=0.95,
        )

    match_error = excinfo.value
    assert isinstance(match_error, ParameterMatchError)
    assert match_error.pattern == "missing"
    assert match_error.available == state.params.get_vary_names()


def test_raises_when_no_varying_parameters():
    state = _make_state([["A_x0"]])
    for name in state.params:
        state.params[name].vary = False

    with pytest.raises(NoVaryingParametersError):
        ProfileLikelihoodService.run(
            state,
            param_name=None,
            n_points=5,
            confidence_level=0.95,
        )


def test_marks_missing_parameters(monkeypatch):
    state = _make_state([["A_x0"]])
    state.params.add("B_x0", value=1.0)
    state.params["B_x0"].stderr = 0.1

    monkeypatch.setattr(
        "peakfit.services.analyze.profile_service.compute_profile_likelihood",
        lambda *_, **__: (np.array([0.0]), np.array([0.0]), (0.0, 1.0)),
    )

    result = ProfileLikelihoodService.run(
        state,
        param_name=None,
        n_points=5,
        confidence_level=0.95,
    )

    assert "B_x0" in result.missing_parameters
    assert {res.parameter_name for res in result.results} == {"A_x0"}
