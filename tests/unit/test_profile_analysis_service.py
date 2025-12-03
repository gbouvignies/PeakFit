"""Tests for the profile likelihood analysis service."""

from __future__ import annotations

import pytest  # type: ignore[import-not-found]

import numpy as np

from peakfit.core.domain.cluster import Cluster
from peakfit.core.domain.peaks import Peak
from peakfit.core.domain.state import FittingState
from peakfit.core.fitting.parameters import Parameters
from peakfit.services.analyze import (
    NoVaryingParametersError,
    ParameterMatchError,
    ProfileLikelihoodService,
)


def _make_peak(name: str) -> Peak:
    """Create a minimal valid Peak object for testing."""
    return Peak(name=name, positions=np.array([1.0, 2.0]), shapes=[])


def _make_cluster(cluster_id: int, peaks: list[Peak]) -> Cluster:
    """Create a minimal valid Cluster object for testing."""
    return Cluster(
        cluster_id=cluster_id,
        peaks=peaks,
        positions=[np.array([0, 1, 2])],
        data=np.zeros((3,)),
    )


def _make_state(param_layout: list[list[str]]) -> FittingState:
    """Create a FittingState with parameters matching the layout.

    param_layout is a list of clusters, each with parameter names.
    E.g., [["A_x0"], ["B_x0"]] creates 2 clusters with 1 param each.
    """
    params = Parameters()
    peaks = []
    clusters = []

    for cluster_id, cluster_params in enumerate(param_layout):
        cluster_peaks = []
        for param_name in cluster_params:
            params.add(param_name, value=1.0)
            params[param_name].stderr = 0.2
            # Extract peak name from param_name (e.g., "A_x0" -> "A")
            peak_name = param_name.split("_", 1)[0]
            peak = _make_peak(peak_name)
            cluster_peaks.append(peak)
            peaks.append(peak)
        clusters.append(_make_cluster(cluster_id, cluster_peaks))

    return FittingState(
        clusters=clusters,
        params=params,
        noise=0.1,
        peaks=peaks,
    )


def _fake_get_cluster_params(cluster, global_params, cache):
    """Return a Parameters object that contains all params from global_params that exist in the cluster's peaks.

    This mirrors _get_cluster_params but uses the global_params directly
    instead of relying on peak.shapes to generate parameter names.
    """
    cache_key = id(cluster)
    if cache_key not in cache:
        cluster_params = Parameters()
        # Get all peak names in this cluster
        peak_names = {peak.name for peak in cluster.peaks}
        # Include any global param that starts with one of the peak names
        for key in global_params:
            for peak_name in peak_names:
                if key.startswith(f"{peak_name}_"):
                    cluster_params.add(key, value=global_params[key].value)
                    cluster_params[key].stderr = global_params[key].stderr
                    cluster_params[key].vary = global_params[key].vary
                    break
        cache[cache_key] = cluster_params
    return cache[cache_key]


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
    monkeypatch.setattr(
        "peakfit.services.analyze.profile_service._get_cluster_params",
        _fake_get_cluster_params,
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
    monkeypatch.setattr(
        "peakfit.services.analyze.profile_service._get_cluster_params",
        _fake_get_cluster_params,
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
    monkeypatch.setattr(
        "peakfit.services.analyze.profile_service._get_cluster_params",
        _fake_get_cluster_params,
    )

    result = ProfileLikelihoodService.run(
        state,
        param_name=None,
        n_points=5,
        confidence_level=0.95,
    )

    assert "B_x0" in result.missing_parameters
    assert {res.parameter_name for res in result.results} == {"A_x0"}
