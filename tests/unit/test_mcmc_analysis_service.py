"""Tests for the MCMC analysis service."""

from __future__ import annotations

from types import SimpleNamespace

import pytest  # type: ignore[import-not-found]

import numpy as np  # type: ignore[import-not-found]

from peakfit.core.domain.state import FittingState
from peakfit.core.fitting.advanced import UncertaintyResult
from peakfit.core.fitting.parameters import Parameters
from peakfit.services.analyze import MCMCAnalysisService, PeaksNotFoundError


def _make_uncertainty(std_err: float = 0.2) -> UncertaintyResult:
    return UncertaintyResult(
        parameter_names=["amp"],
        values=np.array([1.0]),
        std_errors=np.array([std_err]),
        confidence_intervals_68=np.array([[0.8, 1.2]]),
        confidence_intervals_95=np.array([[0.5, 1.5]]),
        correlation_matrix=np.array([[1.0]]),
    )


def _make_state() -> FittingState:
    params = Parameters()
    params.add("amp", value=1.0)
    cluster = SimpleNamespace(peaks=[SimpleNamespace(name="A")])
    peak = SimpleNamespace(name="A")
    return FittingState(
        clusters=[cluster],  # type: ignore[arg-type]
        params=params,
        noise=0.5,
        peaks=[peak],  # type: ignore[arg-type]
    )


def test_service_updates_parameter_stderr(monkeypatch):
    state = _make_state()

    def fake_create_params(peaks):
        params = Parameters()
        params.add("amp", value=1.0)
        return params

    monkeypatch.setattr(
        "peakfit.services.analyze.mcmc_service.create_params",
        fake_create_params,
    )
    monkeypatch.setattr(
        "peakfit.services.analyze.mcmc_service.estimate_uncertainties_mcmc",
        lambda *args, **kwargs: _make_uncertainty(0.3),
    )

    result = MCMCAnalysisService.run(
        state,
        peaks=None,
        n_walkers=16,
        n_steps=100,
        burn_in=50,
        auto_burnin=False,
    )

    assert len(result.cluster_results) == 1
    assert result.cluster_results[0].result.std_errors[0] == pytest.approx(0.3)
    assert state.params["amp"].stderr == pytest.approx(0.3)


def test_service_filters_peaks(monkeypatch):
    state = _make_state()
    extra_cluster = SimpleNamespace(peaks=[SimpleNamespace(name="B")])
    state.clusters.append(extra_cluster)  # type: ignore[arg-type]

    monkeypatch.setattr(
        "peakfit.services.analyze.mcmc_service.create_params",
        lambda peaks: Parameters(),
    )
    monkeypatch.setattr(
        "peakfit.services.analyze.mcmc_service.estimate_uncertainties_mcmc",
        lambda *_, **__: _make_uncertainty(),
    )

    result = MCMCAnalysisService.run(
        state,
        peaks=["B"],
        n_walkers=16,
        n_steps=100,
        burn_in=None,
        auto_burnin=True,
    )

    assert len(result.clusters) == 1
    assert result.cluster_results[0].cluster is extra_cluster


def test_service_raises_for_missing_peaks(monkeypatch):
    state = _make_state()

    monkeypatch.setattr(
        "peakfit.services.analyze.mcmc_service.create_params",
        lambda peaks: Parameters(),
    )
    monkeypatch.setattr(
        "peakfit.services.analyze.mcmc_service.estimate_uncertainties_mcmc",
        lambda *_, **__: _make_uncertainty(),
    )

    with pytest.raises(PeaksNotFoundError):
        MCMCAnalysisService.run(
            state,
            peaks=["Z"],
            n_walkers=8,
            n_steps=50,
            burn_in=None,
            auto_burnin=True,
        )
