"""Tests for global optimization strategies seeding and metadata."""

from __future__ import annotations

import pytest

import numpy as np

from peakfit.core.fitting.global_optimization import GlobalFitResult
from peakfit.core.fitting.protocol import FitProtocol
from peakfit.core.fitting.strategies import (
    BasinHoppingStrategy,
    DifferentialEvolutionStrategy,
)
from peakfit.services.fit.fitting import FitRunner


class _FakeParams:
    def __init__(self, values: list[float]):
        self._values = np.array(values, dtype=float)

    def get_vary_values(self) -> np.ndarray:
        return self._values

    def set_vary_values(self, values: np.ndarray) -> None:
        self._values = np.array(values, dtype=float)

    def get_vary_names(self) -> list[str]:
        return [f"p{i}" for i in range(len(self._values))]


class _DummyCluster:
    def __init__(self) -> None:
        self.n_amplitude_params = 0
        self.peaks: list = []
        self.positions: np.ndarray | None = None
        self.corrected_data: np.ndarray | None = None


@pytest.fixture
def fake_params() -> _FakeParams:
    return _FakeParams([0.0, 0.0])


@pytest.fixture
def dummy_cluster() -> _DummyCluster:
    return _DummyCluster()


def test_basin_hopping_seed_and_metadata(monkeypatch, fake_params, dummy_cluster):
    calls: dict[str, int | None] = {}

    def fake_residuals(params, cluster, noise):
        return np.array([1.0, 2.0], dtype=float)

    def fake_fit(params, cluster, noise, n_iterations, temperature, step_size, seed):
        calls["seed"] = seed
        return GlobalFitResult(
            params=params,
            residual=np.array([0.5], dtype=float),
            cost=0.5,
            nfev=4,
            success=True,
            message="ok",
            global_iterations=n_iterations,
            local_minimizations=1,
            global_minimum_found=True,
            basin_hopping_temperature=temperature,
        )

    monkeypatch.setattr("peakfit.core.fitting.strategies.residuals", fake_residuals)
    monkeypatch.setattr("peakfit.core.fitting.strategies.fit_basin_hopping", fake_fit)

    strategy = BasinHoppingStrategy(n_iterations=3, temperature=2.0, step_size=0.1, seed=123)
    result = strategy.optimize(fake_params, dummy_cluster, noise=1.0)

    assert calls["seed"] == 123
    assert result.metadata is not None
    assert result.metadata["seed"] == 123
    assert result.metadata["global_iterations"] == 3
    assert pytest.approx(result.metadata["initial_cost"], rel=1e-9) == 5.0
    assert pytest.approx(result.metadata["final_cost"], rel=1e-9) == 0.5


def test_differential_evolution_seed_and_metadata(monkeypatch, fake_params, dummy_cluster):
    calls: dict[str, int | None] = {}

    def fake_residuals(params, cluster, noise):
        return np.array([3.0], dtype=float)

    def fake_fit(
        params,
        cluster,
        noise,
        max_iterations,
        population_size,
        mutation,
        recombination,
        polish,
        seed,
    ):
        calls["seed"] = seed
        return GlobalFitResult(
            params=params,
            residual=np.array([0.25], dtype=float),
            cost=0.25,
            nfev=7,
            success=True,
            message="ok",
            global_iterations=max_iterations,
            local_minimizations=1 if polish else 0,
            global_minimum_found=True,
        )

    monkeypatch.setattr("peakfit.core.fitting.strategies.residuals", fake_residuals)
    monkeypatch.setattr("peakfit.core.fitting.strategies.fit_differential_evolution", fake_fit)

    strategy = DifferentialEvolutionStrategy(max_iterations=4, population_size=5, seed=99)
    result = strategy.optimize(fake_params, dummy_cluster, noise=1.0)

    assert calls["seed"] == 99
    assert result.metadata is not None
    assert result.metadata["seed"] == 99
    assert result.metadata["global_iterations"] == 4
    assert pytest.approx(result.metadata["initial_cost"], rel=1e-9) == 9.0
    assert pytest.approx(result.metadata["final_cost"], rel=1e-9) == 0.25


def test_fit_runner_caps_iterations_and_seeds_global_strategies():
    protocol = FitProtocol.default(refine_iterations=1)

    runner_bh = FitRunner(
        clusters=[],
        protocol=protocol,
        noise=1.0,
        optimizer="basin-hopping",
        optimizer_seed=7,
        max_nfev=10,
    )
    assert isinstance(runner_bh.strategy, BasinHoppingStrategy)
    assert runner_bh.strategy._seed == 7
    assert runner_bh.strategy._n_iterations == 10

    runner_de = FitRunner(
        clusters=[],
        protocol=protocol,
        noise=1.0,
        optimizer="differential-evolution",
        optimizer_seed=11,
        max_nfev=8,
    )
    assert isinstance(runner_de.strategy, DifferentialEvolutionStrategy)
    assert runner_de.strategy._seed == 11
    assert runner_de.strategy._max_iterations == 8
