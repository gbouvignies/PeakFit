"""Optimization strategy implementations used by the fitting pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from scipy.optimize import least_squares

from peakfit.core.fitting.advanced import fit_basin_hopping, fit_differential_evolution
from peakfit.core.fitting.computation import residuals
from peakfit.core.fitting.parameters import Parameters
from peakfit.core.shared.typing import FloatArray

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from peakfit.core.domain.cluster import Cluster


class OptimizationStrategy(Protocol):
    """Protocol implemented by all optimization strategies."""

    def optimize(
        self,
        params: Parameters,
        cluster: Cluster,
        noise: float,
    ) -> OptimizationResult:
        """Optimize the supplied parameters for the provided cluster."""
        ...


@dataclass(slots=True)
class OptimizationResult:
    """Normalized result object for strategy executions."""

    x: FloatArray
    cost: float
    success: bool
    message: str
    n_evaluations: int | str
    params: Parameters
    metadata: dict[str, Any] | None = None


class LeastSquaresStrategy:
    """Local optimizer using scipy.optimize.least_squares."""

    def __init__(
        self,
        *,
        ftol: float = 1e-8,
        xtol: float = 1e-8,
        max_nfev: int | None = None,
        verbose: int = 0,
    ) -> None:
        self._ftol = ftol
        self._xtol = xtol
        self._max_nfev = max_nfev
        self._verbose = verbose

    def optimize(
        self,
        params: Parameters,
        cluster: Cluster,
        noise: float,
    ) -> OptimizationResult:
        vary_names = params.get_vary_names()
        x0 = params.get_vary_values()
        lower = np.array([params[name].min for name in vary_names], dtype=float)
        upper = np.array([params[name].max for name in vary_names], dtype=float)

        def objective(values: FloatArray) -> FloatArray:
            params.set_vary_values(values)
            return residuals(params, cluster, noise)

        result = least_squares(
            objective,
            x0,
            bounds=(lower, upper),
            ftol=self._ftol,
            xtol=self._xtol,
            max_nfev=self._max_nfev,
            verbose=self._verbose,
        )

        params.set_vary_values(result.x)
        self._estimate_uncertainties(params, result)

        return OptimizationResult(
            x=result.x,
            cost=float(result.cost),
            success=bool(result.success),
            message=result.message,
            n_evaluations=result.nfev,
            params=params,
        )

    @staticmethod
    def _estimate_uncertainties(params: Parameters, result: Any) -> None:
        """Populate stderr values using the jacobian if possible."""

        vary_names = params.get_vary_names()
        if result.jac is None or len(result.fun) <= len(vary_names):
            return

        try:
            ndata = len(result.fun)
            nvarys = len(vary_names)
            redchi = np.sum(result.fun**2) / max(1, ndata - nvarys)
            jtj = result.jac.T @ result.jac
            cov = np.linalg.inv(jtj) * redchi
            stderr = np.sqrt(np.diag(cov))
            for idx, name in enumerate(vary_names):
                params[name].stderr = float(stderr[idx])
        except np.linalg.LinAlgError:  # pragma: no cover - numerical edge case
            return


class BasinHoppingStrategy:
    """Global optimizer leveraging the advanced basin-hopping helper."""

    def __init__(
        self,
        *,
        n_iterations: int = 50,
        temperature: float = 1.0,
        step_size: float = 0.5,
    ) -> None:
        self._n_iterations = n_iterations
        self._temperature = temperature
        self._step_size = step_size

    def optimize(
        self,
        params: Parameters,
        cluster: Cluster,
        noise: float,
    ) -> OptimizationResult:
        result = fit_basin_hopping(
            params,
            cluster,
            noise,
            n_iterations=self._n_iterations,
            temperature=self._temperature,
            step_size=self._step_size,
        )

        return OptimizationResult(
            x=result.params.get_vary_values(),
            cost=float(result.cost),
            success=result.success,
            message=result.message,
            n_evaluations=result.nfev,
            params=result.params,
            metadata={
                "global_iterations": result.global_iterations,
                "local_minimizations": result.local_minimizations,
                "global_minimum_found": result.global_minimum_found,
            },
        )


class DifferentialEvolutionStrategy:
    """Population-based optimizer for exploring rugged landscapes."""

    def __init__(
        self,
        *,
        max_iterations: int = 500,
        population_size: int = 15,
        mutation: tuple[float, float] = (0.5, 1.0),
        recombination: float = 0.7,
        polish: bool = True,
    ) -> None:
        self._max_iterations = max_iterations
        self._population_size = population_size
        self._mutation = mutation
        self._recombination = recombination
        self._polish = polish

    def optimize(
        self,
        params: Parameters,
        cluster: Cluster,
        noise: float,
    ) -> OptimizationResult:
        result = fit_differential_evolution(
            params,
            cluster,
            noise,
            max_iterations=self._max_iterations,
            population_size=self._population_size,
            mutation=self._mutation,
            recombination=self._recombination,
            polish=self._polish,
        )

        return OptimizationResult(
            x=result.params.get_vary_values(),
            cost=float(result.cost),
            success=result.success,
            message=result.message,
            n_evaluations=result.nfev,
            params=result.params,
            metadata={
                "global_iterations": result.global_iterations,
                "polished": self._polish,
            },
        )


STRATEGIES: dict[str, type[OptimizationStrategy]] = {
    "leastsq": LeastSquaresStrategy,
    "basin-hopping": BasinHoppingStrategy,
    "differential-evolution": DifferentialEvolutionStrategy,
}


def get_strategy(name: str, **kwargs: Any) -> OptimizationStrategy:
    """Return an instantiated strategy by name."""

    try:
        strategy_cls = STRATEGIES[name]
    except KeyError as exc:  # pragma: no cover - defensive guard
        msg = f"Unknown optimization strategy: {name}"
        raise KeyError(msg) from exc
    return strategy_cls(**kwargs)


def register_strategy(name: str, strategy_cls: type[OptimizationStrategy]) -> None:
    """Register a custom strategy implementation."""

    STRATEGIES[name] = strategy_cls
