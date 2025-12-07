"""Optimization strategy implementations used by the fitting pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from scipy.optimize import least_squares

from peakfit.core.fitting.computation import residuals
from peakfit.core.algorithms.global_optimization import (
    fit_basin_hopping,
    fit_differential_evolution,
)
from peakfit.core.results.statistics import compute_chi_squared, compute_reduced_chi_squared

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from peakfit.core.domain.cluster import Cluster
    from peakfit.core.fitting.parameters import Parameters
    from peakfit.core.shared.typing import FloatArray


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
        """Optimize the given cluster parameters and return result.

        Args:
            params: Parameters container for the cluster
            cluster: Cluster with peaks and data
            noise: Noise estimate for weighting residuals

        Returns
        -------
            OptimizationResult with final parameters and diagnostics
        """
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
        self._estimate_uncertainties(params, result, cluster)

        return OptimizationResult(
            x=result.x,
            cost=float(result.cost),
            success=bool(result.success),
            message=result.message,
            n_evaluations=result.nfev,
            params=params,
        )

    @staticmethod
    def _estimate_uncertainties(params: Parameters, result: Any, cluster: Cluster) -> None:
        """Populate stderr values using the jacobian if possible.

        Args:
            params: Parameters container
            result: scipy least_squares result object
            cluster: Cluster with peaks and data (needed for amplitude DOF)
        """
        vary_names = params.get_vary_names()
        if result.jac is None or len(result.fun) <= len(vary_names):
            return

        try:
            ndata = len(result.fun)
            nvarys = len(vary_names)
            # Degrees of freedom must include amplitude parameters
            n_amplitude_params = cluster.n_amplitude_params
            n_total_fitted = nvarys + n_amplitude_params
            chisqr = compute_chi_squared(result.fun)
            redchi = compute_reduced_chi_squared(chisqr, ndata, n_total_fitted)
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
        seed: int | None = None,
    ) -> None:
        self._n_iterations = n_iterations
        self._temperature = temperature
        self._step_size = step_size
        self._seed = seed

    def optimize(
        self,
        params: Parameters,
        cluster: Cluster,
        noise: float,
    ) -> OptimizationResult:
        """Perform basin-hopping global optimization on the cluster.

        Returns an OptimizationResult with additional metadata about the run.
        """
        start = perf_counter()
        initial_cost = float(np.sum(residuals(params, cluster, noise) ** 2))
        result = fit_basin_hopping(
            params,
            cluster,
            noise,
            n_iterations=self._n_iterations,
            temperature=self._temperature,
            step_size=self._step_size,
            seed=self._seed,
        )
        wall_time = perf_counter() - start

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
                "seed": self._seed,
                "temperature": self._temperature,
                "step_size": self._step_size,
                "initial_cost": initial_cost,
                "final_cost": float(result.cost),
                "wall_time_sec": wall_time,
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
        seed: int | None = None,
    ) -> None:
        self._max_iterations = max_iterations
        self._population_size = population_size
        self._mutation = mutation
        self._recombination = recombination
        self._polish = polish
        self._seed = seed

    def optimize(
        self,
        params: Parameters,
        cluster: Cluster,
        noise: float,
    ) -> OptimizationResult:
        """Run differential evolution to explore parameter space globally.

        Returns an OptimizationResult and includes metadata for the global run.
        """
        start = perf_counter()
        initial_cost = float(np.sum(residuals(params, cluster, noise) ** 2))
        result = fit_differential_evolution(
            params,
            cluster,
            noise,
            max_iterations=self._max_iterations,
            population_size=self._population_size,
            mutation=self._mutation,
            recombination=self._recombination,
            polish=self._polish,
            seed=self._seed,
        )
        wall_time = perf_counter() - start

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
                "seed": self._seed,
                "initial_cost": initial_cost,
                "final_cost": float(result.cost),
                "wall_time_sec": wall_time,
            },
        )


class VarProStrategy:
    """Variable Projection optimizer with analytical Jacobian.

    This strategy uses the VarProOptimizer which:
    1. Analytically solves for amplitudes (linear parameters)
    2. Provides an analytical Jacobian for faster convergence
    3. Caches intermediate results to avoid redundant computations

    This is typically 2-5x faster than LeastSquaresStrategy for NMR peak fitting.
    """

    def __init__(
        self,
        *,
        ftol: float = 1e-8,
        xtol: float = 1e-8,
        gtol: float = 1e-8,
        max_nfev: int = 1000,
        verbose: int = 0,
    ) -> None:
        self._ftol = ftol
        self._xtol = xtol
        self._gtol = gtol
        self._max_nfev = max_nfev
        self._verbose = verbose

    def optimize(
        self,
        params: Parameters,
        cluster: Cluster,
        noise: float,
    ) -> OptimizationResult:
        """Optimize using Variable Projection with analytical Jacobian.

        Args:
            params: Parameters container for the cluster
            cluster: Cluster with peaks and data
            noise: Noise estimate for weighting residuals

        Returns
        -------
            OptimizationResult with final parameters and diagnostics
        """
        from peakfit.core.algorithms.varpro import fit_cluster

        result = fit_cluster(
            params,
            cluster,
            noise,
            max_nfev=self._max_nfev,
            ftol=self._ftol,
            xtol=self._xtol,
            gtol=self._gtol,
            verbose=self._verbose,
        )

        return OptimizationResult(
            x=result.params.get_vary_values(),
            cost=float(result.cost),
            success=result.success,
            message=result.message,
            n_evaluations=result.nfev,
            params=result.params,
            metadata={
                "njev": result.njev,
                "optimality": result.optimality,
            },
        )


STRATEGIES: dict[str, type[OptimizationStrategy]] = {
    "leastsq": LeastSquaresStrategy,
    "varpro": VarProStrategy,
    "basin-hopping": BasinHoppingStrategy,
    "basin_hopping": BasinHoppingStrategy,
    "differential-evolution": DifferentialEvolutionStrategy,
    "differential_evolution": DifferentialEvolutionStrategy,
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
