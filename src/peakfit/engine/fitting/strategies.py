"""Optimization strategy implementations used by the fitting pipeline."""

from dataclasses import asdict
from time import perf_counter
from typing import TYPE_CHECKING, Protocol

import numpy as np

from peakfit.engine.algorithms.common import residuals
from peakfit.engine.algorithms.global_optimization import (
    fit_basin_hopping,
    fit_differential_evolution,
)
from peakfit.engine.algorithms.mcmc import estimate_uncertainties_mcmc
from peakfit.engine.algorithms.varpro import fit_cluster
from peakfit.engine.results import FitResult
from peakfit.shared.constants import (
    BASIN_HOPPING_NITER,
    BASIN_HOPPING_STEPSIZE,
    BASIN_HOPPING_TEMPERATURE,
    DIFF_EVOLUTION_INIT,
    DIFF_EVOLUTION_MAXITER,
    DIFF_EVOLUTION_MUTATION,
    DIFF_EVOLUTION_RECOMBINATION,
    MCMC_N_STEPS,
    MCMC_N_WALKERS,
)

if TYPE_CHECKING:
    from peakfit.engine.domain.cluster import Cluster
    from peakfit.engine.domain.config import StrategyConfig
    from peakfit.engine.domain.params_scalar import Parameters


class FitStrategy(Protocol):
    """Protocol implemented by all optimization strategies."""

    def fit(
        self,
        params: Parameters,
        cluster: Cluster,
        noise: float,
    ) -> FitResult:
        """Optimize the supplied parameters for the provided cluster."""
        ...


class VarProStrategy:
    """Variable Projection optimizer with analytical Jacobian."""

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

    def fit(
        self,
        params: Parameters,
        cluster: Cluster,
        noise: float,
    ) -> FitResult:
        """Optimize using Variable Projection."""
        start_time = perf_counter()
        # `fit_cluster` returns a FitResult-like object; normalize it to our
        # `peakfit.engine.results.fit_results.FitResult` for downstream consumers.

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

        residual = (
            np.array([result.residual]) if np.shape(result.residual) == () else result.residual
        )

        return FitResult(
            params=result.params,
            residual=residual,
            cost=result.cost,
            nfev=result.nfev,
            njev=result.njev,
            success=result.success,
            message=result.message,
            optimality=result.optimality,
            n_amplitude_params=result.n_amplitude_params,
            metadata={
                "cluster_id": cluster.cluster_id,
                "peak_names": [p.name for p in cluster.peaks],
                "fit_time": perf_counter() - start_time,
            },
        )


class BasinHoppingStrategy:
    """Global optimizer leveraging basin-hopping."""

    def __init__(
        self,
        *,
        n_iterations: int = BASIN_HOPPING_NITER,
        temperature: float = BASIN_HOPPING_TEMPERATURE,
        step_size: float = BASIN_HOPPING_STEPSIZE,
        seed: int | None = None,
    ) -> None:
        self._n_iterations = n_iterations
        self._temperature = temperature
        self._step_size = step_size
        self._seed = seed

    def fit(
        self,
        params: Parameters,
        cluster: Cluster,
        noise: float,
    ) -> FitResult:
        """Optimize using Basin Hopping."""
        start_time = perf_counter()
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
        wall_time = perf_counter() - start_time

        return FitResult(
            params=result.params,
            residual=result.residual,
            cost=float(result.cost),
            nfev=result.nfev,
            success=result.success,
            message=result.message,
            n_amplitude_params=result.n_amplitude_params,
            metadata={
                "cluster_id": cluster.cluster_id,
                "peak_names": [p.name for p in cluster.peaks],
                "global_iterations": result.global_iterations,
                "local_minimizations": result.local_minimizations,
                "global_minimum_found": result.global_minimum_found,
                "initial_cost": initial_cost,
                "fit_time": wall_time,
                "seed": self._seed,
            },
        )


class DifferentialEvolutionStrategy:
    """Global optimizer using Differential Evolution."""

    def __init__(
        self,
        *,
        max_iterations: int = DIFF_EVOLUTION_MAXITER,
        mutation: tuple[float, float] = DIFF_EVOLUTION_MUTATION,
        recombination: float = DIFF_EVOLUTION_RECOMBINATION,
        init: str = DIFF_EVOLUTION_INIT,
        polish: bool = True,
        seed: int | None = None,
    ) -> None:
        self._max_iterations = max_iterations
        self._mutation = mutation
        self._recombination = recombination
        self._init = init
        self._polish = polish
        self._seed = seed

    def fit(
        self,
        params: Parameters,
        cluster: Cluster,
        noise: float,
    ) -> FitResult:
        """Optimize using Differential Evolution."""
        start_time = perf_counter()
        initial_cost = float(np.sum(residuals(params, cluster, noise) ** 2))

        result = fit_differential_evolution(
            params,
            cluster,
            noise,
            max_iterations=self._max_iterations,
            mutation=self._mutation,
            recombination=self._recombination,
            init=self._init,
            polish=self._polish,
            seed=self._seed,
        )
        wall_time = perf_counter() - start_time

        return FitResult(
            params=result.params,
            residual=result.residual,
            cost=float(result.cost),
            nfev=result.nfev,
            success=result.success,
            message=result.message,
            n_amplitude_params=result.n_amplitude_params,
            metadata={
                "cluster_id": cluster.cluster_id,
                "peak_names": [p.name for p in cluster.peaks],
                "global_iterations": result.global_iterations,
                "polished": self._polish,
                "initial_cost": initial_cost,
                "fit_time": wall_time,
                "seed": self._seed,
            },
        )


class MCMCStrategy:
    """Bayesian uncertainty estimation using MCMC."""

    def __init__(
        self,
        *,
        n_walkers: int = MCMC_N_WALKERS,
        n_steps: int = MCMC_N_STEPS,
        burn_in: int | None = None,
        workers: int = 1,
    ) -> None:
        self._n_walkers = n_walkers
        self._n_steps = n_steps
        self._burn_in = burn_in
        self._workers = workers

    def fit(
        self,
        params: Parameters,
        cluster: Cluster,
        noise: float,
    ) -> FitResult:
        """Run MCMC sampling."""
        start_time = perf_counter()
        uncertainty = estimate_uncertainties_mcmc(
            params,
            cluster,
            noise,
            n_walkers=self._n_walkers,
            n_steps=self._n_steps,
            burn_in=self._burn_in,
            workers=self._workers,
        )

        # MCMC doesn't inherently minimize residuals to a point estimate in the same way,
        # but we return the Best Fit (median) values.
        # We need to calculate residual/cost for these median values.

        final_params = params.copy()
        # Update values to MCMC medians
        for i, name in enumerate(uncertainty.parameter_names):
            if name in final_params:
                final_params[name].value = uncertainty.values[i]
                final_params[name].stderr = uncertainty.std_errors[i]

        # Compute final residual
        res = residuals(final_params, cluster, noise)
        cost = 0.5 * np.sum(res**2)  # Cost is typically chi2/2 or similar

        return FitResult(
            params=final_params,
            residual=res,
            cost=float(cost),
            nfev=self._n_walkers * self._n_steps,
            success=True,  # MCMC finished
            message="MCMC Complete",
            n_amplitude_params=cluster.n_amplitude_params,
            uncertainty=uncertainty,
            metadata={
                "cluster_id": cluster.cluster_id,
                "peak_names": [p.name for p in cluster.peaks],
                "burn_in": uncertainty.burn_in_info,
                "n_samples": uncertainty.mcmc_samples.shape[0]
                if uncertainty.mcmc_samples is not None
                else 0,
                "fit_time": perf_counter() - start_time,
            },
        )


# Registry
STRATEGIES: dict[str, type[FitStrategy]] = {
    "varpro": VarProStrategy,
    "basin_hopping": BasinHoppingStrategy,
    "differential_evolution": DifferentialEvolutionStrategy,
    "mcmc": MCMCStrategy,
}


def get_strategy(name: str, config: StrategyConfig | None = None) -> FitStrategy:
    """Return an instantiated strategy by name."""
    try:
        strategy_cls = STRATEGIES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown optimization strategy: {name}") from exc

    if config:
        return strategy_cls(**asdict(config))
    return strategy_cls()


def register_strategy(name: str, strategy_cls: type[FitStrategy]) -> None:
    """Register a new optimization strategy."""
    STRATEGIES[name] = strategy_cls
