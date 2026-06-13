"""Fit one cluster with the selected optimizer."""

from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.algorithms.basin_hopping import fit_basin_hopping
from peakfit.engine.algorithms.common import residuals
from peakfit.engine.algorithms.varpro import fit_cluster
from peakfit.engine.domain.config import BasinHoppingConfig, OptimizerConfig, VarProConfig
from peakfit.engine.results import FitResult

if TYPE_CHECKING:
    from peakfit.engine.domain.cluster import Cluster
    from peakfit.engine.domain.params_scalar import Parameters


def fit_with_optimizer(
    name: str,
    params: Parameters,
    cluster: Cluster,
    noise: float,
    config: OptimizerConfig,
) -> FitResult:
    """Fit one cluster with the selected optimizer."""
    if name == "varpro":
        if not isinstance(config, VarProConfig):
            raise TypeError("varpro requires VarProConfig")
        return _fit_varpro(params, cluster, noise, config)

    if name == "basin_hopping":
        if not isinstance(config, BasinHoppingConfig):
            raise TypeError("basin_hopping requires BasinHoppingConfig")
        return _fit_basin_hopping(params, cluster, noise, config)

    raise KeyError(f"Unknown optimizer: {name}")


def _fit_varpro(
    params: Parameters,
    cluster: Cluster,
    noise: float,
    config: VarProConfig,
) -> FitResult:
    """Optimize using Variable Projection."""
    start_time = perf_counter()
    result = fit_cluster(
        params,
        cluster,
        noise,
        max_nfev=config.max_nfev,
        ftol=config.ftol,
        xtol=config.xtol,
        gtol=config.gtol,
        verbose=config.verbose,
    )

    residual = np.array([result.residual]) if np.shape(result.residual) == () else result.residual

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


def _fit_basin_hopping(
    params: Parameters,
    cluster: Cluster,
    noise: float,
    config: BasinHoppingConfig,
) -> FitResult:
    """Optimize using basin-hopping."""
    start_time = perf_counter()
    initial_cost = float(np.sum(residuals(params, cluster, noise) ** 2))

    result = fit_basin_hopping(
        params,
        cluster,
        noise,
        n_iterations=config.n_iterations,
        temperature=config.temperature,
        step_size=config.step_size,
        seed=config.seed,
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
            "seed": config.seed,
        },
    )


__all__ = ["fit_with_optimizer"]
