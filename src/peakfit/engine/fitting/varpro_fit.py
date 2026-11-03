"""Fit one cluster with VarPro and record the optimizer result."""

from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.algorithms.varpro import fit_cluster
from peakfit.engine.results import FitResult

if TYPE_CHECKING:
    from peakfit.engine.domain.cluster import Cluster
    from peakfit.engine.domain.config import VarProConfig
    from peakfit.engine.domain.params_scalar import Parameters


def fit_varpro(
    params: Parameters,
    cluster: Cluster,
    noise: float,
    config: VarProConfig,
) -> FitResult:
    """Optimize one cluster's nonlinear parameters by Variable Projection."""
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
        cluster_id=cluster.cluster_id,
        params=result.params,
        residual=residual,
        cost=result.cost,
        optimizer_kind="varpro",
        nfev=result.nfev,
        njev=result.njev,
        success=result.success,
        message=result.message,
        optimality=result.optimality,
        n_amplitude_params=result.n_amplitude_params,
        metadata={
            "peak_names": [p.name for p in cluster.peaks],
            "fit_time": perf_counter() - start_time,
        },
    )


__all__ = ["fit_varpro"]
