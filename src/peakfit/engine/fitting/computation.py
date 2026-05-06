"""Core computational functions for peak fitting.

This module now acts as a facade, exporting functions from `common` and `linear_algebra`.
It also contains the `fit_cluster_worker` for parallel processing.
"""

from typing import TYPE_CHECKING

from peakfit.engine.algorithms.common import (
    calculate_shape_heights,
    residuals,
)
from peakfit.engine.fitting.strategies import get_strategy

if TYPE_CHECKING:
    from peakfit.engine.domain.cluster import Cluster
    from peakfit.engine.domain.config import StrategyConfig
    from peakfit.engine.domain.params_scalar import Parameters
    from peakfit.engine.results import FitResult

__all__ = ["calculate_shape_heights", "fit_cluster_worker", "residuals"]


def fit_cluster_worker(
    cluster: Cluster,
    params: Parameters,
    noise: float,
    strategy_name: str = "varpro",
    config: StrategyConfig | None = None,
) -> FitResult:
    """Stateless worker function for parallel fitting.

    Compatible with multiprocessing (pickleable arguments).

    Args:
        cluster: Cluster to fit
        params: Parameters object (initial guess)
        noise: Noise level
        strategy_name: Name of strategy to use ("varpro", etc)
        config: Configuration object for the strategy

    Returns:
    -------
        FitResult object
    """
    if config is None:
        raise ValueError("Strategy configuration must be provided")

    strategy = get_strategy(strategy_name, config)

    return strategy.fit(params, cluster, noise)
