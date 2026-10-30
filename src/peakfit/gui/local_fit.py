"""Qt-independent execution of an already reviewed, isolated local fit."""

import sys
import warnings
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
from threadpoolctl import threadpool_limits

from peakfit.engine.algorithms.evaluation import FitEvaluation, classify_optimizer_result
from peakfit.engine.domain.config import PeakFitConfig, VarProConfig
from peakfit.engine.domain.params_vector import FitParameters
from peakfit.engine.domain.state import FittingState
from peakfit.fit.pipeline import (
    ClusterFitTask,
    CorrectionSnapshot,
    PipelineCompletion,
    build_optimizer_config,
    complete_fit,
    fit_single_cluster_task,
)

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from peakfit.engine.results import FitResult
    from peakfit.engine.scientific.final_outcome import FinalFitOutcome
    from peakfit.engine.scientific.simulation import FinalModelSnapshot
    from peakfit.gui.fitting import PreparedLocalFit


@dataclass(frozen=True, slots=True)
class LocalFitResult:
    """Owned inputs and authoritative completed quantities, separate from peak edits."""

    inputs: PreparedLocalFit
    outcome: FinalFitOutcome
    simulation_snapshot: FinalModelSnapshot
    optimizer_config: VarProConfig
    warnings: tuple[str, ...]
    execution_seconds: float
    peak_memory_bytes: int | None
    software_version: str
    run_id: str
    completed_at: str


@dataclass(frozen=True, slots=True)
class LocalFitAttempt:
    """Pickleable terminal engine values; frozen by the receiver before Qt delivery."""

    result: FitResult
    evaluation: FitEvaluation
    optimizer_config: VarProConfig
    warnings: tuple[str, ...]
    execution_seconds: float
    peak_memory_bytes: int | None


def execution_unavailable(prepared: PreparedLocalFit) -> str | None:
    """Explain why a reviewed preparation cannot be executed without approximations."""
    if prepared.cluster is None or prepared.parameters is None:
        return prepared.limitation or "This scope has no executable numerical inputs."
    if not prepared.parameters.get_vary_names():
        return "Local optimization requires at least one variable nonlinear parameter."
    if not np.isfinite(prepared.noise) or prepared.noise <= 0:
        return "Statistical noise must be positive and finite."
    if np.any(prepared.cluster.corrections != 0):
        return "Isolated local execution does not support cross-cluster corrections."
    return None


def execute_local_fit(prepared: PreparedLocalFit) -> LocalFitResult:
    """Fit a private copy of the reviewed inputs, without segmentation or parameter setup."""
    inputs = deepcopy(prepared)
    return finish_local_fit(inputs, compute_local_fit(inputs))


def compute_local_fit(prepared: PreparedLocalFit) -> LocalFitAttempt:
    """Execute the same scientific operation in-process or in the isolated child."""
    reason = execution_unavailable(prepared)
    if reason:
        raise ValueError(reason)
    assert prepared.cluster is not None
    assert prepared.parameters is not None
    config = PeakFitConfig.model_validate_json(prepared.snapshot.configuration)
    optimizer_config = build_optimizer_config(config, "varpro")
    assert isinstance(optimizer_config, VarProConfig)
    # The optimizer mutates parameters; the reviewed starting values remain intact.
    params = prepared.parameters.copy(deep=True)
    task = ClusterFitTask(
        prepared.cluster.cluster_id,
        0,
        prepared.cluster,
        params,
        prepared.noise,
        "varpro",
        optimizer_config,
    )
    started = perf_counter()
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        result = fit_single_cluster_task(task)
        evaluation = classify_optimizer_result(
            cluster=prepared.cluster,
            result=result,
            noise=prepared.noise,
        )
    return LocalFitAttempt(
        result,
        evaluation,
        optimizer_config,
        tuple(dict.fromkeys(str(item.message) for item in captured)),
        perf_counter() - started,
        _peak_memory_bytes(),
    )


def _peak_memory_bytes() -> int | None:
    if sys.platform == "win32":
        return None
    import resource  # noqa: PLC0415

    memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(memory if sys.platform == "darwin" else memory * 1024)


def finish_local_fit(prepared: PreparedLocalFit, attempt: LocalFitAttempt) -> LocalFitResult:
    """Use regular completion validation and freezing on a real one-pass terminal state."""
    cluster = prepared.cluster
    assert cluster is not None
    assert prepared.parameters is not None
    final_params = (
        attempt.result.params if attempt.evaluation.usable else prepared.parameters.copy(deep=True)
    )
    state = FittingState(
        clusters=[cluster],
        params=FitParameters.from_parameters(final_params, cluster.peaks),
        scalar_params=final_params,
        noise=prepared.noise,
    )
    completed = complete_fit(
        PipelineCompletion(
            state=state,
            results=[attempt.result],
            evaluations=[attempt.evaluation],
            correction_snapshot=CorrectionSnapshot.from_clusters([cluster], 0),
            n_optimizer_passes=1,
            n_correction_updates=0,
        )
    )
    # Retained samples/indices are independent of the viewer and read-only.
    cluster.data.flags.writeable = False
    cluster.corrections.flags.writeable = False
    for indices in cluster.grid_indices:
        indices.flags.writeable = False
    return LocalFitResult(
        prepared,
        completed.final_outcome,
        completed.simulation_snapshot,
        attempt.optimizer_config,
        attempt.warnings,
        attempt.execution_seconds,
        attempt.peak_memory_bytes,
        version("peakfit"),
        uuid4().hex,
        datetime.now(UTC).isoformat(),
    )


def run_fit_child(connection: Connection) -> None:
    """Spawn entry point: no Qt, files, full spectrum, or frozen mapping proxies."""
    try:
        connection.send("ready")
        prepared = connection.recv()
        try:
            connection.send(compute_local_fit(prepared))
        except Exception as error:
            connection.send(f"{type(error).__name__}: {error}")
    finally:
        connection.close()
