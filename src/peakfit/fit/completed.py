"""Retained terminal fitting inputs alongside the authoritative completed outcome."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from peakfit.fit.simulation import FinalModelSnapshot

if TYPE_CHECKING:
    from peakfit.engine.domain.state import FittingState
    from peakfit.fit.final_outcome import FinalFitOutcome
    from peakfit.fit.pipeline import CorrectionSnapshot
    from peakfit.shared.typing import FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class FinalClusterSamples:
    """Exact paired sample order and numerical inputs of one terminal problem.

    ``data`` and ``neighbor_corrections`` are (points, fitted series). They are
    inputs, not predictions. Corrected observations are their difference; recorded
    model/residual values belong exclusively to the completed outcome.
    """

    cluster_id: int
    grid_indices: tuple[IntArray, ...]
    data: FloatArray
    neighbor_corrections: FloatArray


@dataclass(frozen=True, slots=True)
class CompletedFit:
    """Minimal review record captured before continuation state can change.

    The outcome alone owns fitted values and diagnostics. The model snapshot
    owns copied evaluable lineshape geometry. Samples retain the actual terminal
    fitting inputs, including unusable clusters. This is an in-memory contract,
    not a persisted run, a dataset identity, or an MCMC continuation guarantee.
    """

    outcome: FinalFitOutcome
    model_snapshot: FinalModelSnapshot
    samples: tuple[FinalClusterSamples, ...]

    @classmethod
    def capture(
        cls,
        outcome: FinalFitOutcome,
        state: FittingState,
        corrections: CorrectionSnapshot,
    ) -> CompletedFit:
        """Validate terminal associations and copy inputs without fitting again."""
        model_snapshot = FinalModelSnapshot.capture(outcome, state, corrections)
        by_id = {cluster.cluster_id: cluster for cluster in state.clusters}
        samples = []
        for terminal in outcome.clusters:
            cluster = by_id[terminal.cluster_id]
            data = _copy_readonly(cluster.data)
            correction = _copy_readonly(corrections.corrections[terminal.cluster_id])
            evaluation = terminal.analytical_evaluation
            if evaluation is not None and not np.array_equal(
                data - correction - evaluation.model_values, evaluation.raw_residuals
            ):
                raise ValueError(
                    f"Recorded residuals disagree with terminal inputs for cluster_id "
                    f"{terminal.cluster_id}"
                )
            samples.append(
                FinalClusterSamples(
                    terminal.cluster_id,
                    tuple(_copy_readonly(index) for index in cluster.grid_indices),
                    data,
                    correction,
                )
            )
        return cls(outcome, model_snapshot, tuple(samples))


def _copy_readonly[T: np.ndarray](values: T) -> T:
    copied = values.copy()
    copied.flags.writeable = False
    return copied
