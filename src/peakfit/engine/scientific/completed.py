"""Retained terminal fitting inputs alongside the authoritative completed outcome."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.algorithms.covariance_factors import lineshape_order
from peakfit.engine.diagnostics.couplings import coupling_qualifications
from peakfit.engine.scientific.simulation import FinalModelSnapshot

if TYPE_CHECKING:
    from peakfit.engine.diagnostics.couplings import CouplingQualification
    from peakfit.engine.domain.state import FittingState
    from peakfit.engine.scientific.completion import CorrectionSnapshot
    from peakfit.engine.scientific.final_outcome import FinalFitOutcome
    from peakfit.shared.typing import FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class FinalClusterMapping:
    """Exact paired sample order; it is never a Cartesian grid product."""

    cluster_id: int
    grid_indices: tuple[IntArray, ...]


@dataclass(frozen=True, slots=True)
class FinalClusterInputs(FinalClusterMapping):
    """Irreducible terminal corrections; observations may come from a verified source."""

    neighbor_corrections: FloatArray
    observation_sha256: str
    observation_dtype: str


@dataclass(frozen=True, slots=True)
class FinalClusterSamples(FinalClusterMapping):
    """Exact observations and irreducible terminal corrections at mapped samples."""

    data: FloatArray
    neighbor_corrections: FloatArray


@dataclass(frozen=True, slots=True)
class CompletedFit:
    """Minimal review record captured before continuation state can change.

    The outcome alone owns fitted values and diagnostics. The model snapshot
    owns copied evaluable lineshape geometry. Samples carry either actual observations
    or their exact reconstruction identities, plus irreducible terminal corrections,
    including unusable clusters. This is an in-memory contract,
    not a persisted run, a dataset identity, or an MCMC continuation guarantee.
    """

    outcome: FinalFitOutcome
    model_snapshot: FinalModelSnapshot
    samples: tuple[FinalClusterSamples | FinalClusterInputs, ...]

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
            evaluation = terminal.sample_evaluation
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

    def couplings(self, cluster_id: int | None = None) -> dict[str, CouplingQualification]:
        """The recorded J couplings' qualifications, by parameter name; one cluster's if given.

        The recorded lineshape covariance's rows are named from each cluster's recorded
        model geometry (``lineshape_order``); the values are the recorded ones.
        """
        geometry = {cluster.cluster_id: cluster for cluster in self.model_snapshot.clusters}
        qualifications: dict[str, CouplingQualification] = {}
        for cluster in self.outcome.clusters:
            if cluster_id is not None and cluster.cluster_id != cluster_id:
                continue
            evaluation = cluster.analytical_evaluation
            model = geometry.get(cluster.cluster_id)
            if evaluation is None or model is None:
                continue
            qualifications.update(
                coupling_qualifications(
                    cluster.final_nonlinear_parameters,
                    lineshape_order(model.peaks),
                    evaluation.covariance_factors.parameter_covariance,
                )
            )
        return qualifications


def _copy_readonly[T: np.ndarray](values: T) -> T:
    copied = values.copy()
    copied.flags.writeable = False
    return copied
