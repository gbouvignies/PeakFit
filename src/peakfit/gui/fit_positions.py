"""Validated projection of a completed dataset fit onto stable working peak IDs."""

import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

from peakfit.engine.domain.spectrum import SpectralParameters
from peakfit.engine.lineshapes.utils import get_axis_label
from peakfit.engine.scientific.coordinates import collinear_partners, resolved_coordinates
from peakfit.gui.fitting import ContinuationSource

if TYPE_CHECKING:
    from peakfit.gui.dataset_fit import DatasetFitResult
    from peakfit.gui.fitting import FitInputSnapshot
    from peakfit.gui.peaks import WorkingPeak

_SPECTRAL_DIMENSIONS = 2


@dataclass(frozen=True, slots=True)
class DatasetCompletion:
    """One validated run-owned I → W transition, ready for atomic UI publication.

    This is a scientific value contract, not a fit launcher or a state framework.
    Milestone 2 installs this record and a fresh edit history in one UI delivery.
    The immutable outcome and exact submitted seeds remain in ``result.inputs``.
    """

    result: DatasetFitResult
    working_peaks: tuple[WorkingPeak, ...]
    manual_revision: int
    warnings: tuple[str, ...]
    adopted: frozenset[str] = frozenset()  # working IDs whose fitted positions were adopted
    # Working IDs whose position was retained, with why: unusable, invalid or a
    # near-collinear fitted shape (the fitted values themselves stay in the result).
    retained: tuple[tuple[str, str], ...] = ()

    def is_current(self, inputs: FitInputSnapshot, manual_revision: int) -> bool:
        """Compare against verified post-fit W, never against the original seeds I."""
        expected = replace(self.result.inputs.snapshot, peaks=self.working_peaks)
        return manual_revision == self.manual_revision and inputs == expected


def dataset_completion(
    result: DatasetFitResult,
    current: FitInputSnapshot,
    *,
    submitted_revision: int,
    manual_revision: int,
    recorded: tuple[WorkingPeak, ...] | None = None,
) -> DatasetCompletion:
    """Validate the whole completion before returning any adopted coordinates.

    Fatal source/identity errors produce no value to publish. Unusable or invalid
    individual pairs retain their exact original positions with specific warnings,
    and so do peaks whose fitted shape is near-collinear with another's: their
    split, and the positions that produced it, are poorly constrained. Convergence
    and uncertainty warnings do not invalidate usable finite positions. No accepted
    numerical quantity, bound, mask or shape start is rewritten here.

    ``recorded`` re-validates a saved transition instead of deciding afresh: a run
    recorded before the near-collinearity rule may have adopted such peaks.
    """
    submitted = result.inputs.snapshot
    if current != submitted or submitted_revision != manual_revision:
        raise ValueError("Scientific inputs changed during fitting; completion was not published.")
    identities = result.inputs.identities
    working = {p.id: p for p in submitted.peaks}
    engine = {item.engine_name: item.working_id for item in identities}
    snapshot = result.completed_fit.model_snapshot
    members = {p.name: p for cluster in snapshot.clusters for p in cluster.peaks}
    terminal_names = [name for cluster in result.outcome.clusters for name in cluster.peak_names]
    if (
        len(working) != len(submitted.peaks)
        or len(engine) != len(identities)
        or len({item.working_id for item in identities}) != len(identities)
        or set(engine.values()) != set(working)
        or set(engine) != set(members)
        or set(terminal_names) != set(engine)
        or len(terminal_names) != len(set(terminal_names))
    ):
        raise ValueError("Completion requires a complete, unambiguous stable-ID mapping.")
    axes = [SpectralParameters.model_validate_json(axis) for axis in submitted.spectral_parameters][
        1:
    ]
    labels = [axis.label or get_axis_label(dim) for dim, axis in enumerate(axes, 1)]
    if len(axes) != _SPECTRAL_DIMENSIONS or len(set(labels)) != _SPECTRAL_DIMENSIONS:
        raise ValueError("Completion requires two recorded, unambiguous spectral axes.")
    if any(
        not all(math.isfinite(v) for v in (axis.first, axis.delta, axis.obs))
        or axis.delta == 0
        or axis.obs <= 0
        for axis in axes
    ):
        raise ValueError("Completion requires valid recorded spectral calibration.")
    assignment = {item.engine_name: item.assignment for item in identities}
    saved = {p.id: tuple(p.positions) for p in recorded} if recorded is not None else None
    updates = {}
    warnings = []
    retained = []
    for cluster in result.outcome.clusters:
        for name in cluster.peak_names:
            identity = engine[name]
            peak = working[identity]
            _names, coordinates, reason = resolved_coordinates(
                cluster, name, [s.axis for s in members[name].shapes], axes
            )
            partners = collinear_partners(cluster, name) if coordinates is not None else ()
            if partners:
                reason = "near-collinear fitted shapes with " + ", ".join(
                    assignment[p] for p in partners
                )
            if saved is not None and coordinates is not None and saved[identity] == coordinates:
                reason = None  # adopted under the rule in force when the run was recorded
            if reason:
                warnings.append(f"{peak.name} [{identity}]: retained starting position; {reason}.")
                retained.append((identity, reason))
            else:
                assert coordinates is not None
                updates[identity] = (coordinates[0], coordinates[1])
    post_fit = tuple(
        replace(p, positions=updates[p.id]) if p.id in updates else p for p in submitted.peaks
    )
    if recorded is not None and post_fit != tuple(recorded):
        raise ValueError("Saved working transition disagrees with the fitted outcome.")
    return DatasetCompletion(
        result,
        post_fit,
        manual_revision,
        tuple(warnings),
        frozenset(updates),
        tuple(retained),
    )


def continuation_source(completion: DatasetCompletion) -> ContinuationSource:
    """The run's fitted nonlinear state for each peak whose position it adopted.

    Peaks of unusable clusters, and peaks whose fitted position was not adopted,
    have no values here and start from the configured defaults.
    """
    result = completion.result
    working = {i.engine_name: i.working_id for i in result.inputs.identities}
    values: dict[str, dict[str, float]] = {}
    for cluster in result.outcome.clusters:
        if not cluster.usable:
            continue
        for parameter in cluster.final_nonlinear_parameters:
            engine, _, suffix = parameter.name.partition(".")
            identity = working.get(engine)
            if identity in completion.adopted and math.isfinite(parameter.value):
                values.setdefault(identity, {})[suffix] = float(parameter.value)
    snapshot = result.inputs.snapshot
    return ContinuationSource(
        result.run_id,
        {p.id: p.positions for p in completion.working_peaks},
        {identity: MappingProxyType(v) for identity, v in values.items()},
        result.inputs.shape_names,
        snapshot.configuration,
        snapshot.spectral_parameters,
        MappingProxyType(dict(completion.retained)),
    )
