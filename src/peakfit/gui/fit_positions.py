"""Validated position-only projection of a retained fit onto stable working IDs."""

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from peakfit.engine.domain.spectrum import SpectralParameters
from peakfit.engine.lineshapes.utils import get_axis_label
from peakfit.engine.scientific.coordinates import resolved_coordinates

if TYPE_CHECKING:
    from peakfit.gui.dataset_fit import DatasetFitResult
    from peakfit.gui.fitting import FitInputSnapshot
    from peakfit.gui.local_fit import LocalFitResult
    from peakfit.gui.peaks import WorkingPeak

_SPECTRAL_DIMENSIONS = 2


def fitted_position_updates(
    result: LocalFitResult, peaks: tuple[WorkingPeak, ...]
) -> dict[str, tuple[float, float]]:
    """Read final cs values in shape/working order, without folding or optimizing.

    The caller also guards live scientific-input revisions and an active optimizer.
    Fixed cs values are retained final quantities too, even when no cs varied.
    """
    inputs = result.inputs
    if peaks != inputs.snapshot.peaks:
        raise ValueError("Working peaks changed since this fit. Prepare and run a new fit.")
    try:
        terminal = result.outcome.cluster(inputs.scope.cluster_id)
    except KeyError as error:
        raise ValueError("The fitted group is missing from the retained result.") from error
    if not terminal.usable or terminal.analytical_evaluation is None:
        raise ValueError("This result has no numerically usable fitted positions.")
    _require_reliable_uncertainties(terminal.analytical_evaluation.uncertainty_warning)
    identities = inputs.identities
    engine_names = [item.engine_name for item in identities]
    working_ids = [item.working_id for item in identities]
    members = inputs.scope.peaks
    if (
        not identities
        or len(set(engine_names)) != len(engine_names)
        or len(set(working_ids)) != len(working_ids)
        or len({p.id for p in peaks}) != len(peaks)
        or len({p.name for p in members}) != len(members)
        or len(set(terminal.peak_names)) != len(terminal.peak_names)
        or set(engine_names) != set(terminal.peak_names)
        or set(engine_names) != {p.name for p in members}
        or not set(working_ids) <= {p.id for p in peaks}
    ):
        raise ValueError("The fitted group has missing or ambiguous peak identities.")
    parameters = terminal.final_nonlinear_parameters
    final = {p.name: p for p in parameters}
    if len(final) != len(parameters):
        raise ValueError("The fitted result has ambiguous parameters.")
    axes = [
        SpectralParameters.model_validate_json(axis) for axis in inputs.snapshot.spectral_parameters
    ][1:]
    by_engine = {p.name: p for p in members}
    by_id = {p.id: p for p in peaks}
    expected = set()
    updates = {}
    for identity in identities:
        shapes = by_engine[identity.engine_name].shapes
        labels = [axis.label or get_axis_label(dim) for dim, axis in enumerate(axes, 1)]
        if (
            len(shapes) != _SPECTRAL_DIMENSIONS
            or len(axes) != _SPECTRAL_DIMENSIONS
            or len(set(labels)) != _SPECTRAL_DIMENSIONS
            or [s.axis for s in shapes] != labels
        ):
            raise ValueError("The fitted positions require two unambiguous spectral axes.")
        coordinates = []
        for shape, axis in zip(shapes, axes, strict=True):
            name = f"{identity.engine_name}.{shape.axis}.cs"
            expected.add(name)
            parameter = final.get(name)
            if parameter is None:
                raise ValueError(f"Missing fitted chemical shift: {name}.")
            value = parameter.value
            point = float(axis.ppm2pts(value))
            if (
                not math.isfinite(value)
                or not parameter.min <= value <= parameter.max
                or not math.isfinite(point)
                or (axis.direct and not 0 <= point <= axis.size - 1)
            ):
                raise ValueError(f"Invalid fitted chemical shift: {name}.")
            coordinates.append(value)
        positions = (coordinates[0], coordinates[1])
        if positions != by_id[identity.working_id].positions:
            updates[identity.working_id] = positions
    if {name for name in final if name.endswith(".cs")} != expected:
        raise ValueError("The fitted result has unexpected chemical-shift axes.")
    return updates


def _require_reliable_uncertainties(warning: str | None) -> None:
    if warning:
        raise ValueError(warning)


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
) -> DatasetCompletion:
    """Validate the whole completion before returning any adopted coordinates.

    Fatal source/identity errors produce no value to publish. Unusable or invalid
    individual pairs retain their exact original positions with specific warnings.
    Convergence and uncertainty warnings do not invalidate usable finite positions.
    No accepted numerical quantity, bound, mask or shape start is rewritten here.
    """
    submitted = result.inputs.snapshot
    if current != submitted or submitted_revision != manual_revision:
        raise ValueError("Scientific inputs changed during fitting; completion was not published.")
    identities = result.inputs.identities
    working = {p.id: p for p in submitted.peaks}
    engine = {item.engine_name: item.working_id for item in identities}
    members = {p.name: p for cluster in result.simulation_snapshot.clusters for p in cluster.peaks}
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
    updates = {}
    warnings = []
    for cluster in result.outcome.clusters:
        for name in cluster.peak_names:
            identity = engine[name]
            peak = working[identity]
            _names, coordinates, reason = resolved_coordinates(
                cluster, name, [s.axis for s in members[name].shapes], axes
            )
            if reason:
                warnings.append(f"{peak.name} [{identity}]: retained starting position; {reason}.")
            else:
                assert coordinates is not None
                updates[identity] = (coordinates[0], coordinates[1])
    post_fit = tuple(
        replace(p, positions=updates[p.id]) if p.id in updates else p for p in submitted.peaks
    )
    return DatasetCompletion(result, post_fit, manual_revision, tuple(warnings))
