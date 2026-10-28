"""Validated position-only projection of a retained fit onto stable working IDs."""

import math
from typing import TYPE_CHECKING

from peakfit.engine.domain.spectrum import SpectralParameters
from peakfit.engine.lineshapes.utils import get_axis_label

if TYPE_CHECKING:
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
