"""Exact run coordinates and the shared automatic position-adoption rule."""

import math

import numpy as np

from peakfit.engine.domain.spectrum import SpectralParameters
from peakfit.engine.lineshapes.utils import get_axis_label


def resolved_coordinates(terminal, engine_name, shape_axes, axes):
    """Return a valid fitted pair/tuple or the reason to retain the submitted position.

    Convergence and uncertainty warnings do not veto usable finite coordinates.
    Never fold indirect coordinates or substitute a partial fitted tuple.
    """
    labels = [axis.label or get_axis_label(dim) for dim, axis in enumerate(axes, 1)]
    if list(shape_axes) != labels:
        raise ValueError("Completion chemical-shift axes do not match the recorded model.")
    names = tuple(f"{engine_name}.{axis}.cs" for axis in shape_axes)
    final = {p.name: p for p in terminal.final_nonlinear_parameters}
    if len(final) != len(terminal.final_nonlinear_parameters):
        raise ValueError("Completion contains ambiguous chemical-shift parameters.")
    if not terminal.usable:
        return names, None, "unusable cluster"
    values = []
    for name, axis in zip(names, axes, strict=True):
        parameter = final.get(name)
        if parameter is None or not math.isfinite(parameter.value):
            return names, None, "missing or non-finite chemical-shift pair"
        value = parameter.value
        point = float(axis.ppm2pts(value))
        if (
            not math.isfinite(point)
            or not parameter.min <= value <= parameter.max
            or (axis.direct and not 0 <= point <= axis.size - 1)
        ):
            return names, None, "unmappable or out-of-bounds chemical-shift pair"
        values.append(value)
    return names, tuple(values), None


def coordinate_table(
    completed, identities, spectral_parameters, run_id, *, initial=None, adopted=None
):
    """Build one stable-ID-keyed scientific table, separate from live GUI state.

    ``adopted`` is an optional recorded completion transition, never the mutable
    export-time peak list. Its absence explicitly means no transition was recorded.
    CLI and GUI use identical fields; CLI does not invent working-coordinate adoption.
    """
    axes = [SpectralParameters.model_validate(axis) for axis in spectral_parameters][1:]
    by_name = {i["engine_name"]: i for i in identities}
    expected = {i["peak_id"] for i in identities}
    if initial is not None and not expected <= initial.keys():
        raise ValueError("Missing initial coordinates for exported peak identities.")
    if adopted is not None and not expected <= adopted.keys():
        raise ValueError("Missing adopted coordinates for exported peak identities.")
    rows = {}
    for cluster in completed.model_snapshot.clusters:
        terminal = completed.outcome.cluster(cluster.cluster_id)
        for peak in cluster.peaks:
            identity = by_name[peak.name]
            peak_id = identity["peak_id"]
            submitted = np.asarray(initial[peak_id] if initial is not None else peak.positions)
            if submitted.shape != (len(axes),) or not np.all(np.isfinite(submitted)):
                raise ValueError("Invalid initial unfolded coordinates.")
            names, fitted, reason = resolved_coordinates(
                terminal, peak.name, [s.axis for s in peak.shapes], axes
            )
            status = "not_recorded"
            adopted_values = None
            adoption_reason = "no working-coordinate transition recorded"
            if adopted is not None:
                adopted_values = np.asarray(adopted[peak_id])
                resolved = fitted if fitted is not None else submitted
                if not np.array_equal(adopted_values, resolved):
                    raise ValueError(
                        "Recorded coordinate adoption disagrees with scientific evidence."
                    )
                status = "adopted_fitted" if fitted is not None else "retained_initial"
                adoption_reason = (
                    "usable finite fitted coordinates" if fitted is not None else reason
                )
            rows[peak_id] = {
                "engine_name": peak.name,
                "assignment": identity["assignment"],
                "cluster_id": cluster.cluster_id,
                "axes": [s.axis for s in peak.shapes],
                "parameter_names": names,
                "initial_unfolded_ppm": submitted,
                "fitted_unfolded_ppm": np.asarray(fitted) if fitted is not None else None,
                "resolution_reason": reason,
                "adopted_unfolded_ppm": adopted_values,
                "adoption_status": status,
                "adoption_reason": adoption_reason,
            }
    if set(rows) != expected:
        raise ValueError("Coordinate table identities disagree with the exported result.")
    return {"schema_version": "1.0.0", "run_id": run_id, "unit": "ppm", "peaks": rows}


def validate_coordinate_table(table, outcome, context):
    """Check a recorded coordinate projection against its immutable scientific sources."""
    if (
        table["schema_version"] != "1.0.0"
        or table["unit"] != "ppm"
        or table["run_id"] != context["metadata"]["run_id"]
    ):
        raise ValueError("Invalid scientific coordinate table header.")
    identities = {i["peak_id"]: i for i in context["peak_identities"]}
    if set(table["peaks"]) != set(identities):
        raise ValueError("Coordinate table peak identities disagree with the result.")
    axes = [SpectralParameters.model_validate(axis) for axis in context["spectral_parameters"]][1:]
    models = {p["name"]: (c["cluster_id"], p) for c in context["models"] for p in c["peaks"]}
    for peak_id, row in table["peaks"].items():
        identity = identities[peak_id]
        cid, model = models[identity["engine_name"]]
        if (
            row["engine_name"] != identity["engine_name"]
            or row["assignment"] != identity["assignment"]
            or row["cluster_id"] != cid
            or row["axes"] != [s["axis"] for s in model["shapes"]]
        ):
            raise ValueError("Coordinate table model/parameter mapping disagrees with the result.")
        names, fitted, reason = resolved_coordinates(
            outcome.cluster(cid), model["name"], row["axes"], axes
        )
        if tuple(row["parameter_names"]) != names or row["resolution_reason"] != reason:
            raise ValueError("Coordinate table resolution disagrees with the result.")
        if not np.array_equal(row["initial_unfolded_ppm"], model["positions"]):
            raise ValueError("Initial coordinates disagree with the recorded model inputs.")
        recorded = row["fitted_unfolded_ppm"]
        if (fitted is None) != (recorded is None) or (
            fitted is not None and not np.array_equal(recorded, fitted)
        ):
            raise ValueError("Fitted coordinates disagree with immutable engine parameters.")
        status, adopted = row["adoption_status"], row["adopted_unfolded_ppm"]
        if status == "not_recorded":
            valid = (
                adopted is None
                and row["adoption_reason"] == "no working-coordinate transition recorded"
            )
        elif status == "adopted_fitted":
            valid = (
                fitted is not None
                and np.array_equal(adopted, fitted)
                and row["adoption_reason"] == "usable finite fitted coordinates"
            )
        elif status == "retained_initial":
            valid = (
                fitted is None
                and np.array_equal(adopted, row["initial_unfolded_ppm"])
                and row["adoption_reason"] == reason
            )
        else:
            valid = False
        if not valid:
            raise ValueError("Invalid recorded scientific coordinate adoption.")
