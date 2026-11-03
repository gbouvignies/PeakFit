"""Read fitted nonlinear values from a previous fit to use as starting values."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import ValidationError

from peakfit.io.schemas import FitSummarySchema
from peakfit.shared.paths import format_path

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["read_fitted_nonlinear_values"]


def read_fitted_nonlinear_values(path: Path) -> dict[str, float]:
    """Return each fitted nonlinear parameter value of a completed-fit ``fit.json``.

    Amplitudes are not returned: they are linear and solved analytically for any
    nonlinear values. Unusable clusters record no parameters, so their peaks keep
    their usual starts.
    """
    if not path.is_file():
        msg = f"parameters.from_file not found: {format_path(path)}"
        raise FileNotFoundError(msg)
    try:
        with path.open(encoding="utf-8") as handle:
            summary = FitSummarySchema.model_validate(json.load(handle))
    except (json.JSONDecodeError, ValidationError) as exc:
        msg = (
            f"parameters.from_file {format_path(path)} is not a PeakFit fit summary: {exc}\n"
            "Point it to the summary/fit.json written by a completed fit with this "
            "PeakFit version."
        )
        raise ValueError(msg) from exc
    return {parameter.name: parameter.value for parameter in summary.final_nonlinear_parameters}
