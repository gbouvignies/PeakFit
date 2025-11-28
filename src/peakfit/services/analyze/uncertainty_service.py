"""Service producing covariance-based parameter uncertainty summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from peakfit.core.domain.state import FittingState


@dataclass(slots=True, frozen=True)
class ParameterUncertaintyEntry:
    """Snapshot of a varying parameter with relative-error metadata."""

    name: str
    value: float
    stderr: float
    rel_error_pct: float | None
    at_boundary: bool
    min_bound: float
    max_bound: float


@dataclass(slots=True, frozen=True)
class ParameterUncertaintyResult:
    """Aggregate result for uncertainty reporting."""

    parameters: list[ParameterUncertaintyEntry]
    boundary_parameters: list[ParameterUncertaintyEntry]
    large_uncertainty_parameters: list[ParameterUncertaintyEntry]


class NoVaryingParametersFoundError(RuntimeError):
    """Raised when the state contains no varying parameters."""


class ParameterUncertaintyService:
    """Builds parameter uncertainty summaries from a fitting state."""

    LARGE_UNCERTAINTY_THRESHOLD = 0.1  # 10%

    @staticmethod
    def analyze(state: FittingState) -> ParameterUncertaintyResult:
        """Analyze varying parameters in a FittingState and return uncertainty summary.

        Args:
            state: Fitting state containing parameters and fitting metadata

        Returns
        -------
            ParameterUncertaintyResult summarizing parameter uncertainties
        """
        params = state.params
        vary_names = params.get_vary_names()
        if not vary_names:
            raise NoVaryingParametersFoundError("No varying parameters found")

        entries: list[ParameterUncertaintyEntry] = []
        boundary_entries: list[ParameterUncertaintyEntry] = []
        large_uncertainty: list[ParameterUncertaintyEntry] = []

        for name in vary_names:
            param = params[name]
            rel_error = None
            if param.value != 0 and param.stderr > 0:
                rel_error = abs(param.stderr / param.value)

            entry = ParameterUncertaintyEntry(
                name=name,
                value=param.value,
                stderr=param.stderr,
                rel_error_pct=rel_error * 100 if rel_error is not None else None,
                at_boundary=param.is_at_boundary(),
                min_bound=param.min,
                max_bound=param.max,
            )
            entries.append(entry)

            if entry.at_boundary:
                boundary_entries.append(entry)
            if (
                rel_error is not None
                and rel_error > ParameterUncertaintyService.LARGE_UNCERTAINTY_THRESHOLD
            ):
                large_uncertainty.append(entry)

        return ParameterUncertaintyResult(
            parameters=entries,
            boundary_parameters=boundary_entries,
            large_uncertainty_parameters=large_uncertainty,
        )


__all__ = [
    "NoVaryingParametersFoundError",
    "ParameterUncertaintyEntry",
    "ParameterUncertaintyResult",
    "ParameterUncertaintyService",
]
