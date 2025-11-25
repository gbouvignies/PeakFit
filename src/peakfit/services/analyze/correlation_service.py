"""Service for summarizing parameter correlations inputs."""

from __future__ import annotations

from dataclasses import dataclass

from peakfit.core.domain.state import FittingState


@dataclass(slots=True, frozen=True)
class ParameterCorrelationEntry:
    """Lightweight snapshot of a varying parameter for correlation summaries."""

    name: str
    value: float
    stderr: float
    min_bound: float
    max_bound: float
    at_boundary: bool


@dataclass(slots=True, frozen=True)
class ParameterCorrelationResult:
    """Result payload for correlation-oriented summaries."""

    parameters: list[ParameterCorrelationEntry]
    boundary_parameters: list[ParameterCorrelationEntry]


class NotEnoughVaryingParametersError(RuntimeError):
    """Raised when fewer than two varying parameters are available."""

    def __init__(self, vary_names: list[str]):
        super().__init__("At least two varying parameters are required")
        self.vary_names = vary_names


class ParameterCorrelationService:
    """Aggregates parameter metadata for correlation diagnostics."""

    @staticmethod
    def analyze(state: FittingState) -> ParameterCorrelationResult:
        params = state.params
        vary_names = params.get_vary_names()
        if len(vary_names) < 2:
            raise NotEnoughVaryingParametersError(vary_names)

        entries: list[ParameterCorrelationEntry] = []
        boundary_entries: list[ParameterCorrelationEntry] = []
        for name in vary_names:
            param = params[name]
            entry = ParameterCorrelationEntry(
                name=name,
                value=param.value,
                stderr=param.stderr,
                min_bound=param.min,
                max_bound=param.max,
                at_boundary=param.is_at_boundary(),
            )
            entries.append(entry)
            if entry.at_boundary:
                boundary_entries.append(entry)

        return ParameterCorrelationResult(parameters=entries, boundary_parameters=boundary_entries)


__all__ = [
    "NotEnoughVaryingParametersError",
    "ParameterCorrelationEntry",
    "ParameterCorrelationResult",
    "ParameterCorrelationService",
]
