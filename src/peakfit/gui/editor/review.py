"""Cluster review read from a completed run: numerical warnings, measures, highlights.

The three families stay separate (specification §8). Numerical warnings are the
engine's recorded classifications. Measures are recorded numbers. Highlights are
descriptive rules with adjustable display thresholds; they change no scientific
data, run state or warning count. Nothing here refits or recomputes statistics.
"""

import math
from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.domain.spectrum import get_dimension_label

if TYPE_CHECKING:
    from peakfit.engine.domain.spectrum import SpectralParameters
    from peakfit.gui.dataset_fit import DatasetFitResult

BOUND_TOLERANCE = 1e-6  # fraction of a parameter's range


@dataclass(frozen=True, slots=True)
class HighlightRules:
    """Display thresholds for highlights; never part of a run or its warnings."""

    correlation: float = 0.9  # |ρ| at or above
    linewidth_low: float = 0.5  # fitted linewidth below this × the axis median
    linewidth_high: float = 2.0  # or above this × the axis median


@dataclass(frozen=True, slots=True)
class NumericalWarning:
    """One recorded numerical warning: its kind and the engine's detail."""

    kind: str  # Unusable, Not converged, Uncertainty unreliable, At bound
    detail: str


@dataclass(frozen=True, slots=True)
class Highlight:
    """One heuristic or measure highlight, labelled with the rule that raised it."""

    kind: str  # Amplitude correlation or Linewidth outlier
    rule: str  # for example "LW 0.43 × median (¹H)"
    peak: str | None = None  # working ID, or None for the whole cluster


@dataclass(frozen=True, slots=True)
class ClusterReview:
    """Everything the navigator and inspector show about one recorded cluster."""

    cluster_id: int
    ids: tuple[str, ...]
    names: tuple[str, ...]
    warnings: tuple[NumericalWarning, ...]
    chi2: float  # NaN when no analytical evaluation was recorded
    largest_residual: float  # in σ; NaN when unavailable
    samples: int
    correlation: float  # largest |ρ|; NaN when unavailable or one peak
    highlights: tuple[Highlight, ...]

    @property
    def single(self) -> bool:
        """A one-peak cluster has no amplitude correlation."""
        return len(self.ids) == 1


WARNING_KINDS = ("Unusable", "Not converged", "Uncertainty unreliable", "At bound")
HIGHLIGHT_KINDS = ("Amplitude correlation", "Linewidth outlier")


def _parameters(cluster) -> dict[tuple[str, str, str], float]:
    """Fitted linewidths keyed by (engine peak, axis, "lw")."""
    values = {}
    for parameter in cluster.final_nonlinear_parameters:
        parts = parameter.name.split(".")
        if len(parts) == 3 and parts[2] == "lw":  # noqa: PLR2004
            values[tuple(parts)] = float(parameter.value)
    return values


def _warnings(cluster) -> tuple[NumericalWarning, ...]:
    warnings = []
    if not cluster.usable:
        warnings.append(
            NumericalWarning("Unusable", cluster.unusable_reason or "no reason recorded")
        )
    elif not cluster.optimizer_provenance.converged:
        warnings.append(
            NumericalWarning(
                "Not converged", cluster.optimizer_provenance.termination_message or ""
            )
        )
    evaluation = cluster.analytical_evaluation
    if evaluation is not None and evaluation.uncertainty_warning:
        warnings.append(NumericalWarning("Uncertainty unreliable", evaluation.uncertainty_warning))
    bounded = [
        p.name
        for p in cluster.final_nonlinear_parameters
        if p.vary
        and math.isfinite(p.max - p.min)
        and min(p.value - p.min, p.max - p.value) <= BOUND_TOLERANCE * (p.max - p.min)
    ]
    if bounded:
        warnings.append(NumericalWarning("At bound", ", ".join(bounded)))
    return tuple(warnings)


def correlation_matrix(cluster) -> np.ndarray | None:
    """Recorded amplitude correlations, or None when the covariance is unavailable."""
    evaluation = cluster.analytical_evaluation
    if evaluation is None:
        return None
    covariance = np.asarray(evaluation.amplitude_covariance, dtype=float)
    if not np.all(np.isfinite(covariance)):
        return None
    scale = np.sqrt(np.diag(covariance))
    with np.errstate(divide="ignore", invalid="ignore"):
        return covariance / np.outer(scale, scale)


def review_clusters(
    result: DatasetFitResult,
    axes: list[SpectralParameters],
    titles: list[str],
    rules: HighlightRules = HighlightRules(),  # noqa: B008
) -> list[ClusterReview]:
    """Review every recorded cluster of a run with the given highlight thresholds.

    ``axes`` are the spectral axes in peak-position order and ``titles`` their
    display names (for example "¹H"). Parameters name axes by the engine rule.
    """
    identities = {i.engine_name: i for i in result.inputs.identities}
    order = [get_dimension_label(index) for index in range(len(axes))]
    names_of = dict(zip(order, titles, strict=True))
    fitted = {c.cluster_id: _parameters(c) for c in result.outcome.clusters}
    medians = {}
    for label in order:
        widths = [
            value
            for cluster in result.outcome.clusters
            if cluster.usable
            for (_peak, axis, kind), value in fitted[cluster.cluster_id].items()
            if axis == label and kind == "lw" and math.isfinite(value) and value > 0
        ]
        medians[label] = median(widths) if widths else math.nan
    reviews = []
    for cluster in result.outcome.clusters:
        names = [n for n in cluster.peak_names if n in identities]
        values = fitted[cluster.cluster_id]
        highlights = []
        correlation = math.nan
        matrix = correlation_matrix(cluster)
        if matrix is not None and len(names) > 1:
            off = matrix.copy()
            np.fill_diagonal(off, 0.0)
            correlation = float(np.nanmax(np.abs(off)))
            if correlation >= rules.correlation:
                highlights.append(
                    Highlight(
                        "Amplitude correlation",
                        f"|ρ| {correlation:.2f} ≥ {rules.correlation:g}",
                    )
                )
        if cluster.usable:
            for name in names:
                working = identities[name].working_id
                for label in order:
                    lw = values.get((name, label, "lw"))
                    if lw is None or not lw > 0:
                        continue
                    ratio = lw / medians[label]
                    if not rules.linewidth_low <= ratio <= rules.linewidth_high:
                        highlights.append(
                            Highlight(
                                "Linewidth outlier",
                                f"LW {ratio:.2f} × median ({names_of[label]})",
                                working,
                            )
                        )
        evaluation = cluster.analytical_evaluation
        sampled = cluster.sample_evaluation
        residuals = (
            np.asarray(sampled.normalized_residuals, dtype=float) if sampled is not None else None
        )
        reviews.append(
            ClusterReview(
                cluster.cluster_id,
                tuple(identities[n].working_id for n in names),
                tuple(identities[n].assignment for n in names),
                _warnings(cluster),
                float(evaluation.statistics.reduced_chi_squared) if evaluation else math.nan,
                float(np.max(np.abs(residuals)))
                if residuals is not None and residuals.size
                else math.nan,
                int(evaluation.statistics.n_observations) if evaluation else 0,
                correlation,
                tuple(highlights),
            )
        )
    return reviews
