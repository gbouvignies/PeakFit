"""Changes between the active run and the run it was refitted from (specification §9).

Both runs are read as recorded: the review of each (numerical warnings and
highlights at the same display thresholds) and their fitted positions. Nothing
is refitted. Peaks are matched by stable working ID, so renamed peaks still
compare; peaks present in only one run are counted, not compared.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from peakfit.engine.domain.spectrum import get_dimension_label

if TYPE_CHECKING:
    from peakfit.gui.dataset_fit import DatasetFitResult
    from peakfit.gui.editor.review import ClusterReview


@dataclass(frozen=True, slots=True)
class ClusterChange:
    """A cluster of the active run whose warnings or highlights differ from before."""

    ids: tuple[str, ...]
    names: tuple[str, ...]
    gained: tuple[str, ...]  # "▲ Uncertainty unreliable", "◇ Position shift", …
    lost: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PeakShift:
    """A fitted position that moved by more than the active run's σ on one axis."""

    working_id: str
    name: str
    axis: str  # display title, for example "¹H"
    delta: float  # ppm, active − parent
    sigma: float  # the active run's standard error, ppm

    @property
    def ratio(self) -> float:
        """Shift in units of σ."""
        return abs(self.delta) / self.sigma


@dataclass(frozen=True, slots=True)
class RunChanges:
    """Everything the analysis page lists about one run against its parent."""

    parent_id: str
    clusters: tuple[ClusterChange, ...]
    shifts: tuple[PeakShift, ...]  # largest first
    added: int  # peaks fitted now but not in the parent run
    removed: int  # peaks fitted in the parent run but not now
    unjudged: int  # common peaks not judged: unusable in either run, or σ unavailable


def _marks(review: ClusterReview) -> set[str]:
    return {f"▲ {w.kind}" for w in review.warnings} | {f"◇ {h.kind}" for h in review.highlights}


def _positions(result: DatasetFitResult, axes: int) -> dict[str, list[tuple[float, float]]]:
    """Fitted (value, σ) per spectral axis, keyed by working ID, in peak-position order.

    Only usable clusters have fitted positions to compare.
    """
    engines = {i.engine_name: i.working_id for i in result.inputs.identities}
    values = {
        p.name: p
        for cluster in result.outcome.clusters
        if cluster.usable
        for p in cluster.final_nonlinear_parameters
    }
    positions = {}
    for engine, working in engines.items():
        found = [values.get(f"{engine}.{get_dimension_label(index)}.cs") for index in range(axes)]
        if all(p is not None for p in found):
            positions[working] = [(p.value, p.standard_error) for p in found if p is not None]
    return positions


def compare_runs(
    active: DatasetFitResult,
    active_reviews: list[ClusterReview],
    parent: DatasetFitResult,
    parent_reviews: list[ClusterReview],
    titles: list[str],
) -> RunChanges:
    """Compare two recorded runs; ``titles`` names the spectral axes in position order."""
    before: dict[str, set[str]] = {}
    for review in parent_reviews:
        for identity in review.ids:
            before[identity] = _marks(review)
    clusters = []
    for review in active_reviews:
        common = [identity for identity in review.ids if identity in before]
        if not common:
            continue
        previous = set().union(*(before[identity] for identity in common))
        now = _marks(review)
        if now != previous:
            clusters.append(
                ClusterChange(
                    review.ids,
                    review.names,
                    tuple(sorted(now - previous)),
                    tuple(sorted(previous - now)),
                )
            )
    names = {i.working_id: i.assignment for i in active.inputs.identities}
    now_positions = _positions(active, len(titles))
    old_positions = _positions(parent, len(titles))
    now_ids = {i.working_id for i in active.inputs.identities}
    old_ids = {i.working_id for i in parent.inputs.identities}
    shifts, unjudged = [], 0
    for identity in sorted(now_ids & old_ids):
        current, old = now_positions.get(identity), old_positions.get(identity)
        if current is None or old is None:
            unjudged += 1
            continue
        judged = True
        for title, (value, sigma), (previous, _old_sigma) in zip(titles, current, old, strict=True):
            if not (math.isfinite(sigma) and sigma > 0):
                judged = False
                continue
            if abs(value - previous) > sigma:
                shifts.append(PeakShift(identity, names[identity], title, value - previous, sigma))
        unjudged += not judged
    shifts.sort(key=lambda shift: shift.ratio, reverse=True)
    return RunChanges(
        parent.run_id,
        tuple(clusters),
        tuple(shifts),
        added=len(now_ids - old_ids),
        removed=len(old_ids - now_ids),
        unjudged=unjudged,
    )
