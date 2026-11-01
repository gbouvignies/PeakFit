"""What changed between the active run and the run it was refitted from (specification §9).

Both runs are read as recorded; nothing is refitted. Peaks are matched by stable
working ID. The comparison is descriptive: it reports how the fit, the warnings
and the clustering changed, and the largest position differences, without
judging whether the optimizer moved peaks too much.
"""

import math
from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING

from peakfit.engine.domain.spectrum import get_dimension_label

if TYPE_CHECKING:
    from peakfit.engine.domain.spectrum import SpectralParameters
    from peakfit.gui.dataset_fit import DatasetFitResult
    from peakfit.gui.editor.review import ClusterReview

DISPLACEMENTS = 5  # largest position differences listed


@dataclass(frozen=True, slots=True)
class ClusterChange:
    """An active cluster whose membership, warnings or highlights differ from before.

    ``regrouped_from`` names the parent clusters its members came from when the
    membership differs; then ``before`` and ``now`` describe different groupings and
    a mark that disappears is not a resolved problem.
    """

    ids: tuple[str, ...]
    names: tuple[str, ...]
    regrouped_from: tuple[str, ...]  # empty when the members are exactly the same
    before: tuple[str, ...]  # "▲ Uncertainty unreliable", "◇ Linewidth outlier", …
    now: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Displacement:
    """A fitted position difference between the runs, as neutral information."""

    working_id: str
    name: str
    axis: str  # display title, for example "¹H"
    delta: float  # ppm, active − parent
    linewidths: float  # |Δ| in Hz over the active fitted linewidth on that axis


@dataclass(frozen=True, slots=True)
class RunChanges:
    """A compact account of one run against its parent."""

    parent_id: str
    clusters: tuple[int, int]  # before, after
    warned: tuple[int, int]  # clusters with numerical warnings, before and after
    median_chi2: tuple[float, float]  # over clusters with a recorded χ²ᵣ
    regrouped: int  # active clusters whose members were grouped differently before
    changes: tuple[ClusterChange, ...]  # warnings first, then highlights, then regroupings
    displacements: tuple[Displacement, ...]  # largest relative to linewidth first
    added: int  # peaks in this run only
    removed: int  # peaks in the parent run only
    uncompared: int  # common peaks unusable in either run: no fitted position to compare


def _marks(review: ClusterReview) -> tuple[str, ...]:
    warnings = sorted({f"▲ {w.kind}" for w in review.warnings})
    highlights = sorted({f"◇ {h.kind}" for h in review.highlights})
    return (*warnings, *highlights)


def _label(review: ClusterReview) -> str:
    names = review.names
    if not names:
        return "–"
    return names[0] + (f" +{len(names) - 1}" if len(names) > 1 else "")


def _median_chi2(reviews: list[ClusterReview]) -> float:
    values = [r.chi2 for r in reviews if math.isfinite(r.chi2)]
    return median(values) if values else math.nan


def _fitted(result: DatasetFitResult, axes: int) -> dict[str, list[tuple[float, float]]]:
    """Fitted (position ppm, linewidth Hz) per spectral axis for peaks of usable clusters."""
    values = {
        p.name: float(p.value)
        for cluster in result.outcome.clusters
        if cluster.usable
        for p in cluster.final_nonlinear_parameters
    }
    labels = [get_dimension_label(index) for index in range(axes)]
    fitted = {}
    for identity in result.inputs.identities:
        engine = identity.engine_name
        names = [(f"{engine}.{a}.cs", f"{engine}.{a}.lw") for a in labels]
        if all(cs in values and lw in values for cs, lw in names):
            fitted[identity.working_id] = [(values[cs], values[lw]) for cs, lw in names]
    return fitted


def compare_runs(
    active: DatasetFitResult,
    active_reviews: list[ClusterReview],
    parent: DatasetFitResult,
    parent_reviews: list[ClusterReview],
    axes: list[SpectralParameters],
    titles: list[str],
) -> RunChanges:
    """Compare two recorded runs; ``axes`` and ``titles`` follow peak-position order."""
    home = {identity: review for review in parent_reviews for identity in review.ids}
    changes, regrouped = [], 0
    for review in active_reviews:
        sources = list({id(home[i]): home[i] for i in review.ids if i in home}.values())
        if not sources:
            continue  # only peaks new to this run
        same = len(sources) == 1 and set(sources[0].ids) == set(review.ids)
        before = tuple(dict.fromkeys(m for source in sources for m in _marks(source)))
        now = _marks(review)
        if same and set(before) == set(now):
            continue
        regrouped += not same
        changes.append(
            ClusterChange(
                review.ids,
                review.names,
                () if same else tuple(_label(source) for source in sources),
                before,
                now,
            )
        )
    # Warnings first, then highlights, then pure regroupings.
    changes.sort(
        key=lambda c: (
            not any(m.startswith("▲") for m in (*c.before, *c.now)),
            not (c.before or c.now),
        )
    )
    names = {i.working_id: i.assignment for i in active.inputs.identities}
    now_fit, old_fit = _fitted(active, len(axes)), _fitted(parent, len(axes))
    now_ids = {i.working_id for i in active.inputs.identities}
    old_ids = {i.working_id for i in parent.inputs.identities}
    displacements, uncompared = [], 0
    for identity in sorted(now_ids & old_ids):
        current, previous = now_fit.get(identity), old_fit.get(identity)
        if current is None or previous is None:
            uncompared += 1
            continue
        moves = [
            Displacement(
                identity,
                names[identity],
                title,
                cs - old_cs,
                abs(cs - old_cs) * float(axis.obs) / lw if lw > 0 else math.inf,
            )
            for axis, title, (cs, lw), (old_cs, _old_lw) in zip(
                axes, titles, current, previous, strict=True
            )
        ]
        largest = max(moves, key=lambda move: move.linewidths)
        if largest.delta != 0:
            displacements.append(largest)
    displacements.sort(key=lambda move: move.linewidths, reverse=True)
    return RunChanges(
        parent.run_id,
        (len(parent_reviews), len(active_reviews)),
        (
            sum(bool(r.warnings) for r in parent_reviews),
            sum(bool(r.warnings) for r in active_reviews),
        ),
        (_median_chi2(parent_reviews), _median_chi2(active_reviews)),
        regrouped,
        tuple(changes),
        tuple(displacements[:DISPLACEMENTS]),
        added=len(now_ids - old_ids),
        removed=len(old_ids - now_ids),
        uncompared=uncompared,
    )
