"""What changed between the active run and the run it was refitted from (specification §9).

Both runs are read as recorded; nothing is refitted. Peaks are matched by stable
working ID. The comparison is descriptive: it reports how the fit, the warnings
and the clustering changed, and the largest position differences, without
judging whether the optimizer moved peaks too much.

Recorded measures (χ²ᵣ, |r|max, |ρ|) of a cluster are compared only when both runs
define them identically: the same members, the same fitting samples, the same σ_stat
and noise model, and the same number of varied parameters. Otherwise they describe
different problems, and the change says why they are not compared. A lower χ²ᵣ is
never presented as a better fit.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.diagnostics.identifiability import Cause, amplitude_identifiability
from peakfit.engine.domain.spectrum import get_dimension_label

if TYPE_CHECKING:
    from peakfit.engine.domain.spectrum import SpectralParameters
    from peakfit.gui.dataset_fit import DatasetFitResult
    from peakfit.gui.editor.review import ClusterReview

DISPLACEMENTS = 5  # largest position differences listed


@dataclass(frozen=True, slots=True)
class Measures:
    """One cluster's recorded review measures in one run."""

    chi2: float
    largest_residual: float  # in σ
    correlation: float  # largest |ρ|; NaN for one peak or when unavailable


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
    edited: tuple[str, ...] = ()  # assignments moved or added since the parent run
    measures: tuple[Measures, Measures] | None = None  # before, after; when comparable
    not_compared: str = ""  # why the recorded measures are not compared

    @property
    def gained(self) -> tuple[str, ...]:
        """Numerical warnings reported now and not before."""
        return tuple(m for m in self.now if m.startswith("▲") and m not in self.before)

    @property
    def lost(self) -> tuple[str, ...]:
        """Numerical warnings no longer reported (not a resolution when regrouped)."""
        return tuple(m for m in self.before if m.startswith("▲") and m not in self.now)


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
    regrouped: int  # active clusters whose members were grouped differently before
    changes: tuple[ClusterChange, ...]  # new or lost warnings, edits, highlights, regroupings
    displacements: tuple[Displacement, ...]  # largest relative to linewidth first
    added: int  # peaks in this run only
    removed: int  # peaks in the parent run only
    uncompared: int  # common peaks unusable in either run: no fitted position to compare


_DECOMPOSITION = {
    Cause.NULL_SPACE: "▲ Amplitudes not identifiable",
    Cause.NEAR_COLLINEAR: "▲ Near-collinear shapes",
    Cause.ILL_CONDITIONED: "▲ Ill-conditioned shapes",
}


def _marks(result: DatasetFitResult, review: ClusterReview) -> tuple[str, ...]:
    """Engine warnings, the amplitude decomposition's status, then highlights."""
    warnings = {f"▲ {w.kind}" for w in review.warnings}
    cluster = result.outcome.by_cluster_id.get(review.cluster_id)
    evaluation = cluster.analytical_evaluation if cluster is not None else None
    if cluster is not None and cluster.usable and evaluation is not None:
        members = amplitude_identifiability(evaluation.shapes, evaluation.amplitudes)
        warnings |= {_DECOMPOSITION[m.cause] for m in members if m.cause in _DECOMPOSITION}
    highlights = sorted({f"◇ {h.kind}" for h in review.highlights})
    return (*sorted(warnings), *highlights)


def _sorted_points(indices) -> np.ndarray:
    """Sampled grid points as rows, in a canonical order."""
    points = np.column_stack([np.asarray(axis) for axis in indices])
    return points[np.lexsort(points.T[::-1])] if points.size else points


def _comparison(
    active: DatasetFitResult,
    review: ClusterReview,
    parent: DatasetFitResult,
    source: ClusterReview,
) -> tuple[tuple[Measures, Measures] | None, str]:
    """The recorded measures of both runs when they share one definition, or why not."""
    if set(review.ids) != set(source.ids):
        return None, "members changed"
    before = parent.outcome.by_cluster_id.get(source.cluster_id)
    after = active.outcome.by_cluster_id.get(review.cluster_id)
    if (
        before is None
        or after is None
        or not (before.usable and after.usable)
        or before.analytical_evaluation is None
        or after.analytical_evaluation is None
    ):
        return None, "no recorded evaluation in one run"
    old = before.analytical_evaluation.statistics
    new = after.analytical_evaluation.statistics
    points = [
        _sorted_points(
            next(s for s in r.completed_fit.samples if s.cluster_id == c.cluster_id).grid_indices
        )
        for r, c in ((parent, source), (active, review))
    ]
    # The definition must match in this order; the first difference is reported.
    checks = (
        (
            points[0].shape != points[1].shape or not np.array_equal(points[0], points[1]),
            f"fitting samples changed ({len(points[0]):,} → {len(points[1]):,} per plane)",
        ),
        (parent.outcome.noise != active.outcome.noise, "σ_stat changed"),
        (parent.outcome.noise_model != active.outcome.noise_model, "noise model changed"),
        (
            (old.n_nonlinear_parameters, old.n_fitted_parameters)
            != (new.n_nonlinear_parameters, new.n_fitted_parameters),
            f"varied parameters changed ({old.n_fitted_parameters} → {new.n_fitted_parameters})",
        ),
    )
    why = next((text for differs, text in checks if differs), "")
    if why:
        return None, why
    return (
        Measures(source.chi2, source.largest_residual, source.correlation),
        Measures(review.chi2, review.largest_residual, review.correlation),
    ), ""


def _edited(active: DatasetFitResult, parent: DatasetFitResult) -> set[str]:
    """Working IDs moved or added since the parent run, as the active run recorded them."""
    continuation = active.inputs.continuation
    if continuation is None or continuation.run_id != parent.run_id:
        return set()
    return {
        identity
        for identity, reason in continuation.reasons
        if reason.startswith(("moved since run", "added since run"))
    }


def _label(review: ClusterReview) -> str:
    names = review.names
    if not names:
        return "–"
    return names[0] + (f" +{len(names) - 1}" if len(names) > 1 else "")


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
    edited = _edited(active, parent)
    changes, regrouped = [], 0
    for review in active_reviews:
        sources = list({id(home[i]): home[i] for i in review.ids if i in home}.values())
        if not sources:
            continue  # only peaks new to this run
        same = len(sources) == 1 and set(sources[0].ids) == set(review.ids)
        before = tuple(dict.fromkeys(m for source in sources for m in _marks(parent, source)))
        now = _marks(active, review)
        touched = tuple(n for i, n in zip(review.ids, review.names, strict=True) if i in edited)
        if same and set(before) == set(now) and not touched:
            continue
        regrouped += not same
        measures, why = _comparison(active, review, parent, sources[0])
        changes.append(
            ClusterChange(
                review.ids,
                review.names,
                () if same else tuple(_label(source) for source in sources),
                before,
                now,
                touched,
                measures,
                why,
            )
        )
    # New numerical warnings first, then lost ones, then edits, highlights, regroupings.
    changes.sort(
        key=lambda c: (
            not c.gained,
            not c.lost,
            not c.edited,
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
        regrouped,
        tuple(changes),
        tuple(displacements[:DISPLACEMENTS]),
        added=len(now_ids - old_ids),
        removed=len(old_ids - now_ids),
        uncompared=uncompared,
    )
