"""What changed between the active run and a baseline run (specification §9).

The baseline is the run the active one was refitted from, its parent, unless the
scientist chooses another retained run. Both runs are read as recorded; nothing is
refitted. Peaks are matched by stable working ID. The comparison is descriptive: it
reports how the fit, the warnings and the clustering changed, and the largest
position differences, without judging whether the optimizer moved peaks too much.

Recorded measures (χ²ᵣ, |r|max, |ρ|) of a cluster are compared only when both runs
define them identically: the same members, the same fitted planes, the same
lineshape model and constraints, the same σ_stat and noise model, the same fitting
samples and the same number of varied parameters. Otherwise they describe different
problems, and the change says why they are not compared. A lower χ²ᵣ is never
presented as a better fit.

Runs are numbered by their place among the journal's completed runs, and a peak is
"edited" only when a refit after the baseline recorded it as moved or added by hand.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.spectrum import get_dimension_label
from peakfit.gui import formatting

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from peakfit.engine.domain.spectrum import SpectralParameters
    from peakfit.gui.dataset_fit import DatasetFitResult
    from peakfit.gui.editor.review import ClusterReview

DISPLACEMENTS = 5  # largest position differences listed
EDITS = ("moved since run", "added since run")  # continuation reasons of a manual edit


def run_numbers(journal: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Each completed run's number, 1-based in journal order; attempts get none."""
    completed = (e["run_id"] for e in journal if e.get("status") == "completed")
    return {identity: number for number, identity in enumerate(completed, 1)}


def edited_since(
    journal: Iterable[Mapping[str, Any]], active_id: str, baseline_id: str
) -> frozenset[str] | None:
    """Working IDs moved or added by hand in a refit after the baseline run.

    Each refit records, against the run it continued from, the peaks whose working
    position was moved or added by hand. Following the parent links from the active
    run back to the baseline collects them; an edit undone before a fit was never
    recorded. None when the baseline is not an earlier run of the active one.
    """
    entries = {e["run_id"]: e for e in journal if e.get("status") == "completed"}
    edited: set[str] = set()
    run, seen = active_id, set()
    while run != baseline_id:
        entry = entries.get(run)
        if entry is None or run in seen:
            return None
        seen.add(run)
        parent = entry.get("parent_run_id")
        continuation = entry.get("continuation")
        if continuation and continuation["run_id"] == parent:
            edited |= {i for i, reason in continuation["reasons"] if reason.startswith(EDITS)}
        if not parent:
            return None
        run = parent
    return frozenset(edited)


@dataclass(frozen=True, slots=True)
class Measures:
    """One cluster's recorded review measures in one run."""

    chi2: float
    largest_residual: float  # in σ
    correlation: float  # largest |ρ|; NaN for one peak or when unavailable


@dataclass(frozen=True, slots=True)
class ClusterChange:
    """An active cluster whose membership, warnings or highlights differ from before.

    ``regrouped_from`` names the baseline clusters its members came from when the
    membership differs; then ``before`` and ``now`` describe different groupings and
    a mark that disappears is not a resolved problem.
    """

    ids: tuple[str, ...]
    names: tuple[str, ...]
    regrouped_from: tuple[str, ...]  # empty when the members are exactly the same
    before: tuple[str, ...]  # "▲ Uncertainty unreliable", "◇ Linewidth outlier", …
    now: tuple[str, ...]
    edited: tuple[str, ...] = ()  # assignments moved or added by hand since the baseline
    measures: tuple[Measures, Measures] | None = None  # before, after; when comparable
    not_compared: str = ""  # why the recorded measures are not compared
    new: bool = False  # every member was added since the baseline: nothing before

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
    delta: float  # ppm, active − baseline
    linewidths: float  # |Δ| in Hz over the active fitted linewidth on that axis


@dataclass(frozen=True, slots=True)
class RunChanges:
    """A compact account of one run against a baseline run."""

    baseline_id: str
    clusters: tuple[int, int]  # before, after
    warned: tuple[int, int]  # clusters with numerical warnings, before and after
    regrouped: int  # active clusters whose members were grouped differently before
    changes: tuple[ClusterChange, ...]  # new or lost warnings, edits, highlights, regroupings
    displacements: tuple[Displacement, ...]  # largest relative to linewidth first
    added: int  # peaks in this run only
    removed: int  # peaks in the baseline run only
    uncompared: int  # common peaks unusable in either run: no fitted position to compare


def _marks(review: ClusterReview) -> tuple[str, ...]:
    """Numerical warnings (the amplitude decomposition's included), then highlights."""
    warnings = {f"▲ {w.kind}" for w in review.warnings}
    highlights = sorted({f"◇ {h.kind}" for h in review.highlights})
    return (*sorted(warnings), *highlights)


def _sorted_points(indices) -> np.ndarray:
    """Sampled grid points as rows, in a canonical order."""
    points = np.column_stack([np.asarray(axis) for axis in indices])
    return points[np.lexsort(points.T[::-1])] if points.size else points


def _definition(active: DatasetFitResult, baseline: DatasetFitResult, axes: int) -> str:
    """Why no measure of the two runs shares one definition, or "" when every one may.

    The fitted planes (their values and the excluded ones), the effective lineshape
    model of every axis and the constraints come from each run's recorded inputs.
    """
    now, before = active.inputs.snapshot, baseline.inputs.snapshot
    if (now.plane_values, now.source_plane_indices) != (
        before.plane_values,
        before.source_plane_indices,
    ):
        return "fitted planes changed"
    new = PeakFitConfig.model_validate_json(now.configuration)
    old = PeakFitConfig.model_validate_json(before.configuration)
    labels = [get_dimension_label(i) for i in range(axes)]
    if any(new.fitting.axis_model(a) != old.fitting.axis_model(a) for a in labels):
        return "lineshape model changed"
    if new.parameters != old.parameters:
        return "constraints changed"
    if baseline.outcome.noise != active.outcome.noise:
        return "σ_stat changed"
    if baseline.outcome.noise_model != active.outcome.noise_model:
        return "noise model changed"
    return ""


def _comparison(
    active: DatasetFitResult,
    review: ClusterReview,
    baseline: DatasetFitResult,
    source: ClusterReview,
    definition: str,
) -> tuple[tuple[Measures, Measures] | None, str]:
    """The recorded measures of both runs when they share one definition, or why not."""
    if set(review.ids) != set(source.ids):
        return None, "members changed"
    if definition:
        return None, definition
    before = baseline.outcome.by_cluster_id.get(source.cluster_id)
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
        for r, c in ((baseline, source), (active, review))
    ]
    # Then the cluster's own definition, in this order; the first difference is reported.
    checks = (
        (
            points[0].shape != points[1].shape or not np.array_equal(points[0], points[1]),
            f"fitting samples changed ({formatting.grouped(len(points[0]))} → "
            f"{formatting.grouped(len(points[1]))} per plane)",
        ),
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


def _edited(active: DatasetFitResult, parent: DatasetFitResult) -> frozenset[str]:
    """Working IDs moved or added since the parent run, as the active run recorded them."""
    continuation = active.inputs.continuation
    if continuation is None or continuation.run_id != parent.run_id:
        return frozenset()
    return frozenset(
        identity for identity, reason in continuation.reasons if reason.startswith(EDITS)
    )


def _label(review: ClusterReview) -> str:
    names = review.names
    if not names:
        return "–"
    return names[0] + (f" +{len(names) - 1}" if len(names) > 1 else "")


def _fitted(result: DatasetFitResult, axes: int) -> dict[str, list[tuple[float, float]]]:
    """Fitted (position ppm, linewidth Hz) per spectral axis for peaks of usable clusters."""
    index = result.index
    labels = [get_dimension_label(i) for i in range(axes)]
    fitted = {}
    for identity in result.inputs.identities:
        row = []
        for a in labels:
            cs = index.fitted(identity.engine_name, a, "cs", usable=True)
            lw = index.fitted(identity.engine_name, a, "lw", usable=True)
            if cs is None or lw is None:
                break
            row.append((float(cs.value), float(lw.value)))
        else:
            fitted[identity.working_id] = row
    return fitted


def compare_runs(
    active: DatasetFitResult,
    active_reviews: list[ClusterReview],
    baseline: DatasetFitResult,
    baseline_reviews: list[ClusterReview],
    axes: list[SpectralParameters],
    titles: list[str],
    *,
    edited: frozenset[str] | None = None,
) -> RunChanges:
    """Compare the active run with a baseline; ``axes`` and ``titles`` follow positions.

    ``edited`` names the peaks moved or added by hand since the baseline (see
    `edited_since`); by default, those the active run recorded against it when the
    baseline is the run it continued from, and none otherwise.
    """
    home = {identity: review for review in baseline_reviews for identity in review.ids}
    if edited is None:
        edited = _edited(active, baseline)
    definition = _definition(active, baseline, len(axes))
    changes, regrouped = [], 0
    for review in active_reviews:
        sources = list({id(home[i]): home[i] for i in review.ids if i in home}.values())
        if not sources:
            # Only peaks new to this run: its current marks, with nothing to compare.
            changes.append(
                ClusterChange(review.ids, review.names, (), (), _marks(review), new=True)
            )
            continue
        same = len(sources) == 1 and set(sources[0].ids) == set(review.ids)
        before = tuple(dict.fromkeys(m for source in sources for m in _marks(source)))
        now = _marks(review)
        touched = tuple(n for i, n in zip(review.ids, review.names, strict=True) if i in edited)
        if same and set(before) == set(now) and not touched:
            continue
        regrouped += not same
        measures, why = _comparison(active, review, baseline, sources[0], definition)
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
            not (c.edited or c.new),
            not (c.before or c.now),
        )
    )
    now, old = active.index.by_working, baseline.index.by_working
    now_fit, old_fit = _fitted(active, len(axes)), _fitted(baseline, len(axes))
    now_ids, old_ids = now.keys(), old.keys()
    displacements, uncompared = [], 0
    for identity in sorted(now_ids & old_ids):
        current, previous = now_fit.get(identity), old_fit.get(identity)
        if current is None or previous is None:
            uncompared += 1
            continue
        moves = [
            Displacement(
                identity,
                now[identity].assignment,
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
        baseline.run_id,
        (len(baseline_reviews), len(active_reviews)),
        (
            sum(bool(r.warnings) for r in baseline_reviews),
            sum(bool(r.warnings) for r in active_reviews),
        ),
        regrouped,
        tuple(changes),
        tuple(displacements[:DISPLACEMENTS]),
        added=len(now_ids - old_ids),
        removed=len(old_ids - now_ids),
        uncompared=uncompared,
    )
