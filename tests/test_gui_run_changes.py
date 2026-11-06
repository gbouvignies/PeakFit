"""Run-to-run changes describe two recorded runs without refitting or judging either."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from builders import two_cluster_dataset

from peakfit.engine.algorithms.evaluation import FitOutcomeClassification
from peakfit.engine.algorithms.noise_correlation import not_modelled
from peakfit.engine.domain.config import FitConfig, PeakFitConfig
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.editor.changes import compare_runs
from peakfit.gui.editor.review import ClusterReview, Highlight, NumericalWarning, review_clusters
from peakfit.gui.editor.workspace import _change_entries
from peakfit.gui.fit_positions import continuation_source, dataset_completion
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.spectrum import read_dataset


@pytest.fixture(scope="module")
def run():
    spectra, _peaks, _config, prepared = two_cluster_dataset()
    result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    return spectra, result


def _titles(spectra):
    return [f"axis {i}" for i in range(len(spectra.spectral_params))]


def _reviews(spectra, result):
    return review_clusters(result, list(spectra.spectral_params), _titles(spectra))


def _compare(spectra, active, parent, active_reviews=None, parent_reviews=None):
    return compare_runs(
        active,
        _reviews(spectra, active) if active_reviews is None else active_reviews,
        parent,
        _reviews(spectra, parent) if parent_reviews is None else parent_reviews,
        list(spectra.spectral_params),
        _titles(spectra),
    )


def _with_outcome(result, clusters):
    outcome = replace(result.outcome, clusters=tuple(clusters))
    return replace(result, completed_fit=replace(result.completed_fit, outcome=outcome))


def _review(ids, warnings=(), chi2=1.0, cluster_id=900):
    return ClusterReview(
        cluster_id,
        tuple(ids),
        tuple(i.upper() for i in ids),
        tuple(NumericalWarning(kind, "") for kind in warnings),
        chi2,
        3.0,
        100,
        float("nan"),
        (),
    )


def test_identical_runs_report_nothing_to_investigate(run):
    spectra, result = run
    changes = _compare(spectra, result, result)
    assert changes.changes == ()
    assert changes.displacements == ()
    assert changes.regrouped == 0
    assert changes.clusters[0] == changes.clusters[1]
    assert (changes.added, changes.removed, changes.uncompared) == (0, 0, 0)


def test_position_differences_are_reported_in_ppm_and_linewidths(run):
    spectra, result = run
    cluster = next(c for c in result.outcome.clusters if c.usable)
    moved = next(p for p in cluster.final_nonlinear_parameters if p.name.endswith(".cs") and p.vary)
    engine, axis, _kind = moved.name.split(".")
    width = next(
        p.value for p in cluster.final_nonlinear_parameters if p.name == f"{engine}.{axis}.lw"
    )
    index = int(axis[1:]) - 2
    obs = float(spectra.spectral_params[index].obs)
    step = 0.02 * width / obs  # 0.02 linewidth, in ppm
    parent = _with_outcome(
        result,
        [
            replace(
                c,
                final_nonlinear_parameters=tuple(
                    replace(p, value=p.value - step) if p is moved else p
                    for p in c.final_nonlinear_parameters
                ),
            )
            if c is cluster
            else c
            for c in result.outcome.clusters
        ],
    )
    changes = _compare(spectra, result, parent)
    (move,) = changes.displacements
    working = next(i.working_id for i in result.inputs.identities if i.engine_name == engine)
    assert move.working_id == working
    assert move.axis == f"axis {index}"
    assert move.delta == pytest.approx(step)
    assert move.linewidths == pytest.approx(0.02)
    # Neutral information: nothing about position differences is a warning or highlight.
    assert changes.changes == ()


def test_warning_on_the_same_members_is_reported_neutrally(run):
    spectra, result = run
    active = [_review(["a", "b"])]
    parent = [_review(["a", "b"], ["Uncertainty unreliable"])]
    changes = _compare(spectra, result, result, active, parent)
    (change,) = changes.changes
    assert change.regrouped_from == ()
    entries = _change_entries(changes, {})
    detail = next(e[2] for e in entries if e[0] == "link" and e[1] == "A +1")
    assert detail.splitlines()[0] == "▲ Uncertainty unreliable no longer reported"
    assert "resolved" not in detail


def test_regrouped_clusters_show_before_and_now_without_claiming_a_resolution(run):
    spectra, result = run
    active = [_review(["a", "b"])]
    parent = [_review(["a"], ["Uncertainty unreliable"]), _review(["b"])]
    changes = _compare(spectra, result, result, active, parent)
    (change,) = changes.changes
    assert change.regrouped_from == ("A", "B")
    assert changes.regrouped == 1
    detail = next(e[2] for e in _change_entries(changes, {}) if e[0] == "link")
    assert detail == "Regrouped from A, B\nBefore regrouping ▲ Uncertainty unreliable"
    assert "no longer reported" not in detail


def test_fit_and_clustering_summary(run):
    spectra, result = run
    active = [_review(["a", "b"], chi2=2.0), _review(["c"], ["Not converged"], chi2=4.0)]
    parent = [_review(["a"], chi2=1.0), _review(["b"], chi2=3.0), _review(["c"], chi2=5.0)]
    changes = _compare(spectra, result, result, active, parent)
    assert changes.clusters == (3, 2)
    assert changes.warned == (0, 1)
    entries = _change_entries(changes, {})
    lines = [e[1] for e in entries if e[0] == "line"]
    # No run-level χ²ᵣ: a median over clusters with different samples is not comparable.
    assert lines == ["Clusters with ▲ engine warnings 0 → 1", "Clusters 3 → 2 · 1 regrouped"]
    headings = [e[1] for e in entries if e[0] == "heading"]
    assert headings.index("Clusters") < headings.index("Whole run")


def test_peaks_only_in_one_run_and_unusable_peaks_are_counted_not_compared(run):
    spectra, result = run
    dropped = result.inputs.identities[0]
    parent = replace(
        result,
        inputs=replace(
            result.inputs,
            identities=tuple(i for i in result.inputs.identities if i is not dropped),
        ),
    )
    changes = _compare(spectra, result, parent)
    assert (changes.added, changes.removed) == (1, 0)
    unusable = next(c for c in result.outcome.clusters if c.usable)
    broken = _with_outcome(
        result,
        [
            replace(c, classification=FitOutcomeClassification.UNUSABLE) if c is unusable else c
            for c in result.outcome.clusters
        ],
    )
    changes = _compare(spectra, result, broken)
    assert changes.uncompared == len(unusable.peak_names)
    engines = {i.engine_name: i.working_id for i in result.inputs.identities}
    excluded = {engines[name] for name in unusable.peak_names}
    assert not excluded & {move.working_id for move in changes.displacements}


def _real(spectra, result, warned=False):
    """The run's own reviews; the first cluster's parent copy optionally warned."""
    reviews = _reviews(spectra, result)
    first = reviews[0]
    if warned:
        first = replace(first, warnings=(NumericalWarning("Uncertainty unreliable", ""),))
    return [first, *reviews[1:]]


def _first_change(spectra, active, parent):
    changes = _compare(
        spectra, active, parent, _real(spectra, active), _real(spectra, parent, True)
    )
    return changes.changes[0]


def _edit_cluster(result, cluster_id, **fields):
    clusters = [
        replace(c, **{k: v(c) for k, v in fields.items()}) if c.cluster_id == cluster_id else c
        for c in result.outcome.clusters
    ]
    return _with_outcome(result, clusters)


def test_measures_are_compared_only_with_one_definition(run):
    spectra, result = run
    review = _reviews(spectra, result)[0]
    change = _first_change(spectra, result, result)
    assert change.lost == ("▲ Uncertainty unreliable",)
    before, after = change.measures
    assert before.chi2 == after.chi2 == review.chi2
    assert change.not_compared == ""
    detail = next(
        e[2]
        for e in _change_entries(
            _compare(spectra, result, result, _real(spectra, result), _real(spectra, result, True)),
            {},
        )
        if e[0] == "link"
    )
    assert f"χ²ᵣ {review.chi2:.2f} → {review.chi2:.2f}" in detail
    # Fewer fitting samples in the parent: a different problem, not compared.
    samples = [
        replace(s, grid_indices=tuple(axis[:-1] for axis in s.grid_indices))
        if s.cluster_id == review.cluster_id
        else s
        for s in result.completed_fit.samples
    ]
    resampled = replace(result, completed_fit=replace(result.completed_fit, samples=tuple(samples)))
    n = len(next(s for s in samples if s.cluster_id == review.cluster_id).grid_indices[0])
    change = _first_change(spectra, result, resampled)
    assert change.measures is None
    assert change.not_compared == f"fitting samples changed ({n:,} → {n + 1:,} per plane)"
    noisier = replace(
        result,
        completed_fit=replace(
            result.completed_fit, outcome=replace(result.outcome, noise=2 * result.outcome.noise)
        ),
    )
    assert _first_change(spectra, result, noisier).not_compared == "σ_stat changed"
    other = (
        not_modelled("test")
        if result.outcome.noise_model.modelled
        else replace(result.outcome.noise_model, reason="other")
    )
    remodelled = replace(
        result,
        completed_fit=replace(
            result.completed_fit, outcome=replace(result.outcome, noise_model=other)
        ),
    )
    assert _first_change(spectra, result, remodelled).not_compared == "noise model changed"

    def fewer(c):
        statistics = c.analytical_evaluation.statistics
        return replace(
            c.analytical_evaluation,
            statistics=replace(
                statistics,
                n_nonlinear_parameters=statistics.n_nonlinear_parameters - 1,
                n_fitted_parameters=statistics.n_fitted_parameters - 1,
            ),
        )

    fixed = _edit_cluster(result, review.cluster_id, analytical_evaluation=fewer)
    assert _first_change(spectra, result, fixed).not_compared.startswith(
        "varied parameters changed"
    )


def test_new_warnings_lead_and_are_drawn_as_warnings(run):
    spectra, result = run
    active = [
        _review(["c"], ["Not converged"], cluster_id=901),
        _review(["a"], cluster_id=902),
        _review(["b"], cluster_id=903),
    ]
    parent = [
        _review(["c"], cluster_id=901),
        _review(["a"], ["At bound"], cluster_id=902),
        replace(_review(["b"], cluster_id=903), highlights=(Highlight("Linewidth outlier", "x"),)),
    ]
    changes = _compare(spectra, result, result, active, parent)
    assert [c.names for c in changes.changes] == [("C",), ("A",), ("B",)]
    links = [e for e in _change_entries(changes, {}) if e[0] == "link"]
    assert [e[5] for e in links] == [True, False, False]


DATA = Path(__file__).resolve().parents[1] / "examples/02-advanced-fitting/data"


def _bundled_fit(spectra, peaks, settings, completion=None):
    source = continuation_source(completion) if completion is not None else None
    prepared = prepare_dataset_fit(spectra, tuple(peaks), settings, source)
    result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    return dataset_completion(
        result, result.inputs.snapshot, submitted_revision=0, manual_revision=0
    )


@pytest.mark.slow
def test_the_degenerate_refit_leads_with_new_warnings_and_compares_nothing_it_cannot():
    # The GUI assessment's correction: 147N-H moved onto a +3σ residual, then refit.
    # The pair collapses (χ²ᵣ 3.00 → 2.53), but on different fitting samples.
    dataset = read_dataset(DATA / "pseudo3d.ft2", DATA / "pseudo3d.list", DATA / "b1_offsets.txt")
    settings = PeakFitConfig(fitting=FitConfig(refine_iterations=2))
    parent = _bundled_fit(dataset.spectra, dataset.peaks, settings)
    moved = tuple(
        replace(p, positions=(115.155, 8.3488)) if p.name == "147N-H" else p
        for p in parent.working_peaks
    )
    active = _bundled_fit(dataset.spectra, moved, settings, parent)
    axes = list(dataset.spectra.spectral_params)
    titles = [f"axis {i}" for i in range(len(axes))]
    changes = compare_runs(
        active.result,
        review_clusters(active.result, axes, titles),
        parent.result,
        review_clusters(parent.result, axes, titles),
        axes,
        titles,
    )
    first = changes.changes[0]
    assert set(first.names) == {"53N-H", "147N-H"}
    assert first.edited == ("147N-H",)
    assert {"▲ Near-collinear shapes", "▲ Uncertainty unreliable"} <= set(first.gained)
    assert first.measures is None
    assert first.not_compared == "fitting samples changed (67 → 68 per plane)"
    entries = _change_entries(changes, dict.fromkeys(titles, 4))
    link = next(e for e in entries if e[0] == "link")
    assert link[5]  # drawn as a warning
    assert link[2].splitlines()[:2] == [
        "▲ Near-collinear shapes now reported",
        "▲ Uncertainty unreliable now reported",
    ]
    assert "Edited: 147N-H" in link[2]
    assert "Measures not compared: fitting samples changed (67 → 68 per plane)" in link[2]
    assert "2.53" not in link[2]
    assert not any("Median" in str(entry) for entry in entries)
