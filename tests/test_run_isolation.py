"""A fit run owns its correction and parameter state; no state leaks between runs."""

import numpy as np
from builders import two_cluster_dataset

from peakfit.engine.domain.params_scalar import Parameter, Parameters
from peakfit.fit.pipeline import run_pipeline


def _run(config, prepared):
    peaks = [p for c in prepared.clusters for p in c.peaks]
    return run_pipeline(config, prepared.clusters, prepared.noise, prepared.parameters, peaks)


def _recorded(result):
    outcome = result.completed_fit.outcome
    return (
        [(p.name, p.value, p.standard_error) for p in outcome.final_nonlinear_parameters],
        outcome.statistics.chi_squared,
    )


def test_a_run_starts_uncorrected_and_leaves_its_input_clusters_unchanged():
    _spectra, _peaks, config, prepared = two_cluster_dataset()
    reference = _run(config, prepared)
    # Corrections left on the input clusters by earlier work must not seed this run.
    stale = {}
    for cluster in prepared.clusters:
        cluster.corrections = np.full_like(cluster.data, 0.5)
        stale[cluster.cluster_id] = cluster.corrections.copy()

    again = _run(config, prepared)

    assert _recorded(again) == _recorded(reference)
    assert _recorded(_run(config, prepared)) == _recorded(reference)
    for cluster in prepared.clusters:
        np.testing.assert_array_equal(cluster.corrections, stale[cluster.cluster_id])
    owned = {id(c) for c in again.continuation_state.clusters}
    assert owned.isdisjoint(id(c) for c in prepared.clusters)


def test_parameter_sets_share_no_parameter_objects_after_update():
    source = Parameters()
    source.add("a", value=1.0, min_value=0.0, max_value=2.0)
    target = Parameters()
    target.update(source)

    assert target["a"] is not source["a"]
    target["a"].value = 1.5
    assert source["a"].value == 1.0
    # The source keeps its own vary bookkeeping.
    source["a"].vary = False
    assert source.get_vary_names() == []
    assert target.get_vary_names() == ["a"]
    assert isinstance(target["a"], Parameter)
