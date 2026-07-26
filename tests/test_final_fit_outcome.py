"""Contracts for immutable completed-fit outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pytest

from peakfit.engine.algorithms.evaluation import (
    AnalyticalModelEvaluation,
    FitOutcomeClassification,
    classify_optimizer_result,
)
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import FitConfig
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.params_vector import FitParameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters
from peakfit.engine.domain.state import FittingState
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.engine.results import FitResult
from peakfit.fit.final_outcome import FinalFitOutcome, finalize_fit
from peakfit.fit.pipeline import CorrectionSnapshot, PipelineCompletion


@dataclass(frozen=True)
class CompletionFixture:
    completion: PipelineCompletion
    clusters: tuple[Cluster, ...]


def _spectral_parameters(*, size: int, direct: bool) -> SpectralParameters:
    return SpectralParameters(
        size=size,
        sw=500.0,
        obs=500.0,
        car=0.0,
        aq_time=0.01,
        apocode=0.0,
        apodq1=0.0,
        apodq2=0.0,
        apodq3=0.0,
        p180=False,
        direct=direct,
        ft=True,
    )


def _completion_fixture(
    states: dict[int, tuple[bool, bool]],
) -> CompletionFixture:
    """Build terminal results with converged, usable, or unusable states."""
    n_points = 9
    n_series = 2
    spectra = Spectra(
        dic={},
        data=np.zeros((n_series, n_points), dtype=np.float64),
        z_values=np.array([0.0, 1.0], dtype=np.float64),
        params=[
            _spectral_parameters(size=n_series, direct=False),
            _spectral_parameters(size=n_points, direct=True),
        ],
    )
    clusters: list[Cluster] = []
    peaks: list[Peak] = []
    results: list[FitResult] = []
    evaluations = []
    for index, cluster_id in enumerate(sorted(states), start=1):
        center = float(spectra.spectral_params[0].pts2ppm(index + 2.0))
        peak = Peak(
            name=f"P{cluster_id}",
            positions=np.array([center], dtype=np.float64),
            shapes=create_shapes(
                spectra,
                FitConfig(lineshape="gaussian"),
                f"P{cluster_id}",
                [center],
                ["gaussian"],
            ),
        )
        peak.set_cluster_id(cluster_id)
        params = peak.create_params()
        grid_indices = [np.arange(n_points, dtype=np.intp)]
        empty = Cluster(
            cluster_id=cluster_id,
            peaks=[peak],
            grid_indices=grid_indices,
            data=np.zeros((n_points, n_series), dtype=np.float64),
        )
        data = empty.evaluate(params).T @ np.array([[2.0, 3.0]], dtype=np.float64)
        cluster = Cluster(
            cluster_id=cluster_id,
            peaks=[peak],
            grid_indices=grid_indices,
            data=data,
        )
        converged, usable = states[cluster_id]
        preliminary = classify_optimizer_result(
            cluster=cluster,
            result=FitResult(
                cluster_id=cluster_id,
                params=params,
                residual=np.zeros(cluster.n_observations, dtype=np.float64),
                cost=0.0,
                n_amplitude_params=cluster.n_amplitude_params,
            ),
            noise=1.0,
        )
        assert preliminary.analytical is not None
        residual = preliminary.analytical.normalized_residuals.copy()
        if not usable:
            residual[0] = np.nan
        result = FitResult(
            cluster_id=cluster_id,
            params=params,
            residual=residual,
            cost=0.5 * float(np.nansum(residual**2)),
            correction_revision=2,
            nfev=10 + index,
            njev=index,
            success=converged,
            message=f"terminal {cluster_id}",
            optimality=0.125 * index,
            n_amplitude_params=cluster.n_amplitude_params,
            metadata={"nested": {"cluster_id": cluster_id}, "values": [index]},
            optimizer_kind="varpro",
            noise=1.0,
        )
        evaluation = classify_optimizer_result(cluster=cluster, result=result, noise=1.0)
        clusters.append(cluster)
        peaks.append(peak)
        results.append(result)
        evaluations.append(evaluation)

    final_params = Parameters.from_peaks(peaks, fixed=False)
    state = FittingState(
        clusters=clusters,
        params=FitParameters.from_parameters(final_params, peaks),
        scalar_params=final_params,
        noise=1.0,
    )
    snapshot = CorrectionSnapshot(
        revision=2,
        corrections=MappingProxyType(
            {cluster.cluster_id: np.array(cluster.corrections, copy=True) for cluster in clusters}
        ),
    )
    return CompletionFixture(
        completion=PipelineCompletion(
            state=state,
            results=results,
            evaluations=evaluations,
            correction_snapshot=snapshot,
            n_optimizer_passes=3,
            n_correction_updates=2,
        ),
        clusters=tuple(clusters),
    )


@pytest.mark.parametrize(
    ("states", "expected"),
    [
        ({11: (True, True), 37: (True, True)}, ["converged", "converged"]),
        ({11: (False, True)}, ["usable_non_converged"]),
        ({11: (False, False)}, ["unusable"]),
        (
            {11: (True, True), 37: (False, True), 91: (False, False)},
            ["converged", "usable_non_converged", "unusable"],
        ),
    ],
)
def test_finalization_preserves_each_usability_classification(
    states: dict[int, tuple[bool, bool]],
    expected: list[str],
) -> None:
    outcome = finalize_fit(_completion_fixture(states).completion)

    assert [cluster.classification.value for cluster in outcome.clusters] == expected
    assert [cluster.cluster_id for cluster in outcome.clusters] == sorted(states)
    assert outcome.overall_converged is all(value == "converged" for value in expected)
    for cluster in outcome.clusters:
        if cluster.classification is FitOutcomeClassification.UNUSABLE:
            assert cluster.analytical_evaluation is None
            assert cluster.unusable_reason is not None
            assert cluster.final_nonlinear_parameters == ()
        else:
            assert cluster.analytical_evaluation is not None
            assert cluster.unusable_reason is None
            assert cluster.final_nonlinear_parameters


def test_finalization_uses_cluster_identity_for_lookup_and_display_order() -> None:
    fixture = _completion_fixture({91: (False, False), 11: (True, True), 37: (False, True)})
    fixture.completion.results.reverse()
    fixture.completion.evaluations.reverse()

    outcome = finalize_fit(fixture.completion)

    assert isinstance(outcome, FinalFitOutcome)
    assert [cluster.cluster_id for cluster in outcome.clusters] == [11, 37, 91]
    assert outcome.cluster(37).classification is FitOutcomeClassification.USABLE_NON_CONVERGED
    with pytest.raises(KeyError, match="999"):
        outcome.cluster(999)
    with pytest.raises(TypeError):
        outcome.by_cluster_id[11] = outcome.cluster(11)  # type: ignore[index]


def test_finalization_copies_ticket_three_evaluation_and_freezes_nested_values() -> None:
    fixture = _completion_fixture({11: (True, True)})
    source = fixture.completion.evaluations[0].analytical
    assert isinstance(source, AnalyticalModelEvaluation)
    fixture.clusters[0].data.fill(123.0)

    outcome = finalize_fit(fixture.completion)
    evaluation = outcome.cluster(11).analytical_evaluation
    assert evaluation is not None
    assert evaluation is not source
    np.testing.assert_allclose(evaluation.amplitudes, source.amplitudes)
    assert evaluation.amplitudes.flags.writeable is False
    assert evaluation.model_values.flags.writeable is False
    assert evaluation.statistics == source.statistics
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        evaluation.amplitudes[0, 0] = 0.0
    with pytest.raises(TypeError):
        outcome.cluster(11).optimizer_provenance.metadata["new"] = "value"  # type: ignore[index]


def test_finalization_isolated_from_later_pipeline_mutation() -> None:
    fixture = _completion_fixture({11: (True, True)})
    original_correction = fixture.clusters[0].corrections.copy()
    outcome = finalize_fit(fixture.completion)
    cluster = outcome.cluster(11)
    assert cluster.analytical_evaluation is not None
    original_amplitude = float(cluster.analytical_evaluation.amplitudes[0, 0])
    original_parameter = cluster.final_nonlinear_parameters[0].value

    fixture.completion.results[0].params[cluster.final_nonlinear_parameters[0].name].value += 0.001
    fixture.completion.results[0].metadata["nested"]["cluster_id"] = 999
    fixture.completion.evaluations[0].analytical.amplitudes[0, 0] = 999.0  # type: ignore[union-attr]
    fixture.completion.state.clusters[0].data.fill(999.0)

    assert cluster.final_nonlinear_parameters[0].value == original_parameter
    assert float(cluster.analytical_evaluation.amplitudes[0, 0]) == original_amplitude
    assert cluster.optimizer_provenance.metadata["nested"]["cluster_id"] == 11
    np.testing.assert_array_equal(fixture.clusters[0].corrections, original_correction)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda completion: completion.results.pop(), "Missing.*37"),
        (lambda completion: completion.results.append(completion.results[0]), "Duplicate.*11"),
        (
            lambda completion: setattr(completion.results[0], "cluster_id", 999),
            "Unexpected.*999.*Missing.*11",
        ),
        (
            lambda completion: setattr(completion.results[0], "correction_revision", 1),
            "stale.*11",
        ),
    ],
)
def test_finalization_rejects_invalid_terminal_result_identity_or_revision(
    mutation: object,
    match: str,
) -> None:
    fixture = _completion_fixture({11: (True, True), 37: (True, True)})
    mutation(fixture.completion)  # type: ignore[operator]

    with pytest.raises(ValueError, match=match):
        finalize_fit(fixture.completion)


def test_finalization_rejects_missing_or_mismatched_evaluations_and_noise() -> None:
    fixture = _completion_fixture({11: (True, True)})
    fixture.completion.evaluations.clear()
    with pytest.raises(ValueError, match=r"Missing.*evaluation.*11"):
        finalize_fit(fixture.completion)

    fixture = _completion_fixture({11: (True, True)})
    fixture.completion.results[0].noise = 2.0
    with pytest.raises(ValueError, match=r"noise.*11"):
        finalize_fit(fixture.completion)

    fixture = _completion_fixture({11: (True, True)})
    fixture.completion.state.noise = np.nan
    with pytest.raises(ValueError, match=r"noise.*positive and finite"):
        finalize_fit(fixture.completion)


def test_finalization_rejects_an_empty_cluster_set() -> None:
    fixture = _completion_fixture({11: (True, True)})
    fixture.completion.state.clusters.clear()

    with pytest.raises(ValueError, match="at least one peak cluster"):
        finalize_fit(fixture.completion)


def test_finalization_rejects_terminal_parameters_that_disagree_with_final_merge() -> None:
    fixture = _completion_fixture({11: (True, True)})
    result = fixture.completion.results[0]
    parameter_name = result.params.get_vary_names()[0]
    fixture.completion.state.scalar_params[parameter_name].value += 0.001

    with pytest.raises(ValueError, match=rf"Final nonlinear parameter.*{parameter_name}"):
        finalize_fit(fixture.completion)


def test_finalization_copies_actual_optimizer_provenance_without_synthesis() -> None:
    fixture = _completion_fixture({11: (False, True)})
    result = fixture.completion.results[0]

    outcome = finalize_fit(fixture.completion)
    provenance = outcome.cluster(11).optimizer_provenance

    assert provenance.optimizer_kind == result.optimizer_kind
    assert provenance.converged is result.success
    assert provenance.termination_message == result.message
    assert provenance.function_evaluations == result.nfev
    assert provenance.jacobian_evaluations == result.njev
    assert provenance.final_cost == result.cost
    assert provenance.correction_revision == 2


def test_finalization_leaves_unavailable_optimizer_metadata_absent() -> None:
    fixture = _completion_fixture({11: (True, True)})
    result = fixture.completion.results[0]
    result.optimizer_kind = "basin_hopping"
    result.metadata["global_iterations"] = 3

    provenance = finalize_fit(fixture.completion).cluster(11).optimizer_provenance

    assert provenance.jacobian_evaluations is None
    assert provenance.optimality is None
    assert provenance.iterations == 3
