from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from peakfit.engine.algorithms.evaluation import FitOutcomeClassification, classify_optimizer_result
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import (
    FitConfig,
    OutputConfig,
    PeakFitConfig,
)
from peakfit.engine.domain.fit_steps import FitStep
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.params_vector import FitParameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters
from peakfit.engine.domain.state import FittingState
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.engine.results import FitResult
from peakfit.fit.final_outcome import FinalFitOutcome, finalize_fit
from peakfit.fit.fitting import find_review_clusters, write_fit_run_outputs
from peakfit.fit.output_metadata import RunMetadata
from peakfit.fit.pipeline import (
    CorrectionSnapshot,
    PipelineCompletion,
    PipelineResult,
    run_pipeline,
)
from peakfit.fit.run_models import FitRun
from peakfit.io.readers.results import ResultsLoader
from peakfit.io.schemas import FitSummarySchema
from peakfit.io.writers.config import WriterConfig
from peakfit.io.writers.json import write_final_outcome_summary
from peakfit.io.writers.orchestrator import write_fit_outputs

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class SyntheticFit:
    spectra: Spectra
    peaks: list[Peak]
    clusters: list[Cluster]
    params: Parameters


@dataclass(frozen=True)
class CompetingTruth:
    terminal_results: list[FitResult]
    fit_run: FitRun


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


def _synthetic_fit(cluster_ids: tuple[int, ...] = (11, 37, 91)) -> SyntheticFit:
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
    fit_config = FitConfig(lineshape="gaussian")
    grid_indices = [np.arange(n_points, dtype=np.intp)]

    peaks: list[Peak] = []
    clusters: list[Cluster] = []
    center_points = np.linspace(2, n_points - 3, len(cluster_ids))
    for index, (cluster_id, center_point) in enumerate(
        zip(cluster_ids, center_points, strict=True),
        start=1,
    ):
        center = float(spectra.spectral_params[0].pts2ppm(center_point))
        peak = Peak(
            name=f"P{index}",
            positions=np.array([center], dtype=np.float64),
            shapes=create_shapes(
                spectra,
                fit_config,
                f"P{index}",
                [center],
                ["gaussian"],
            ),
        )
        peak.set_cluster_id(cluster_id)
        peak_params = peak.create_params()
        empty_cluster = Cluster(
            cluster_id=cluster_id,
            peaks=[peak],
            grid_indices=grid_indices,
            data=np.zeros((n_points, n_series), dtype=np.float64),
        )
        shapes = empty_cluster.evaluate(peak_params)
        amplitudes = np.array([[2.0 * index, 3.0 * index]], dtype=np.float64)
        deterministic_residual = 0.01 * index * np.linspace(-1.0, 1.0, n_points, dtype=np.float64)
        data = (shapes.T @ amplitudes) + deterministic_residual[:, np.newaxis]
        clusters.append(
            Cluster(
                cluster_id=cluster_id,
                peaks=[peak],
                grid_indices=grid_indices,
                data=data,
            )
        )
        peaks.append(peak)

    return SyntheticFit(
        spectra=spectra,
        peaks=peaks,
        clusters=clusters,
        params=Parameters.from_peaks(peaks, fixed=False),
    )


def _terminal_result(
    cluster: Cluster,
    *,
    success: bool,
    residual_value: float,
    nfev: int,
    message: str,
    nonfinite_parameter: bool = False,
) -> FitResult:
    params = Parameters.from_peaks(cluster.peaks, fixed=False)
    if nonfinite_parameter:
        params[params.get_vary_names()[0]].__dict__["value"] = np.nan
    residual = np.full(cluster.n_observations, residual_value, dtype=np.float64)
    return FitResult(
        cluster_id=cluster.cluster_id,
        params=params,
        residual=residual,
        cost=float(np.sum(residual**2) / 2.0),
        nfev=nfev,
        njev=max(0, nfev - 1),
        success=success,
        message=message,
        optimality=0.125,
        n_amplitude_params=cluster.n_amplitude_params,
        correction_revision=0,
        optimizer_kind="varpro",
        noise=1.0,
        metadata={
            "peak_names": [peak.name for peak in cluster.peaks],
        },
    )


def _final_outcome_for_json(
    states: dict[int, tuple[bool, bool]],
    *,
    optimizer_kind: str = "varpro",
) -> tuple[FinalFitOutcome, Spectra, list[FitResult]]:
    """Build final outcomes with deterministic classifications for JSON contracts."""
    fit = _synthetic_fit(tuple(states))
    results = []
    for index, (cluster, (success, usable)) in enumerate(
        zip(fit.clusters, states.values(), strict=True),
        start=1,
    ):
        result = _terminal_result(
            cluster,
            success=success,
            residual_value=0.1 * index if usable else np.nan,
            nfev=7 * index,
            message=f"terminal-{cluster.cluster_id}",
            nonfinite_parameter=not usable,
        )
        result.metadata["nested"] = {
            "values": np.array([cluster.cluster_id, index], dtype=np.int64),
        }
        result.optimizer_kind = optimizer_kind
        results.append(result)
    state = FittingState(
        clusters=fit.clusters,
        params=FitParameters.from_parameters(fit.params, fit.peaks),
        scalar_params=fit.params,
        noise=1.0,
    )
    evaluations = [
        classify_optimizer_result(cluster=cluster, result=result, noise=1.0)
        for cluster, result in zip(fit.clusters, results, strict=True)
    ]
    outcome = finalize_fit(
        PipelineCompletion(
            state=state,
            results=results,
            evaluations=evaluations,
            correction_snapshot=CorrectionSnapshot(
                revision=0,
                corrections=MappingProxyType(
                    {cluster.cluster_id: cluster.corrections.copy() for cluster in fit.clusters}
                ),
            ),
            n_optimizer_passes=1,
        )
    )
    return outcome, fit.spectra, results


def _write_final_outcome_json(
    outcome: FinalFitOutcome,
    spectra: Spectra,
    path: Path,
) -> dict[str, object]:
    json_path = write_final_outcome_summary(
        outcome,
        metadata=RunMetadata(),
        z_values=spectra.z_values,
        path=path,
    )
    return json.loads(json_path.read_text(encoding="utf-8"))


@pytest.fixture
def competing_truth(tmp_path: Path) -> CompetingTruth:
    fit = _synthetic_fit()
    terminal_results = [
        _terminal_result(
            fit.clusters[0],
            success=True,
            residual_value=0.25,
            nfev=7,
            message="terminal converged",
        ),
        _terminal_result(
            fit.clusters[1],
            success=False,
            residual_value=0.5,
            nfev=13,
            message="finite but not converged",
        ),
        _terminal_result(
            fit.clusters[2],
            success=False,
            residual_value=np.nan,
            nfev=19,
            message="non-finite terminal result",
            nonfinite_parameter=True,
        ),
    ]
    state = FittingState(
        clusters=fit.clusters,
        params=FitParameters.from_parameters(fit.params, fit.peaks),
        scalar_params=fit.params,
        noise=1.0,
    )
    evaluations = [
        classify_optimizer_result(cluster=cluster, result=terminal, noise=1.0)
        for cluster, terminal in zip(fit.clusters, terminal_results, strict=True)
    ]
    pipeline_result = PipelineCompletion(
        state=state,
        results=terminal_results,
        evaluations=evaluations,
        correction_snapshot=CorrectionSnapshot(
            revision=0,
            corrections=MappingProxyType(
                {cluster.cluster_id: cluster.corrections.copy() for cluster in fit.clusters}
            ),
        ),
        n_optimizer_passes=1,
    )
    outcome = finalize_fit(pipeline_result)
    fit_run = FitRun(
        outcome=outcome,
        continuation_state=state,
        output_dir=tmp_path,
        spectra=fit.spectra,
    )
    return CompetingTruth(
        terminal_results=terminal_results,
        fit_run=fit_run,
    )


def test_cli_review_projects_final_outcomes_with_actual_terminal_provenance(
    competing_truth: CompetingTruth,
) -> None:
    reviews = find_review_clusters(competing_truth.fit_run)

    assert [(review.cluster_id, review.reason) for review in reviews] == [
        ("37", "not_converged"),
        ("91", "unusable"),
    ]
    assert reviews[0].redchi is not None
    assert np.isfinite(reviews[0].redchi)
    assert reviews[0].termination_message == "finite but not converged"
    assert reviews[1].redchi is None
    assert reviews[1].unusable_reason is not None


def test_run_summary_uses_final_classifications_and_usable_distributions(
    competing_truth: CompetingTruth,
) -> None:
    summary = competing_truth.fit_run.summary
    assert summary.n_clusters == 3
    assert summary.n_converged == 1
    assert summary.n_usable_non_converged == 1
    assert summary.n_unusable == 1
    assert summary.redchi_population_size == 2
    assert summary.success_rate == pytest.approx(1.0 / 3.0)
    assert summary.mean_redchi is not None
    assert summary.median_redchi is not None


def test_future_contract_cli_json_and_markdown_share_terminal_failures(
    competing_truth: CompetingTruth,
    tmp_path: Path,
) -> None:
    assert competing_truth.fit_run.spectra is not None
    review_ids = {
        int(review.cluster_id) for review in find_review_clusters(competing_truth.fit_run)
    }
    json_path = write_final_outcome_summary(
        competing_truth.fit_run.outcome,
        metadata=RunMetadata(),
        z_values=competing_truth.fit_run.spectra.z_values,
        path=tmp_path / "fit.json",
    )
    written = write_fit_outputs(
        competing_truth.fit_run.outcome,
        tmp_path,
        WriterConfig(formats=("csv", "txt")),
        summary=competing_truth.fit_run.summary,
    )
    report_path = written["report"]
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")
    with written["clusters"].open(newline="", encoding="utf-8") as handle:
        table = {int(row["cluster_id"]): row for row in csv.DictReader(handle)}
    clusters_by_id = {int(cluster["cluster_id"]): cluster for cluster in payload["clusters"]}
    persisted_failure_ids = {
        cluster_id
        for cluster_id, cluster in clusters_by_id.items()
        if cluster["classification"] != "converged"
    }

    assert persisted_failure_ids == review_ids == {37, 91}
    assert "finite but not converged" in report
    assert "non-finite terminal result" in report
    assert table[37]["classification"] == "usable_non_converged"
    assert table[37]["function_evaluations"] == "13"
    assert table[37]["termination_message"] == "finite but not converged"
    assert table[91]["classification"] == "unusable"
    assert table[91]["chi_squared"] == ""


def test_completed_writer_path_ignores_mutated_continuation_state(
    competing_truth: CompetingTruth,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("peakfit.fit.fitting.save_state", lambda *_args, **_kwargs: None)
    config = PeakFitConfig(
        output=OutputConfig(formats=["json", "csv", "txt"], save_simulated=False),
    )

    assert competing_truth.fit_run.spectra is not None
    competing_truth.fit_run.continuation_state.clusters[0].data.fill(np.nan)
    competing_truth.fit_run.continuation_state.noise = np.nan
    write_fit_run_outputs(
        competing_truth.fit_run,
        competing_truth.fit_run.spectra,
        config,
        input_paths={},
    )

    output_dir = competing_truth.fit_run.output_dir
    payload = json.loads((output_dir / "summary" / "fit.json").read_text(encoding="utf-8"))
    report = (output_dir / "summary" / "report.md").read_text(encoding="utf-8")
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    with (output_dir / "tables" / "clusters.csv").open(newline="", encoding="utf-8") as handle:
        rows = {int(row["cluster_id"]): row for row in csv.DictReader(handle)}

    expected = competing_truth.fit_run.outcome.cluster(11)
    evaluation = expected.analytical_evaluation
    assert evaluation is not None
    json_cluster = next(cluster for cluster in payload["clusters"] if cluster["cluster_id"] == 11)
    assert json_cluster["analytical_evaluation"]["amplitudes"] == evaluation.amplitudes.tolist()
    assert json_cluster["optimizer_provenance"]["function_evaluations"] == 7
    assert rows[11]["chi_squared"] == f"{evaluation.statistics.chi_squared:.6f}"
    assert "Usable, not converged: 1" in report
    assert "**Unusable clusters**: 1" in readme


def test_future_contract_persistence_uses_terminal_classification_and_provenance(
    competing_truth: CompetingTruth,
    tmp_path: Path,
) -> None:
    assert competing_truth.fit_run.spectra is not None
    terminal = competing_truth.terminal_results
    json_path = write_final_outcome_summary(
        competing_truth.fit_run.outcome,
        metadata=RunMetadata(),
        z_values=competing_truth.fit_run.spectra.z_values,
        path=tmp_path / "fit.json",
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    clusters_by_id = {int(cluster["cluster_id"]): cluster for cluster in payload["clusters"]}

    assert payload["schema_version"] == "4.0.0"
    assert "statistics" in payload
    assert "global_statistics" not in payload
    assert clusters_by_id[11]["classification"] == "converged"
    assert clusters_by_id[37]["classification"] == "usable_non_converged"
    assert clusters_by_id[91]["classification"] == "unusable"
    assert clusters_by_id[11]["analytical_evaluation"] is not None
    assert clusters_by_id[37]["analytical_evaluation"] is not None
    assert clusters_by_id[91]["analytical_evaluation"] is None
    terminal_by_id = {result.cluster_id: result for result in terminal}
    for cluster_id, result in terminal_by_id.items():
        provenance = clusters_by_id[cluster_id]["optimizer_provenance"]
        assert provenance["function_evaluations"] == result.nfev
        assert provenance["termination_message"] == result.message


def test_json_projects_all_converged_outcomes_in_final_identity_order(tmp_path: Path) -> None:
    outcome, spectra, _ = _final_outcome_for_json(
        {91: (True, True), 11: (True, True), 37: (True, True)}
    )

    payload = _write_final_outcome_json(outcome, spectra, tmp_path / "fit.json")

    assert payload["schema_version"] == "4.0.0"
    assert [cluster["cluster_id"] for cluster in payload["clusters"]] == [11, 37, 91]  # type: ignore[index]
    assert all(cluster["classification"] == "converged" for cluster in payload["clusters"])  # type: ignore[index]
    assert all(cluster["analytical_evaluation"] is not None for cluster in payload["clusters"])  # type: ignore[index]


def test_json_projects_all_unusable_outcomes_without_numerical_placeholders(tmp_path: Path) -> None:
    outcome, spectra, results = _final_outcome_for_json({11: (False, False), 37: (True, False)})

    payload = _write_final_outcome_json(outcome, spectra, tmp_path / "fit.json")

    for cluster, result in zip(payload["clusters"], results, strict=True):  # type: ignore[arg-type]
        assert cluster["classification"] == "unusable"
        assert cluster["unusable_reason"]
        assert cluster["final_nonlinear_parameters"] == []
        assert cluster["analytical_evaluation"] is None
        assert cluster["optimizer_provenance"]["success"] is result.success
    assert payload["statistics"]["n_observations"] == 0  # type: ignore[index]


def test_json_copies_frozen_analytical_values_and_jsonifies_actual_provenance(
    tmp_path: Path,
) -> None:
    outcome, spectra, _ = _final_outcome_for_json({11: (True, True)})

    payload = _write_final_outcome_json(outcome, spectra, tmp_path / "fit.json")
    cluster = payload["clusters"][0]  # type: ignore[index]
    evaluation = outcome.cluster(11).analytical_evaluation

    assert evaluation is not None
    assert cluster["analytical_evaluation"]["amplitudes"] == evaluation.amplitudes.tolist()
    assert cluster["analytical_evaluation"]["model_values"] == evaluation.model_values.tolist()
    assert cluster["analytical_evaluation"]["raw_residuals"] == evaluation.raw_residuals.tolist()
    assert (
        cluster["analytical_evaluation"]["normalized_residuals"]
        == evaluation.normalized_residuals.tolist()
    )
    assert cluster["optimizer_provenance"]["metadata"]["nested"]["values"] == [11, 1]


def test_json_leaves_unavailable_optimizer_diagnostics_absent(tmp_path: Path) -> None:
    outcome, spectra, _ = _final_outcome_for_json(
        {11: (True, True)}, optimizer_kind="basin_hopping"
    )

    payload = _write_final_outcome_json(outcome, spectra, tmp_path / "fit.json")
    provenance = payload["clusters"][0]["optimizer_provenance"]  # type: ignore[index]

    assert provenance["optimizer_kind"] == "basin_hopping"
    assert "jacobian_evaluations" not in provenance
    assert "optimality" not in provenance
    assert "iterations" not in provenance


def test_json_reader_round_trips_classifications_and_distinguishes_absent_evaluation(
    tmp_path: Path,
) -> None:
    outcome, spectra, _ = _final_outcome_for_json(
        {91: (False, False), 11: (True, True), 37: (False, True)}
    )
    summary_path = tmp_path / "summary" / "fit.json"
    _write_final_outcome_json(outcome, spectra, summary_path)

    summary = ResultsLoader(tmp_path).load_summary()

    assert [cluster.cluster_id for cluster in summary.clusters] == [11, 37, 91]
    assert [cluster.classification for cluster in summary.clusters] == [
        "converged",
        "usable_non_converged",
        "unusable",
    ]
    assert summary.clusters[-1].analytical_evaluation is None


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload["clusters"][1].update(cluster_id=11), "unique"),
        (lambda payload: payload["clusters"][0].pop("cluster_id"), "cluster_id"),
        (lambda payload: payload["clusters"][0].update(cluster_id="11"), "integer"),
        (lambda payload: payload["clusters"].reverse(), "ascending cluster_id order"),
        (
            lambda payload: payload["clusters"][2].update(
                analytical_evaluation=payload["clusters"][0]["analytical_evaluation"]
            ),
            "unusable outcomes must not contain an analytical evaluation",
        ),
        (lambda payload: payload.update(schema_version="3.0.0"), "3.0.0.*4.0.0"),
    ],
)
def test_json_schema_rejects_malformed_identity_order_and_outcome_combinations(
    tmp_path: Path,
    mutation: Any,
    match: str,
) -> None:
    outcome, spectra, _ = _final_outcome_for_json(
        {11: (True, True), 37: (False, True), 91: (False, False)}
    )
    payload = _write_final_outcome_json(outcome, spectra, tmp_path / "fit.json")
    mutation(payload)

    with pytest.raises(ValueError, match=match):
        FitSummarySchema.model_validate(payload)


def test_future_contract_run_summary_reports_all_classifications(
    competing_truth: CompetingTruth,
) -> None:
    summary = competing_truth.fit_run.summary

    assert summary.n_converged == 1
    assert getattr(summary, "n_usable_non_converged", None) == 1
    assert getattr(summary, "n_unusable", None) == 1
    assert getattr(summary, "redchi_population_size", None) == 2


def _run_with_executor(
    fit: SyntheticFit,
    config: FitConfig,
    *,
    reverse: bool = False,
) -> tuple[PipelineResult, list[int], list[dict[int, np.ndarray]]]:
    completion_order: list[int] = []
    correction_snapshots: list[dict[int, np.ndarray]] = []

    def executor(function: Any, tasks: list[Any]) -> list[Any]:
        correction_snapshots.append(
            {task.cluster_id: task.cluster.corrections.copy() for task in tasks}
        )
        ordered_tasks = list(reversed(tasks)) if reverse else tasks
        completed = [function(task) for task in ordered_tasks]
        completion_order.extend(result.cluster_id for result in completed)
        return completed

    result = run_pipeline(
        config=config,
        clusters=fit.clusters,
        data_noise=1.0,
        base_params=fit.params,
        peaks=fit.peaks,
        executor=executor,
    )
    return result, completion_order, correction_snapshots


def test_unordered_execution_associates_nonconsecutive_cluster_ids_by_identity() -> None:
    ordered_fit = _synthetic_fit(cluster_ids=(91, 11, 37))
    ordered_result, ordered_completion, _ordered_snapshots = _run_with_executor(
        ordered_fit,
        FitConfig(lineshape="gaussian", refine_iterations=1, max_iterations=25),
    )
    reversed_fit = _synthetic_fit(cluster_ids=(91, 11, 37))
    reversed_result, reversed_completion, _reversed_snapshots = _run_with_executor(
        reversed_fit,
        FitConfig(lineshape="gaussian", refine_iterations=1, max_iterations=25),
        reverse=True,
    )

    input_ids = [cluster.cluster_id for cluster in ordered_fit.clusters]
    expected_associations = sorted(
        (cluster.cluster_id, [peak.name for peak in cluster.peaks])
        for cluster in ordered_fit.clusters
    )
    ordered_associations = [
        (cluster.cluster_id, list(cluster.peak_names))
        for cluster in ordered_result.final_outcome.clusters
    ]
    reversed_associations = [
        (cluster.cluster_id, list(cluster.peak_names))
        for cluster in reversed_result.final_outcome.clusters
    ]

    assert ordered_completion == input_ids
    assert reversed_completion == list(reversed(input_ids))
    assert ordered_associations == reversed_associations == expected_associations


def test_cluster_identity_rejects_duplicate_input_identifiers() -> None:
    fit = _synthetic_fit(cluster_ids=(11, 37, 37))

    def executor(_function: Any, _tasks: list[Any]) -> list[Any]:
        pytest.fail("duplicate cluster identities must fail before task submission")

    with pytest.raises(ValueError, match=r"Duplicate cluster_id.*37"):
        run_pipeline(
            config=FitConfig(lineshape="gaussian", refine_iterations=1),
            clusters=fit.clusters,
            data_noise=1.0,
            base_params=fit.params,
            peaks=fit.peaks,
            executor=executor,
        )


def test_cluster_identity_rejects_missing_optimizer_results() -> None:
    fit = _synthetic_fit()

    def executor(function: Any, tasks: list[Any]) -> list[Any]:
        return [function(task) for task in tasks if task.cluster_id != 37]

    with pytest.raises(ValueError, match=r"Missing optimizer result cluster_id values: \[37\]"):
        run_pipeline(
            config=FitConfig(lineshape="gaussian", refine_iterations=1),
            clusters=fit.clusters,
            data_noise=1.0,
            base_params=fit.params,
            peaks=fit.peaks,
            executor=executor,
        )


def test_cluster_identity_rejects_duplicate_optimizer_results() -> None:
    fit = _synthetic_fit()

    def executor(function: Any, tasks: list[Any]) -> list[Any]:
        first_result = function(tasks[0])
        return [first_result, first_result, *(function(task) for task in tasks[1:])]

    with pytest.raises(
        ValueError,
        match=r"Duplicate optimizer result cluster_id values: \[11\]",
    ):
        run_pipeline(
            config=FitConfig(lineshape="gaussian", refine_iterations=1),
            clusters=fit.clusters,
            data_noise=1.0,
            base_params=fit.params,
            peaks=fit.peaks,
            executor=executor,
        )


def test_cluster_identity_rejects_unexpected_optimizer_results() -> None:
    fit = _synthetic_fit()

    def executor(function: Any, tasks: list[Any]) -> list[Any]:
        completed = [function(task) for task in tasks]
        completed[1].cluster_id = 73
        return completed

    with pytest.raises(
        ValueError,
        match=r"Unexpected optimizer result cluster_id values: \[73\]",
    ):
        run_pipeline(
            config=FitConfig(lineshape="gaussian", refine_iterations=1),
            clusters=fit.clusters,
            data_noise=1.0,
            base_params=fit.params,
            peaks=fit.peaks,
            executor=executor,
        )


def test_cluster_identity_preserves_returned_nonconverged_result() -> None:
    fit = _synthetic_fit(cluster_ids=(37,))

    def executor(_function: Any, tasks: list[Any]) -> list[FitResult]:
        return [
            _terminal_result(
                tasks[0].cluster,
                success=False,
                residual_value=0.5,
                nfev=13,
                message="finite but not converged",
            )
        ]

    result = run_pipeline(
        config=FitConfig(lineshape="gaussian", refine_iterations=1),
        clusters=fit.clusters,
        data_noise=1.0,
        base_params=fit.params,
        peaks=fit.peaks,
        executor=executor,
    )

    outcome = result.final_outcome.cluster(37)
    assert outcome.classification is FitOutcomeClassification.USABLE_NON_CONVERGED
    assert outcome.optimizer_provenance.converged is False
    assert outcome.optimizer_provenance.termination_message == "finite but not converged"


def test_cluster_identity_optimizer_exception_aborts_run() -> None:
    fit = _synthetic_fit(cluster_ids=(37,))

    def executor(_function: Any, _tasks: list[Any]) -> list[Any]:
        raise RuntimeError("optimizer failed for cluster_id 37")

    with pytest.raises(RuntimeError, match=r"optimizer failed for cluster_id 37"):
        run_pipeline(
            config=FitConfig(lineshape="gaussian", refine_iterations=1),
            clusters=fit.clusters,
            data_noise=1.0,
            base_params=fit.params,
            peaks=fit.peaks,
            executor=executor,
        )


def test_terminal_pass_uses_the_frozen_final_correction() -> None:
    fit = _synthetic_fit(cluster_ids=(11, 37))
    result, _completion_order, snapshots = _run_with_executor(
        fit,
        FitConfig(lineshape="gaussian", refine_iterations=1, max_iterations=25),
    )

    terminal_snapshot = snapshots[-1]
    final_corrections = {
        cluster.cluster_id: cluster.corrections for cluster in result.continuation_state.clusters
    }

    assert len(snapshots) == 1
    assert result.final_outcome.n_optimizer_passes == 1
    assert result.final_outcome.n_correction_updates == 0
    assert all(
        np.array_equal(final_corrections[cluster_id], terminal_snapshot[cluster_id])
        for cluster_id in terminal_snapshot
    )
    assert result.final_outcome.terminal_correction_revision == 0
    assert [cluster.correction_revision for cluster in result.final_outcome.clusters] == [0, 0]


def test_numerical_usability_controls_parameter_merging_and_corrections() -> None:
    def run_with_unusable_parameter(
        relative_value: float,
    ) -> tuple[
        PipelineResult,
        str,
        float,
        float,
    ]:
        fit = _synthetic_fit()
        unusable_parameter: list[tuple[str, float, float]] = []

        def executor(_function: Any, tasks: list[Any]) -> list[FitResult]:
            completed: list[FitResult] = []
            residual_values = (0.25, 0.5, np.nan)
            for task_index, (task, residual_value) in enumerate(
                zip(tasks, residual_values, strict=True)
            ):
                result = _terminal_result(
                    task.cluster,
                    success=task_index == 0,
                    residual_value=residual_value,
                    nfev=task_index + 1,
                    message=f"result {task_index}",
                )
                if task_index == 2:
                    parameter_name = result.params.get_vary_names()[0]
                    parameter = result.params[parameter_name]
                    original_value = fit.params[parameter_name].value
                    rejected_value = parameter.min + relative_value * (
                        parameter.max - parameter.min
                    )
                    result.params[parameter_name].value = rejected_value
                    unusable_parameter.append((parameter_name, original_value, rejected_value))
                completed.append(result)
            return completed

        result = run_pipeline(
            config=FitConfig(lineshape="gaussian", refine_iterations=1),
            clusters=fit.clusters,
            data_noise=1.0,
            base_params=fit.params,
            peaks=fit.peaks,
            executor=executor,
        )
        parameter_name, original_value, rejected_value = unusable_parameter[0]
        return result, parameter_name, original_value, rejected_value

    result, parameter_name, original_value, rejected_value = run_with_unusable_parameter(0.25)
    alternate, _, _, _ = run_with_unusable_parameter(0.75)

    assert [cluster.classification for cluster in result.final_outcome.clusters] == [
        FitOutcomeClassification.CONVERGED,
        FitOutcomeClassification.USABLE_NON_CONVERGED,
        FitOutcomeClassification.UNUSABLE,
    ]
    assert result.final_outcome.cluster(91).unusable_reason == "non-finite optimizer residuals"
    assert result.continuation_state.scalar_params[parameter_name].value == pytest.approx(
        original_value
    )
    assert result.continuation_state.scalar_params[parameter_name].value != pytest.approx(
        rejected_value
    )
    for cluster, alternate_cluster in zip(
        result.continuation_state.clusters,
        alternate.continuation_state.clusters,
        strict=True,
    ):
        np.testing.assert_allclose(
            cluster.corrections,
            alternate_cluster.corrections,
        )


@pytest.mark.parametrize(
    ("refine_iterations", "expected_passes", "expected_updates"),
    [(1, 1, 0), (2, 2, 1), (4, 4, 3)],
)
def test_refine_iterations_is_the_exact_pass_count(
    refine_iterations: int,
    expected_passes: int,
    expected_updates: int,
) -> None:
    fit = _synthetic_fit(cluster_ids=(11,))
    _result, _completion_order, snapshots = _run_with_executor(
        fit,
        FitConfig(
            lineshape="gaussian",
            refine_iterations=refine_iterations,
            max_iterations=25,
        ),
    )

    assert len(snapshots) == expected_passes
    assert _result.final_outcome.n_optimizer_passes == expected_passes
    assert _result.final_outcome.n_correction_updates == expected_updates


def test_current_explicit_steps_use_their_configured_pass_counts() -> None:
    fit = _synthetic_fit(cluster_ids=(11,))
    config = FitConfig(
        lineshape="gaussian",
        max_iterations=25,
        steps=[
            FitStep(name="first", iterations=1),
            FitStep(name="second", iterations=2),
        ],
    )

    _result, _completion_order, snapshots = _run_with_executor(fit, config)

    assert len(snapshots) == 3
    assert _result.final_outcome.n_optimizer_passes == 3
    assert _result.final_outcome.n_correction_updates == 2


def test_refine_iterations_must_be_positive() -> None:
    with pytest.raises(ValueError, match=r"refine_iterations.*at least 1"):
        FitConfig(lineshape="gaussian", refine_iterations=0)


def test_correction_snapshots_are_isolated_from_later_mutation() -> None:
    fit = _synthetic_fit(cluster_ids=(11, 37))
    task_snapshots: list[dict[int, np.ndarray]] = []
    task_records: list[list[Any]] = []

    def executor(function: Any, tasks: list[Any]) -> list[Any]:
        task_snapshots.append({task.cluster_id: task.cluster.corrections.copy() for task in tasks})
        task_records.append(tasks)
        return [function(task) for task in tasks]

    result = run_pipeline(
        config=FitConfig(lineshape="gaussian", refine_iterations=2, max_iterations=25),
        clusters=fit.clusters,
        data_noise=1.0,
        base_params=fit.params,
        peaks=fit.peaks,
        executor=executor,
    )

    final_corrections = {
        cluster.cluster_id: cluster.corrections.copy()
        for cluster in result.continuation_state.clusters
    }
    fit.clusters[0].corrections.fill(123.0)

    assert len(task_snapshots) == 2
    assert [task.correction_revision for task in task_records[0]] == [0, 0]
    assert [task.correction_revision for task in task_records[1]] == [1, 1]
    for task in task_records[0]:
        cluster_id = task.cluster_id
        task_cluster = task.cluster
        np.testing.assert_array_equal(task_cluster.corrections, task_snapshots[0][cluster_id])
        assert not task_cluster.corrections.flags.writeable
    assert any(
        not np.array_equal(task_snapshots[1][cluster_id], task_snapshots[0][cluster_id])
        for cluster_id in final_corrections
    )
    for cluster_id, correction in final_corrections.items():
        np.testing.assert_array_equal(correction, task_snapshots[1][cluster_id])
    assert result.final_outcome.terminal_correction_revision == 1
    assert [cluster.correction_revision for cluster in result.final_outcome.clusters] == [1, 1]


def test_pipeline_rejects_a_stale_result_correction_revision() -> None:
    fit = _synthetic_fit(cluster_ids=(11,))

    def executor(function: Any, tasks: list[Any]) -> list[Any]:
        result = function(tasks[0])
        result.correction_revision = 73
        return [result]

    with pytest.raises(ValueError, match=r"correction_revision.*expected 0, got 73"):
        run_pipeline(
            config=FitConfig(lineshape="gaussian", refine_iterations=1, max_iterations=25),
            clusters=fit.clusters,
            data_noise=1.0,
            base_params=fit.params,
            peaks=fit.peaks,
            executor=executor,
        )


def test_usable_nonconverged_results_contribute_to_the_next_correction() -> None:
    def next_correction_for_nonconverged_result(usable: bool) -> np.ndarray:
        fit = _synthetic_fit(cluster_ids=(11, 37))
        next_pass_corrections: list[np.ndarray] = []

        def executor(function: Any, tasks: list[Any]) -> list[Any]:
            if tasks[0].correction_revision == 1:
                next_pass_corrections.append(tasks[1].cluster.corrections.copy())
            results = [function(task) for task in tasks]
            if tasks[0].correction_revision == 0:
                for task, result in zip(tasks, results, strict=True):
                    result.success = False
                    if task.cluster_id == 37 or not usable:
                        result.residual[:] = np.nan
            return results

        run_pipeline(
            config=FitConfig(lineshape="gaussian", refine_iterations=2, max_iterations=25),
            clusters=fit.clusters,
            data_noise=1.0,
            base_params=fit.params,
            peaks=fit.peaks,
            executor=executor,
        )
        return next_pass_corrections[0]

    correction_with_usable_result = next_correction_for_nonconverged_result(usable=True)
    correction_with_only_unusable_results = next_correction_for_nonconverged_result(usable=False)

    assert not np.allclose(correction_with_usable_result, 0.0)
    np.testing.assert_array_equal(correction_with_only_unusable_results, 0.0)
