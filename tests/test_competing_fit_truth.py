from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from peakfit.engine.algorithms.common import calculate_shape_heights, residuals
from peakfit.engine.algorithms.evaluation import FitOutcomeClassification
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import BasinHoppingConfig, FitConfig, VarProConfig
from peakfit.engine.domain.fit_steps import FitStep
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.params_vector import FitParameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters
from peakfit.engine.domain.state import FittingState
from peakfit.engine.fitting.optimizers import fit_with_optimizer
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.engine.results import FitResult
from peakfit.fit.fitting import _build_summary, find_review_clusters
from peakfit.fit.pipeline import PipelineResult, run_pipeline
from peakfit.fit.results import build_fit_results
from peakfit.fit.run_models import FitRun, LoadedData
from peakfit.io.writers.json import write_summary
from peakfit.io.writers.markdown import write_report

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.fit.result_models import FitResults


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
    reconstructed: FitResults


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
) -> FitResult:
    params = Parameters.from_peaks(cluster.peaks, fixed=False)
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
        metadata={
            "peak_names": [peak.name for peak in cluster.peaks],
        },
    )


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
        ),
    ]
    state = FittingState(
        clusters=fit.clusters,
        params=FitParameters.from_parameters(fit.params, fit.peaks),
        scalar_params=fit.params,
        noise=1.0,
    )
    pipeline_result = PipelineResult(state=state, results=terminal_results)
    loaded = LoadedData(
        spectra=fit.spectra,
        peaks=fit.peaks,
        noise=1.0,
        noise_source="synthetic",
        shape_names=["gaussian"],
        contour_level=1.0,
        clusters=fit.clusters,
    )
    summary = _build_summary(loaded, pipeline_result)
    fit_run = FitRun(
        state=state,
        results=terminal_results,
        output_dir=tmp_path,
        success=False,
        summary=summary,
        spectra=fit.spectra,
    )
    reconstructed = build_fit_results(
        state=state,
        spectra=fit.spectra,
        config={},
        input_files={},
    )
    return CompetingTruth(
        terminal_results=terminal_results,
        fit_run=fit_run,
        reconstructed=reconstructed,
    )


def test_current_cli_review_uses_terminal_optimizer_results(
    competing_truth: CompetingTruth,
) -> None:
    reviews = find_review_clusters(competing_truth.fit_run)

    assert [(review.cluster_id, review.reason) for review in reviews] == [
        ("37", "diverged"),
        ("91", "diverged"),
    ]
    assert np.isfinite(reviews[0].redchi)
    assert np.isnan(reviews[1].redchi)


def test_current_run_summary_uses_success_and_excludes_every_failed_result(
    competing_truth: CompetingTruth,
) -> None:
    summary = competing_truth.fit_run.summary
    converged_result = competing_truth.terminal_results[0]

    assert summary.n_clusters == 3
    assert summary.n_converged == 1
    assert summary.success_rate == pytest.approx(1.0 / 3.0)
    assert summary.mean_redchi == pytest.approx(converged_result.redchi)
    assert summary.median_redchi == pytest.approx(converged_result.redchi)
    assert not hasattr(summary, "n_usable_non_converged")
    assert not hasattr(summary, "n_unusable")
    assert not hasattr(summary, "redchi_population_size")


def test_current_reconstruction_synthesizes_convergence_and_provenance(
    competing_truth: CompetingTruth,
) -> None:
    terminal = competing_truth.terminal_results
    persisted = competing_truth.reconstructed.statistics

    assert [result.success for result in terminal] == [True, False, False]
    assert [np.isfinite(result.residual).all() for result in terminal] == [
        np.True_,
        np.True_,
        np.False_,
    ]
    assert [result.nfev for result in terminal] == [7, 13, 19]
    assert [statistics.fit_converged for statistics in persisted] == [True, True, True]
    assert [statistics.n_function_evals for statistics in persisted] == [0, 0, 0]
    assert [statistics.fit_message for statistics in persisted] == [
        "Statistics computed from fitted model",
        "Statistics computed from fitted model",
        "Statistics computed from fitted model",
    ]
    assert persisted[0].reduced_chi_squared != pytest.approx(terminal[0].redchi)


def test_current_json_and_markdown_hide_terminal_failure_provenance(
    competing_truth: CompetingTruth,
    tmp_path: Path,
) -> None:
    json_path = write_summary(competing_truth.reconstructed, tmp_path / "fit.json")
    report_path = write_report(competing_truth.reconstructed, tmp_path / "report.md")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")

    assert [statistics["fit_converged"] for statistics in payload["statistics"]] == [
        True,
        True,
        True,
    ]
    assert all("n_function_evals" not in statistics for statistics in payload["statistics"])
    assert all("fit_message" not in statistics for statistics in payload["statistics"])
    assert "did not converge" not in report
    assert "finite but not converged" not in report
    assert "non-finite terminal result" not in report


@pytest.mark.xfail(
    strict=True,
    reason="Tickets 06 through 08 will make every final consumer use one outcome.",
)
def test_future_contract_cli_json_and_markdown_share_terminal_failures(
    competing_truth: CompetingTruth,
    tmp_path: Path,
) -> None:
    review_ids = {
        int(review.cluster_id) for review in find_review_clusters(competing_truth.fit_run)
    }
    json_path = write_summary(competing_truth.reconstructed, tmp_path / "fit.json")
    report_path = write_report(competing_truth.reconstructed, tmp_path / "report.md")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")
    clusters_by_id = {int(cluster["cluster_id"]): cluster for cluster in payload["clusters"]}
    persisted_failure_ids = {
        cluster_id
        for cluster_id, cluster in clusters_by_id.items()
        if cluster["classification"] != "converged"
    }

    assert persisted_failure_ids == review_ids == {37, 91}
    assert "finite but not converged" in report
    assert "non-finite terminal result" in report


@pytest.mark.xfail(
    strict=True,
    reason="Tickets 05 and 07 will project terminal classification and provenance.",
)
def test_future_contract_persistence_uses_terminal_classification_and_provenance(
    competing_truth: CompetingTruth,
    tmp_path: Path,
) -> None:
    terminal = competing_truth.terminal_results
    json_path = write_summary(competing_truth.reconstructed, tmp_path / "fit.json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    clusters_by_id = {int(cluster["cluster_id"]): cluster for cluster in payload["clusters"]}

    assert "statistics" not in payload
    assert clusters_by_id[11]["classification"] == "converged"
    assert clusters_by_id[37]["classification"] == "usable_non_converged"
    assert clusters_by_id[91]["classification"] == "unusable"
    assert clusters_by_id[11]["statistics"] is not None
    assert clusters_by_id[37]["statistics"] is not None
    assert clusters_by_id[91].get("statistics") is None
    terminal_by_id = {result.cluster_id: result for result in terminal}
    for cluster_id, result in terminal_by_id.items():
        provenance = clusters_by_id[cluster_id]["optimizer_provenance"]
        assert provenance["n_function_evals"] == result.nfev
        assert provenance["termination_message"] == result.message


@pytest.mark.xfail(
    strict=True,
    reason="Ticket 06 will report all outcome classes and distribution populations.",
)
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
        FitConfig(lineshape="gaussian", refine_iterations=0, max_iterations=25),
    )
    reversed_fit = _synthetic_fit(cluster_ids=(91, 11, 37))
    reversed_result, reversed_completion, _reversed_snapshots = _run_with_executor(
        reversed_fit,
        FitConfig(lineshape="gaussian", refine_iterations=0, max_iterations=25),
        reverse=True,
    )

    input_ids = [cluster.cluster_id for cluster in ordered_fit.clusters]
    expected_associations = sorted(
        (cluster.cluster_id, [peak.name for peak in cluster.peaks])
        for cluster in ordered_fit.clusters
    )
    ordered_associations = [
        (fit_result.cluster_id, fit_result.metadata["peak_names"])
        for fit_result in ordered_result.results
    ]
    reversed_associations = [
        (fit_result.cluster_id, fit_result.metadata["peak_names"])
        for fit_result in reversed_result.results
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
            config=FitConfig(lineshape="gaussian", refine_iterations=0),
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
            config=FitConfig(lineshape="gaussian", refine_iterations=0),
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
            config=FitConfig(lineshape="gaussian", refine_iterations=0),
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
            config=FitConfig(lineshape="gaussian", refine_iterations=0),
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
        config=FitConfig(lineshape="gaussian", refine_iterations=0),
        clusters=fit.clusters,
        data_noise=1.0,
        base_params=fit.params,
        peaks=fit.peaks,
        executor=executor,
    )

    assert [(item.cluster_id, item.success, item.message) for item in result.results] == [
        (37, False, "finite but not converged")
    ]


def test_cluster_identity_optimizer_exception_aborts_run() -> None:
    fit = _synthetic_fit(cluster_ids=(37,))

    def executor(_function: Any, _tasks: list[Any]) -> list[Any]:
        raise RuntimeError("optimizer failed for cluster_id 37")

    with pytest.raises(RuntimeError, match=r"optimizer failed for cluster_id 37"):
        run_pipeline(
            config=FitConfig(lineshape="gaussian", refine_iterations=0),
            clusters=fit.clusters,
            data_noise=1.0,
            base_params=fit.params,
            peaks=fit.peaks,
            executor=executor,
        )


def test_current_terminal_pass_is_followed_by_a_correction_update() -> None:
    fit = _synthetic_fit(cluster_ids=(11, 37))
    result, _completion_order, snapshots = _run_with_executor(
        fit,
        FitConfig(lineshape="gaussian", refine_iterations=0, max_iterations=25),
    )

    terminal_snapshot = snapshots[-1]
    final_corrections = {
        cluster.cluster_id: cluster.corrections for cluster in result.state.clusters
    }

    assert all(np.allclose(correction, 0.0) for correction in terminal_snapshot.values())
    assert any(
        not np.allclose(final_corrections[cluster_id], terminal_snapshot[cluster_id])
        for cluster_id in terminal_snapshot
    )
    reconstructed = build_fit_results(
        result.state,
        fit.spectra,
        config={},
        input_files={},
    )
    assert any(
        statistics.reduced_chi_squared != pytest.approx(terminal.redchi)
        for statistics, terminal in zip(
            reconstructed.statistics,
            result.results,
            strict=True,
        )
    )


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
            config=FitConfig(lineshape="gaussian", refine_iterations=0),
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

    assert [terminal.success for terminal in result.results] == [True, False, False]
    assert [np.isfinite(terminal.residual).all() for terminal in result.results] == [
        np.True_,
        np.True_,
        np.False_,
    ]
    assert [evaluation.classification for evaluation in result.evaluations] == [
        FitOutcomeClassification.CONVERGED,
        FitOutcomeClassification.USABLE_NON_CONVERGED,
        FitOutcomeClassification.UNUSABLE,
    ]
    assert result.evaluations[-1].unusable_reason == "non-finite optimizer residuals"
    assert result.state.scalar_params[parameter_name].value == pytest.approx(original_value)
    assert result.state.scalar_params[parameter_name].value != pytest.approx(rejected_value)
    for cluster, alternate_cluster in zip(
        result.state.clusters,
        alternate.state.clusters,
        strict=True,
    ):
        np.testing.assert_allclose(
            cluster.corrections,
            alternate_cluster.corrections,
        )


@pytest.mark.xfail(
    strict=True,
    reason="Ticket 04 will prohibit correction updates after the terminal pass.",
)
def test_future_contract_terminal_pass_uses_the_frozen_final_correction() -> None:
    fit = _synthetic_fit(cluster_ids=(11, 37))
    result, _completion_order, snapshots = _run_with_executor(
        fit,
        FitConfig(lineshape="gaussian", refine_iterations=1, max_iterations=25),
    )

    terminal_snapshot = snapshots[-1]
    final_corrections = {
        cluster.cluster_id: cluster.corrections for cluster in result.state.clusters
    }

    assert all(
        np.array_equal(final_corrections[cluster_id], terminal_snapshot[cluster_id])
        for cluster_id in terminal_snapshot
    )


@pytest.mark.parametrize(
    ("refine_iterations", "expected_current_passes"),
    [(0, 1), (1, 2), (2, 3)],
)
def test_current_default_schedule_adds_one_optimizer_pass(
    refine_iterations: int,
    expected_current_passes: int,
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

    assert len(snapshots) == expected_current_passes


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


@pytest.mark.xfail(
    strict=True,
    reason="Ticket 04 will define refine_iterations as the exact optimizer-pass count.",
)
def test_future_contract_refine_iterations_must_be_positive() -> None:
    with pytest.raises(ValueError, match=r"refine_iterations.*at least 1"):
        FitConfig(lineshape="gaussian", refine_iterations=0)


@pytest.mark.xfail(
    strict=True,
    reason="Ticket 04 will define refine_iterations as the exact optimizer-pass count.",
)
def test_future_contract_refine_iterations_is_the_exact_pass_count() -> None:
    fit = _synthetic_fit(cluster_ids=(11,))
    _result, _completion_order, snapshots = _run_with_executor(
        fit,
        FitConfig(lineshape="gaussian", refine_iterations=1, max_iterations=25),
    )

    assert len(snapshots) == 1


@pytest.mark.parametrize(
    ("optimizer", "optimizer_config"),
    [
        ("varpro", VarProConfig(max_nfev=25)),
        ("basin_hopping", BasinHoppingConfig(n_iterations=1, seed=23)),
    ],
)
def test_current_actual_optimizer_provenance_is_replaced_during_reconstruction(
    optimizer: str,
    optimizer_config: VarProConfig | BasinHoppingConfig,
) -> None:
    fit = _synthetic_fit(cluster_ids=(11,))
    cluster = fit.clusters[0]
    params = Parameters.from_peaks(fit.peaks, fixed=False)

    terminal = fit_with_optimizer(
        optimizer,
        params,
        cluster,
        noise=1.0,
        config=optimizer_config,
    )
    state = FittingState(
        clusters=[cluster],
        params=FitParameters.from_parameters(terminal.params, fit.peaks),
        scalar_params=terminal.params,
        noise=1.0,
    )
    reconstructed = build_fit_results(state, fit.spectra, config={}, input_files={})
    persisted = reconstructed.statistics[0]
    _shapes, analytical_amplitudes = calculate_shape_heights(terminal.params, cluster)
    injected_amplitudes = np.array(
        [
            parameter.value
            for parameter in terminal.params.params.values()
            if parameter.param_id is not None and parameter.param_id.label == "I"
        ],
        dtype=np.float64,
    )
    persisted_amplitudes = np.array(
        [amplitude.value for amplitude in reconstructed.clusters[0].amplitudes],
        dtype=np.float64,
    )

    assert terminal.nfev > 0
    assert terminal.message
    assert np.isfinite(terminal.residual).all()
    np.testing.assert_allclose(
        terminal.residual,
        residuals(terminal.params, cluster, 1.0),
        rtol=1e-8,
        atol=1e-12,
    )
    np.testing.assert_allclose(persisted_amplitudes, analytical_amplitudes.ravel())
    assert persisted.fit_converged is True
    assert persisted.n_function_evals == 0
    assert persisted.fit_message == "Statistics computed from fitted model"
    if optimizer == "varpro":
        np.testing.assert_allclose(injected_amplitudes, analytical_amplitudes.ravel())
        assert terminal.njev >= 0
        assert terminal.optimality >= 0.0
        assert len(terminal.params.get_computed_names()) == cluster.n_amplitude_params
    else:
        assert injected_amplitudes.size == 0
        assert terminal.params.get_computed_names() == []
        assert terminal.metadata["global_iterations"] == 1
        assert terminal.metadata["seed"] == 23
        assert terminal.metadata["local_minimizations"] >= 0
        assert isinstance(terminal.metadata["global_minimum_found"], bool)
        assert np.isfinite(terminal.metadata["initial_cost"])
        assert terminal.success is terminal.metadata["global_minimum_found"]


def test_current_persistence_resolves_amplitudes_independently_of_optimizer_parameters() -> None:
    fit = _synthetic_fit(cluster_ids=(11,))
    cluster = fit.clusters[0]
    terminal = fit_with_optimizer(
        "varpro",
        Parameters.from_peaks(fit.peaks, fixed=False),
        cluster,
        noise=1.0,
        config=VarProConfig(max_nfev=25),
    )
    state = FittingState(
        clusters=[cluster],
        params=FitParameters.from_parameters(terminal.params, fit.peaks),
        scalar_params=terminal.params,
        noise=1.0,
    )

    before = build_fit_results(state, fit.spectra, config={}, input_files={})
    for parameter_name in terminal.params.get_computed_names():
        terminal.params[parameter_name].value += 10_000.0
    after = build_fit_results(state, fit.spectra, config={}, input_files={})

    assert [amplitude.value for amplitude in before.clusters[0].amplitudes] == pytest.approx(
        [amplitude.value for amplitude in after.clusters[0].amplitudes]
    )
