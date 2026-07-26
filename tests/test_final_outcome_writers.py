"""Human and tabular completed-fit projections from final outcomes."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.algorithms.evaluation import AnalyticalFitStatistics, FitOutcomeClassification
from peakfit.fit.final_outcome import (
    FinalAnalyticalEvaluation,
    FinalClusterOutcome,
    FinalFitOutcome,
    FinalFitStatistics,
    FinalParameter,
    OptimizerProvenance,
)
from peakfit.fit.fitting import find_review_clusters
from peakfit.fit.result_models import RunMetadata
from peakfit.fit.run_models import FitRun, RunSummary
from peakfit.io.writers.config import WriterConfig
from peakfit.io.writers.orchestrator import write_fit_outputs
from peakfit.io.writers.run_files import write_readme

if TYPE_CHECKING:
    from pathlib import Path


def _outcome(
    records: list[tuple[int, FitOutcomeClassification, str]],
) -> FinalFitOutcome:
    clusters = tuple(
        _cluster(cluster_id, classification, message)
        for cluster_id, classification, message in records
    )
    return FinalFitOutcome(
        clusters=tuple(sorted(clusters, key=lambda cluster: cluster.cluster_id)),
        by_cluster_id=MappingProxyType({cluster.cluster_id: cluster for cluster in clusters}),
        final_nonlinear_parameters=(),
        terminal_correction_revision=4,
        noise=1.0,
        n_optimizer_passes=2,
        n_correction_updates=1,
        overall_converged=all(
            cluster.classification is FitOutcomeClassification.CONVERGED for cluster in clusters
        ),
        statistics=FinalFitStatistics(
            chi_squared=6.0,
            reduced_chi_squared=2.0,
            n_observations=6,
            n_fitted_parameters=3,
            degrees_of_freedom=3,
            aic=None,
            bic=None,
            log_likelihood=None,
            function_evaluations=20,
        ),
    )


def _cluster(
    cluster_id: int,
    classification: FitOutcomeClassification,
    message: str,
) -> FinalClusterOutcome:
    usable = classification is not FitOutcomeClassification.UNUSABLE
    matrix = np.array([[float(cluster_id)]])
    matrix.flags.writeable = False
    vector = np.array([float(cluster_id)])
    vector.flags.writeable = False
    evaluation = (
        FinalAnalyticalEvaluation(
            cluster_id=cluster_id,
            shapes=matrix,
            amplitudes=matrix,
            amplitude_standard_errors=vector,
            amplitude_covariance=matrix,
            scaled_amplitude_standard_errors=vector,
            model_values=matrix,
            raw_residuals=matrix,
            normalized_residuals=vector,
            statistics=AnalyticalFitStatistics(
                chi_squared=float(cluster_id),
                n_observations=1,
                n_nonlinear_parameters=1,
                n_amplitude_parameters=1,
                n_fitted_parameters=2,
                degrees_of_freedom=1,
                reduced_chi_squared=float(cluster_id),
                amplitude_uncertainty_scale=1.0,
                aic=0.0,
                bic=0.0,
                log_likelihood=0.0,
            ),
        )
        if usable
        else None
    )
    return FinalClusterOutcome(
        cluster_id=cluster_id,
        peak_names=(f"P{cluster_id}",),
        classification=classification,
        correction_revision=4,
        optimizer_provenance=OptimizerProvenance(
            optimizer_kind="varpro",
            converged=classification is FitOutcomeClassification.CONVERGED,
            termination_message=message,
            function_evaluations=cluster_id,
            jacobian_evaluations=cluster_id + 1,
            iterations=cluster_id + 2,
            optimality=0.1,
            final_cost=float(cluster_id),
            correction_revision=4,
            metadata=MappingProxyType({}),
        ),
        final_nonlinear_parameters=(
            FinalParameter(
                name=f"P{cluster_id}.F2.cs",
                value=float(cluster_id),
                min=0.0,
                max=100.0,
                vary=True,
                unit="ppm",
                standard_error=0.5,
            ),
        )
        if usable
        else (),
        analytical_evaluation=evaluation,
        unusable_reason="non-finite terminal result" if not usable else None,
    )


def _write(outcome: FinalFitOutcome, tmp_path) -> dict[str, Path]:
    return write_fit_outputs(
        outcome,
        tmp_path,
        WriterConfig(formats=("json", "csv", "txt")),
        metadata=RunMetadata(),
        z_values=np.array([1.5]),
        summary=RunSummary.from_outcome(outcome),
    )


def test_writers_project_mixed_outcomes_in_final_identity_order(tmp_path) -> None:
    outcome = _outcome(
        [
            (91, FitOutcomeClassification.UNUSABLE, "singular solve"),
            (
                37,
                FitOutcomeClassification.USABLE_NON_CONVERGED,
                "Optimization terminated successfully",
            ),
            (11, FitOutcomeClassification.CONVERGED, "converged"),
        ]
    )

    written = _write(outcome, tmp_path)
    report = written["report"].read_text(encoding="utf-8")
    with written["clusters"].open(newline="", encoding="utf-8") as handle:
        clusters = list(csv.DictReader(handle))
    payload = json.loads(written["summary_fit"].read_text(encoding="utf-8"))

    assert [row["cluster_id"] for row in clusters] == ["11", "37", "91"]
    assert [row["classification"] for row in clusters] == [
        "converged",
        "usable_non_converged",
        "unusable",
    ]
    assert clusters[1]["converged"] == "False"
    assert clusters[1]["usable"] == "True"
    assert clusters[1]["termination_message"] == "Optimization terminated successfully"
    assert clusters[2]["unusable_reason"] == "non-finite terminal result"
    assert clusters[2]["chi_squared"] == ""
    assert clusters[2]["reduced_chi_squared"] == ""
    assert "| 11 | P11 | 11 | converged |" in report
    assert "| 37 | P37 | 37 | usable, not converged |" in report
    assert "| 91 | P91 | N/A | unusable: non-finite terminal result |" in report
    assert "Optimization terminated successfully" in report
    assert "Usable, not converged: 1" in report
    assert "Unusable: 1" in report
    assert [cluster["cluster_id"] for cluster in payload["clusters"]] == [11, 37, 91]
    assert [cluster["classification"] for cluster in payload["clusters"]] == [
        row["classification"] for row in clusters
    ]
    summary = RunSummary.from_outcome(outcome)
    run = FitRun(
        outcome=outcome,
        continuation_state=None,  # type: ignore[arg-type]
        output_dir=tmp_path,
    )
    review_ids = {review.cluster_id for review in find_review_clusters(run)}
    readme = write_readme(tmp_path, summary).read_text(encoding="utf-8")

    assert {"37", "91"}.issubset(review_ids)
    assert summary.n_converged == 1
    assert summary.n_usable_non_converged == 1
    assert summary.n_unusable == 1
    assert "**Usable, not converged**: 1" in readme
    assert "**Unusable clusters**: 1" in readme


def test_tabular_projection_keeps_optimizer_success_distinct_from_classification(tmp_path) -> None:
    original = _cluster(91, FitOutcomeClassification.UNUSABLE, "invalid evaluation")
    optimistic_provenance = replace(
        original.optimizer_provenance, converged=True, final_cost=np.nan
    )
    unusable = replace(original, optimizer_provenance=optimistic_provenance)
    outcome = _outcome([(11, FitOutcomeClassification.CONVERGED, "converged")])
    outcome = replace(
        outcome,
        clusters=(outcome.cluster(11), unusable),
        by_cluster_id=MappingProxyType({11: outcome.cluster(11), 91: unusable}),
    )

    written = write_fit_outputs(
        outcome,
        tmp_path,
        WriterConfig(formats=("csv",)),
    )
    with written["clusters"].open(newline="", encoding="utf-8") as handle:
        rows = {int(row["cluster_id"]): row for row in csv.DictReader(handle)}

    assert rows[91]["classification"] == "unusable"
    assert rows[91]["converged"] == "False"
    assert rows[91]["optimizer_success"] == "True"
    assert rows[91]["final_cost"] == ""


def test_writers_omit_numerical_rows_for_all_unusable_outcomes(tmp_path) -> None:
    outcome = _outcome(
        [
            (91, FitOutcomeClassification.UNUSABLE, "singular solve"),
            (11, FitOutcomeClassification.UNUSABLE, "non-finite result"),
        ]
    )

    written = _write(outcome, tmp_path)
    report = written["report"].read_text(encoding="utf-8")
    parameters = written["parameters"].read_text(encoding="utf-8").splitlines()
    intensities = written["intensities"].read_text(encoding="utf-8").splitlines()
    readme = write_readme(tmp_path, RunSummary.from_outcome(outcome)).read_text(encoding="utf-8")

    assert len(parameters) == 1
    assert len(intensities) == 1
    assert "Global reduced chi2: N/A (no usable outcomes)" in report
    assert "unusable: non-finite terminal result" in report
    assert "**Usable clusters**: 0/2" in readme
    assert "N/A (no usable outcomes)" in readme


def test_writers_project_all_converged_statistics_and_populations(tmp_path) -> None:
    outcome = _outcome(
        [
            (37, FitOutcomeClassification.CONVERGED, "converged"),
            (11, FitOutcomeClassification.CONVERGED, "converged"),
        ]
    )

    written = _write(outcome, tmp_path)
    report = written["report"].read_text(encoding="utf-8")

    assert "Converged: 2" in report
    assert "Usable, not converged: 0" in report
    assert "Unusable: 0" in report
    assert "Usable clusters: 2" in report
    assert "Reduced chi2 population: 2" in report
