"""Completed-run projections from the authoritative final outcome."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import pytest

from peakfit.cli.commands.fit import _format_review_clusters
from peakfit.engine.algorithms.evaluation import AnalyticalFitStatistics, FitOutcomeClassification
from peakfit.fit.final_outcome import (
    FinalAnalyticalEvaluation,
    FinalClusterOutcome,
    FinalFitOutcome,
    FinalFitStatistics,
    OptimizerProvenance,
)
from peakfit.fit.fitting import find_review_clusters
from peakfit.fit.run_models import FitRun, RunSummary
from peakfit.io.writers.run_files import write_readme

if TYPE_CHECKING:
    from pathlib import Path


def _outcome(
    records: list[tuple[int, FitOutcomeClassification, float | None, str]],
) -> FinalFitOutcome:
    """Build a completed outcome with independently specified classifications."""
    clusters = tuple(
        _cluster_outcome(cluster_id, classification, redchi, message)
        for cluster_id, classification, redchi, message in records
    )
    return FinalFitOutcome(
        clusters=clusters,
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
            chi_squared=0.0,
            reduced_chi_squared=0.0,
            n_observations=0,
            n_fitted_parameters=0,
            degrees_of_freedom=1,
            aic=None,
            bic=None,
            log_likelihood=None,
            function_evaluations=None,
        ),
    )


def _cluster_outcome(
    cluster_id: int,
    classification: FitOutcomeClassification,
    redchi: float | None,
    message: str,
) -> FinalClusterOutcome:
    """Build one final cluster outcome without reconstructing a fit."""
    analytical = _evaluation(cluster_id, redchi) if redchi is not None else None
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
            jacobian_evaluations=None,
            iterations=None,
            optimality=None,
            final_cost=0.0,
            correction_revision=4,
            metadata=MappingProxyType({}),
        ),
        final_nonlinear_parameters=(),
        analytical_evaluation=analytical,
        unusable_reason="non-finite residuals" if analytical is None else None,
    )


def _evaluation(cluster_id: int, redchi: float) -> FinalAnalyticalEvaluation:
    """Build a frozen numerical record with a known reduced chi-squared value."""
    values = np.array([[redchi]], dtype=np.float64)
    values.flags.writeable = False
    statistics = AnalyticalFitStatistics(
        chi_squared=redchi,
        n_observations=2,
        n_nonlinear_parameters=0,
        n_amplitude_parameters=1,
        n_fitted_parameters=1,
        degrees_of_freedom=1,
        reduced_chi_squared=redchi,
        amplitude_uncertainty_scale=1.0,
        aic=0.0,
        bic=0.0,
        log_likelihood=0.0,
    )
    return FinalAnalyticalEvaluation(
        cluster_id=cluster_id,
        shapes=values,
        amplitudes=values,
        amplitude_standard_errors=values,
        amplitude_covariance=values,
        scaled_amplitude_standard_errors=values,
        model_values=values,
        raw_residuals=values,
        normalized_residuals=values,
        statistics=statistics,
    )


def test_run_summary_counts_all_converged_outcomes() -> None:
    summary = RunSummary.from_outcome(
        _outcome(
            [
                (11, FitOutcomeClassification.CONVERGED, 1.0, "converged"),
                (37, FitOutcomeClassification.CONVERGED, 3.0, "converged"),
            ]
        )
    )

    assert summary.n_clusters == 2
    assert summary.n_converged == 2
    assert summary.n_usable_non_converged == 0
    assert summary.n_unusable == 0
    assert summary.n_usable == 2
    assert summary.success_rate == 1.0
    assert summary.redchi_population_size == 2
    assert summary.mean_redchi == 2.0
    assert summary.median_redchi == 2.0


def test_run_summary_includes_only_usable_outcomes_in_statistics() -> None:
    summary = RunSummary.from_outcome(
        _outcome(
            [
                (11, FitOutcomeClassification.CONVERGED, 1.0, "converged"),
                (
                    37,
                    FitOutcomeClassification.USABLE_NON_CONVERGED,
                    5.0,
                    "maximum iterations reached",
                ),
                (91, FitOutcomeClassification.UNUSABLE, None, "non-finite result"),
            ]
        )
    )

    assert summary.n_clusters == 3
    assert summary.n_converged == 1
    assert summary.n_usable_non_converged == 1
    assert summary.n_unusable == 1
    assert summary.n_usable == 2
    assert summary.success_rate == 1.0 / 3.0
    assert summary.redchi_population_size == 2
    assert summary.mean_redchi == 3.0
    assert summary.std_redchi == 2.0
    assert summary.median_redchi == 3.0


def test_all_unusable_runs_report_no_numeric_distribution_or_fabricated_readme_value(
    tmp_path: Path,
) -> None:
    summary = RunSummary.from_outcome(
        _outcome(
            [
                (11, FitOutcomeClassification.UNUSABLE, None, "non-finite result"),
                (91, FitOutcomeClassification.UNUSABLE, None, "singular solve"),
            ]
        )
    )

    assert summary.n_clusters == 2
    assert summary.n_usable == 0
    assert summary.redchi_population_size == 0
    assert summary.mean_redchi is None
    assert summary.std_redchi is None
    assert summary.median_redchi is None
    assert "N/A (no usable outcomes)" in write_readme(tmp_path, summary).read_text(encoding="utf-8")


def test_cluster_review_uses_ordered_outcomes_and_actual_provenance() -> None:
    outcome = _outcome(
        [
            (91, FitOutcomeClassification.UNUSABLE, None, "singular solve"),
            (
                37,
                FitOutcomeClassification.USABLE_NON_CONVERGED,
                5.0,
                "maximum iterations reached",
            ),
            (11, FitOutcomeClassification.CONVERGED, 1.0, "converged"),
        ]
    )
    run = FitRun(
        outcome=outcome,
        continuation_state=None,  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
        spectra=None,
    )

    reviews = find_review_clusters(run)

    assert [(review.cluster_id, review.reason) for review in reviews] == [
        ("37", "not_converged"),
        ("91", "unusable"),
    ]
    assert reviews[0].classification is FitOutcomeClassification.USABLE_NON_CONVERGED
    assert reviews[0].redchi == 5.0
    assert reviews[0].termination_message == "maximum iterations reached"
    assert reviews[1].classification is FitOutcomeClassification.UNUSABLE
    assert reviews[1].redchi is None
    assert reviews[1].unusable_reason == "non-finite residuals"
    assert reviews[1].termination_message == "singular solve"


def test_cli_review_keeps_terminal_message_distinct_from_convergence() -> None:
    outcome = _outcome(
        [
            (
                37,
                FitOutcomeClassification.USABLE_NON_CONVERGED,
                5.0,
                "Optimization terminated successfully",
            ),
        ]
    )
    run = FitRun(
        outcome=outcome,
        continuation_state=None,  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )

    rendered = _format_review_clusters(find_review_clusters(run))

    assert rendered == [
        {
            "id": "37",
            "label": "P37",
            "status": "Not converged",
            "status_color": "metric.warn",
            "chi_sq": 5.0,
            "chi_sq_color": "metric.warn",
            "details": "Optimization terminated successfully",
        }
    ]


def test_completed_run_exposes_continuation_state_only_by_its_explicit_name() -> None:
    run = FitRun(
        outcome=_outcome([(11, FitOutcomeClassification.CONVERGED, 1.0, "converged")]),
        continuation_state=None,  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )

    with pytest.raises(AttributeError, match="state"):
        object.__getattribute__(run, "state")
