from types import MappingProxyType

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
from peakfit.fit.output_metadata import RunMetadata
from peakfit.fit.run_models import RunSummary
from peakfit.io.writers.config import WriterConfig
from peakfit.io.writers.orchestrator import build_output_plan, write_fit_outputs


def _outcome() -> FinalFitOutcome:
    matrix = np.array([[1000.0]])
    matrix.flags.writeable = False
    vector = np.array([25.0])
    vector.flags.writeable = False
    evaluation = FinalAnalyticalEvaluation(
        cluster_id=3,
        shapes=matrix,
        amplitudes=matrix,
        amplitude_standard_errors=vector,
        amplitude_covariance=matrix,
        scaled_amplitude_standard_errors=vector,
        model_values=matrix,
        raw_residuals=matrix,
        normalized_residuals=vector,
        statistics=AnalyticalFitStatistics(
            chi_squared=10.0,
            n_observations=1,
            n_nonlinear_parameters=1,
            n_amplitude_parameters=1,
            n_fitted_parameters=2,
            degrees_of_freedom=1,
            reduced_chi_squared=10.0,
            amplitude_uncertainty_scale=1.0,
            aic=0.0,
            bic=0.0,
            log_likelihood=0.0,
        ),
    )
    cluster = FinalClusterOutcome(
        cluster_id=3,
        peak_names=("A1",),
        classification=FitOutcomeClassification.CONVERGED,
        correction_revision=2,
        optimizer_provenance=OptimizerProvenance(
            optimizer_kind="varpro",
            converged=True,
            termination_message="converged",
            function_evaluations=7,
            jacobian_evaluations=None,
            iterations=None,
            optimality=None,
            final_cost=5.0,
            correction_revision=2,
            metadata=MappingProxyType({}),
        ),
        final_nonlinear_parameters=(
            FinalParameter(
                name="A1.F2.cs",
                value=8.2,
                min=-np.inf,
                max=np.inf,
                vary=True,
                unit="ppm",
                standard_error=0.01,
            ),
        ),
        analytical_evaluation=evaluation,
        unusable_reason=None,
    )
    return FinalFitOutcome(
        clusters=(cluster,),
        by_cluster_id=MappingProxyType({3: cluster}),
        final_nonlinear_parameters=cluster.final_nonlinear_parameters,
        terminal_correction_revision=2,
        noise=1.0,
        n_optimizer_passes=1,
        n_correction_updates=0,
        overall_converged=True,
        statistics=FinalFitStatistics(
            chi_squared=10.0,
            reduced_chi_squared=10.0,
            n_observations=1,
            n_fitted_parameters=2,
            degrees_of_freedom=1,
            aic=0.0,
            bic=0.0,
            log_likelihood=0.0,
            function_evaluations=7,
        ),
    )


def test_core_output_plan_excludes_markdown_report(tmp_path) -> None:
    plan = build_output_plan(_outcome(), tmp_path, WriterConfig())

    assert set(plan) == {
        "summary_fit",
        "parameters",
        "intensities",
        "shifts",
        "clusters",
    }


def test_txt_format_adds_markdown_report(tmp_path) -> None:
    config = WriterConfig(formats=("json", "csv", "txt"))

    plan = build_output_plan(_outcome(), tmp_path, config)

    assert plan["report"] == tmp_path / "summary" / "report.md"


def test_txt_format_writes_markdown_report(tmp_path) -> None:
    outcome = _outcome()
    written = write_fit_outputs(
        outcome,
        tmp_path,
        WriterConfig(formats=("csv", "txt")),
        summary=RunSummary.from_outcome(outcome),
    )

    report_path = written["report"]
    text = report_path.read_text(encoding="utf-8")

    assert report_path == tmp_path / "summary" / "report.md"
    assert text.startswith("# PeakFit Report")
    assert "## Clusters" in text


def test_write_fit_outputs_returns_fit_artifacts_only(tmp_path) -> None:
    written = write_fit_outputs(_outcome(), tmp_path, WriterConfig(formats=("csv",)))

    assert "manifest" not in written
    assert "summary_fit" not in written
    assert "parameters" in written
    assert "intensities" in written


def test_fit_outputs_do_not_write_manifest(tmp_path) -> None:
    written = write_fit_outputs(_outcome(), tmp_path, WriterConfig(formats=("csv",)))

    assert "manifest" not in written
    assert not (tmp_path / "manifest.json").exists()


def test_json_output_uses_the_authoritative_final_outcome(tmp_path) -> None:
    written = write_fit_outputs(
        _outcome(),
        tmp_path,
        WriterConfig(formats=("json",)),
        metadata=RunMetadata(),
    )

    assert written["summary_fit"] == tmp_path / "summary" / "fit.json"


def test_parameters_csv_keeps_only_table_relevant_columns(tmp_path) -> None:
    written = write_fit_outputs(_outcome(), tmp_path, WriterConfig(formats=("csv",)))

    header = written["parameters"].read_text(encoding="utf-8").splitlines()[0].split(",")

    assert header[:5] == [
        "peak_name",
        "parameter_name",
        "value",
        "std_error",
        "is_fixed",
    ]
    assert "cluster_id" in header
    assert "classification" in header
    assert "category" not in header
    assert "is_global" not in header


def test_intensities_csv_orders_series_columns_for_reading(tmp_path) -> None:
    written = write_fit_outputs(_outcome(), tmp_path, WriterConfig(formats=("csv",)))

    header = written["intensities"].read_text(encoding="utf-8").splitlines()[0].split(",")

    assert header[:5] == [
        "peak_name",
        "z_value",
        "intensity",
        "intensity_err",
        "plane_index",
    ]
    assert "cluster_id" in header
    assert "classification" in header


def test_shifts_csv_orders_shift_columns_for_reading(tmp_path) -> None:
    written = write_fit_outputs(_outcome(), tmp_path, WriterConfig(formats=("csv",)))

    header = written["shifts"].read_text(encoding="utf-8").splitlines()[0].split(",")

    assert header[:3] == ["peak_name", "cs_F2_ppm", "cs_F2_err"]
    assert "cluster_id" in header
    assert "classification" in header


def test_fit_outputs_do_not_duplicate_run_metadata(tmp_path) -> None:
    written = write_fit_outputs(_outcome(), tmp_path, WriterConfig(formats=("csv",)))

    assert "metadata_run" not in written
    assert not (tmp_path / "metadata" / "run.json").exists()


def test_fit_outputs_do_not_duplicate_summary_diagnostics(tmp_path) -> None:
    written = write_fit_outputs(_outcome(), tmp_path, WriterConfig(formats=("csv",)))

    assert "statistics" not in written
    assert "diagnostics_mcmc" not in written
    assert not (tmp_path / "diagnostics" / "statistics.json").exists()
    assert not (tmp_path / "diagnostics" / "mcmc.json").exists()
