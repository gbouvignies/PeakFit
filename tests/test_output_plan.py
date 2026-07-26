import pytest

from peakfit.fit.results import (
    AmplitudeEstimate,
    ClusterEstimates,
    FitResults,
    FitStatistics,
    ParameterEstimate,
)
from peakfit.io.writers.config import WriterConfig
from peakfit.io.writers.orchestrator import build_output_plan, write_fit_outputs


def _results() -> FitResults:
    stats = FitStatistics(
        chi_squared=10.0,
        reduced_chi_squared=1.0,
        n_data=12,
        n_params=2,
        fit_converged=True,
    )
    cluster = ClusterEstimates(
        cluster_id=3,
        peak_names=["A1"],
        lineshape_params=[
            ParameterEstimate(
                name="A1.F2.cs",
                value=8.2,
                std_error=0.01,
                unit="ppm",
            )
        ],
        amplitudes=[
            AmplitudeEstimate(
                peak_name="A1",
                plane_index=0,
                z_value=0.0,
                value=1000.0,
                std_error=25.0,
            )
        ],
    )
    return FitResults(clusters=[cluster], statistics=[stats], global_statistics=stats)


def test_core_output_plan_excludes_markdown_report(tmp_path) -> None:
    plan = build_output_plan(_results(), tmp_path, WriterConfig())

    assert set(plan) == {
        "summary_fit",
        "parameters",
        "intensities",
        "shifts",
    }


def test_txt_format_adds_markdown_report(tmp_path) -> None:
    config = WriterConfig(formats=("json", "csv", "txt"))

    plan = build_output_plan(_results(), tmp_path, config)

    assert plan["report"] == tmp_path / "summary" / "report.md"


def test_txt_format_writes_markdown_report(tmp_path) -> None:
    written = write_fit_outputs(_results(), tmp_path, WriterConfig(formats=("csv", "txt")))

    report_path = written["report"]
    text = report_path.read_text(encoding="utf-8")

    assert report_path == tmp_path / "summary" / "report.md"
    assert text.startswith("# PeakFit Report")
    assert "## Clusters" in text


def test_write_fit_outputs_returns_fit_artifacts_only(tmp_path) -> None:
    written = write_fit_outputs(_results(), tmp_path, WriterConfig(formats=("csv",)))

    assert "manifest" not in written
    assert "summary_fit" not in written
    assert "parameters" in written
    assert "intensities" in written


def test_fit_outputs_do_not_write_manifest(tmp_path) -> None:
    written = write_fit_outputs(_results(), tmp_path, WriterConfig(formats=("csv",)))

    assert "manifest" not in written
    assert not (tmp_path / "manifest.json").exists()


def test_json_output_requires_the_authoritative_final_outcome(tmp_path) -> None:
    with pytest.raises(ValueError, match="FinalFitOutcome"):
        write_fit_outputs(_results(), tmp_path, WriterConfig(formats=("json",)))


def test_parameters_csv_keeps_only_table_relevant_columns(tmp_path) -> None:
    written = write_fit_outputs(_results(), tmp_path, WriterConfig(formats=("csv",)))

    header = written["parameters"].read_text(encoding="utf-8").splitlines()[0].split(",")

    assert header[:5] == [
        "peak_name",
        "parameter_name",
        "value",
        "std_error",
        "is_fixed",
    ]
    assert header[-1] == "cluster_id"
    assert "category" not in header
    assert "is_global" not in header


def test_intensities_csv_orders_series_columns_for_reading(tmp_path) -> None:
    written = write_fit_outputs(_results(), tmp_path, WriterConfig(formats=("csv",)))

    header = written["intensities"].read_text(encoding="utf-8").splitlines()[0].split(",")

    assert header == [
        "peak_name",
        "z_value",
        "intensity",
        "intensity_err",
        "plane_index",
        "cluster_id",
    ]


def test_shifts_csv_orders_shift_columns_for_reading(tmp_path) -> None:
    written = write_fit_outputs(_results(), tmp_path, WriterConfig(formats=("csv",)))

    header = written["shifts"].read_text(encoding="utf-8").splitlines()[0].split(",")

    assert header == ["peak_name", "cs_F2_ppm", "cs_F2_err", "cluster_id"]


def test_fit_outputs_do_not_duplicate_run_metadata(tmp_path) -> None:
    written = write_fit_outputs(_results(), tmp_path, WriterConfig(formats=("csv",)))

    assert "metadata_run" not in written
    assert not (tmp_path / "metadata" / "run.json").exists()


def test_fit_outputs_do_not_duplicate_summary_diagnostics(tmp_path) -> None:
    written = write_fit_outputs(_results(), tmp_path, WriterConfig(formats=("csv",)))

    assert "statistics" not in written
    assert "diagnostics_mcmc" not in written
    assert not (tmp_path / "diagnostics" / "statistics.json").exists()
    assert not (tmp_path / "diagnostics" / "mcmc.json").exists()
