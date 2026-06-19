import json

from peakfit.fit.results import (
    AmplitudeEstimate,
    ClusterEstimates,
    ConvergenceStatus,
    FitResults,
    FitStatistics,
    MCMCDiagnostics,
    ParameterDiagnostic,
    ParameterEstimate,
)
from peakfit.io.schemas import OUTPUT_SCHEMA_VERSION
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
    written = write_fit_outputs(_results(), tmp_path, WriterConfig(formats=("json", "csv", "txt")))

    report_path = written["report"]
    text = report_path.read_text(encoding="utf-8")

    assert report_path == tmp_path / "summary" / "report.md"
    assert text.startswith("# PeakFit Report")
    assert "## Clusters" in text


def test_write_fit_outputs_returns_fit_artifacts_only(tmp_path) -> None:
    written = write_fit_outputs(_results(), tmp_path, WriterConfig())

    assert "manifest" not in written
    assert "summary_fit" in written
    assert "parameters" in written
    assert "intensities" in written


def test_fit_outputs_do_not_write_manifest(tmp_path) -> None:
    written = write_fit_outputs(_results(), tmp_path, WriterConfig())

    assert "manifest" not in written
    assert not (tmp_path / "manifest.json").exists()


def test_json_outputs_use_current_schema_version(tmp_path) -> None:
    written = write_fit_outputs(_results(), tmp_path, WriterConfig())

    with written["summary_fit"].open() as fh:
        payload = json.load(fh)
    assert payload["schema_version"] == OUTPUT_SCHEMA_VERSION


def test_parameters_csv_keeps_only_table_relevant_columns(tmp_path) -> None:
    written = write_fit_outputs(_results(), tmp_path, WriterConfig())

    header = written["parameters"].read_text(encoding="utf-8").splitlines()[0].split(",")

    assert header[:6] == [
        "cluster_id",
        "peak_name",
        "parameter_name",
        "value",
        "std_error",
        "is_fixed",
    ]
    assert "category" not in header
    assert "is_global" not in header


def test_fit_outputs_do_not_duplicate_run_metadata(tmp_path) -> None:
    written = write_fit_outputs(_results(), tmp_path, WriterConfig())

    assert "metadata_run" not in written
    assert not (tmp_path / "metadata" / "run.json").exists()


def test_fit_outputs_do_not_duplicate_summary_diagnostics(tmp_path) -> None:
    written = write_fit_outputs(_results(), tmp_path, WriterConfig())

    assert "statistics" not in written
    assert "diagnostics_mcmc" not in written
    assert not (tmp_path / "diagnostics" / "statistics.json").exists()
    assert not (tmp_path / "diagnostics" / "mcmc.json").exists()


def test_summary_mcmc_uses_schema_parameter_field(tmp_path) -> None:
    results = _results()
    results.mcmc_diagnostics = [
        MCMCDiagnostics(
            n_chains=4,
            n_samples=100,
            burn_in=20,
            parameter_diagnostics=[
                ParameterDiagnostic(
                    name="A1.F2.cs",
                    rhat=1.02,
                    ess_bulk=500.0,
                    ess_tail=450.0,
                    status=ConvergenceStatus.ACCEPTABLE,
                )
            ],
        )
    ]

    written = write_fit_outputs(results, tmp_path, WriterConfig())

    with written["summary_fit"].open() as fh:
        summary = json.load(fh)

    mcmc_summary = summary["mcmc_diagnostics"][0]
    assert "parameters" in mcmc_summary
    assert "parameter_diagnostics" not in mcmc_summary
    assert not (tmp_path / "diagnostics" / "mcmc.json").exists()
