from peakfit.engine.results import ClusterEstimates, FitResults, FitStatistics, ParameterEstimate
from peakfit.io.writers.markdown import write_report


def _cluster(index: int) -> ClusterEstimates:
    return ClusterEstimates(
        cluster_id=index,
        peak_names=[f"P{index}"],
        lineshape_params=[
            ParameterEstimate(name=f"P{index}.F2.cs", value=120.0 + index, std_error=0.01),
            ParameterEstimate(name=f"P{index}.F2.lw", value=20.0, std_error=0.5),
        ],
    )


def _stats(redchi: float = 1.0, converged: bool = True) -> FitStatistics:
    return FitStatistics(
        chi_squared=10.0,
        reduced_chi_squared=redchi,
        n_data=100,
        n_params=10,
        fit_converged=converged,
    )


def test_markdown_report_is_bounded_for_large_runs(tmp_path) -> None:
    clusters = [_cluster(index) for index in range(50)]
    results = FitResults(
        clusters=clusters,
        statistics=[_stats() for _ in clusters],
        global_statistics=_stats(),
    )

    report_path = write_report(results, tmp_path / "summary" / "report.md")

    text = report_path.read_text(encoding="utf-8")
    assert text.startswith("# PeakFit Report")
    assert "_Showing 40 of 50 clusters. See JSON/CSV outputs for full detail._" in text
    assert "_Showing 40 of 100 parameters. See JSON/CSV outputs for full detail._" in text
    assert "| 49 |" not in text
    assert "PeakFit Analysis Report" not in text
    assert "Executive Summary" not in text
    assert "✓" not in text
    assert "⚠" not in text


def test_markdown_report_prioritizes_clusters_to_check(tmp_path) -> None:
    clusters = [_cluster(index) for index in range(3)]
    results = FitResults(
        clusters=clusters,
        statistics=[
            _stats(),
            _stats(redchi=9.0),
            _stats(converged=False),
        ],
        global_statistics=_stats(redchi=9.0),
    )

    report_path = write_report(results, tmp_path / "summary" / "report.md")

    text = report_path.read_text(encoding="utf-8")
    cluster_lines = [
        line
        for line in text.splitlines()
        if line.startswith("| ") and line.split("|")[1].strip().isdigit()
    ]
    assert cluster_lines[0].startswith("| 2 |")
    assert cluster_lines[1].startswith("| 1 |")
    assert "Cluster 1 has reduced chi2 9." in text
    assert "Cluster 2 did not converge." in text
