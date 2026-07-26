from types import SimpleNamespace

from peakfit.io.writers.run_files import write_readme


def test_run_readme_reports_summary_first_and_existing_files(tmp_path) -> None:
    (tmp_path / "summary").mkdir()
    (tmp_path / "summary" / "fit.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "parameters.csv").write_text("parameter_name,value\n", encoding="utf-8")

    summary = SimpleNamespace(
        n_clusters=2,
        n_peaks=3,
        success_rate=0.5,
        n_converged=1,
        n_usable_non_converged=0,
        n_unusable=1,
        n_usable=1,
        redchi_population_size=1,
        median_redchi=1.23456,
    )

    readme_path = write_readme(tmp_path, summary)

    text = readme_path.read_text(encoding="utf-8")
    assert text.startswith("# PeakFit Run\n\n## Summary")
    assert "**Peaks**: 3" in text
    assert "**Converged clusters**: 1/2 (50.0%)" in text
    assert "**Usable, not converged**: 0" in text
    assert "**Unusable clusters**: 1" in text
    assert "**Usable clusters**: 1/2" in text
    assert "**Median reduced chi2**: 1.235" in text
    assert "`summary/fit.json`" in text
    assert "`tables/parameters.csv`" in text
    assert "`summary/report.md`" not in text
    assert "## Next Steps" in text
    assert "peakfit plot intensity <this-run-directory>" in text
    assert "peakfit mcmc <this-run-directory> --peaks <peak-name>" in text
    assert text.endswith("\n")
