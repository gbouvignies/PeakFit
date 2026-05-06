from types import SimpleNamespace

from peakfit.plot.manager.service import PlotService


def _write_intensities_csv(path, rows) -> None:
    path.write_text(
        "\n".join(["cluster_id,peak_name,plane_index,intensity,intensity_err,z_value", *rows])
        + "\n",
        encoding="utf-8",
    )


class _SummaryOnlyLoader:
    def __init__(self, _directory) -> None:
        pass

    def load_summary(self):
        return SimpleNamespace(clusters=[SimpleNamespace(amplitudes=[])])


def test_intensity_plots_fall_back_to_intensities_csv(tmp_path, monkeypatch) -> None:
    params_dir = tmp_path / "parameters"
    params_dir.mkdir()
    _write_intensities_csv(
        params_dir / "intensities.csv",
        [
            "1,P0001,0,100.0,2.0,-12000.0",
            "1,P0001,1,50.0,2.0,-100.0",
            "1,P0001,2,100.0,2.0,12000.0",
        ],
    )
    monkeypatch.setattr("peakfit.plot.manager.service.ResultsLoader", _SummaryOnlyLoader)

    output = PlotService().generate_intensity_plots(
        tmp_path,
        output_path=tmp_path / "intensity_profiles.pdf",
        show=False,
    )

    assert output.n_plots == 1
    assert output.path.exists()


def test_cest_plots_fall_back_to_intensities_csv(tmp_path, monkeypatch) -> None:
    params_dir = tmp_path / "parameters"
    params_dir.mkdir()
    _write_intensities_csv(
        params_dir / "intensities.csv",
        [
            "1,P0001,0,100.0,2.0,-12000.0",
            "1,P0001,1,50.0,2.0,-100.0",
            "1,P0001,2,100.0,2.0,12000.0",
        ],
    )
    monkeypatch.setattr("peakfit.plot.manager.service.ResultsLoader", _SummaryOnlyLoader)

    output = PlotService().generate_cest_plots(
        tmp_path,
        output_path=tmp_path / "cest_profiles.pdf",
        reference_indices=[0, 2],
        show=False,
    )

    assert output.n_plots == 1
    assert output.path.exists()


def test_cest_plots_auto_reference_fallback_for_indexed_z_axis(tmp_path, monkeypatch) -> None:
    params_dir = tmp_path / "parameters"
    params_dir.mkdir()
    _write_intensities_csv(
        params_dir / "intensities.csv",
        [
            "1,P0001,0,100.0,2.0,0.0",
            "1,P0001,1,60.0,2.0,1.0",
            "1,P0001,2,40.0,2.0,2.0",
            "1,P0001,3,60.0,2.0,3.0",
            "1,P0001,4,100.0,2.0,4.0",
        ],
    )
    monkeypatch.setattr("peakfit.plot.manager.service.ResultsLoader", _SummaryOnlyLoader)

    output = PlotService().generate_cest_plots(
        tmp_path,
        output_path=tmp_path / "cest_profiles.pdf",
        show=False,
    )

    assert output.n_plots == 1
    assert output.path.exists()
