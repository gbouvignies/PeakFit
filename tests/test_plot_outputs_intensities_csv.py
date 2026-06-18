import numpy as np
import pytest

from peakfit.plot.outputs import generate_cest_plots, generate_cpmg_plots, generate_intensity_plots
from peakfit.plot.profile_data import prepare_cest_data, prepare_cpmg_data


def _write_intensities_csv(path, rows) -> None:
    path.write_text(
        "\n".join(["cluster_id,peak_name,plane_index,intensity,intensity_err,z_value", *rows])
        + "\n",
        encoding="utf-8",
    )


def test_intensity_plots_use_intensities_csv(tmp_path) -> None:
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    _write_intensities_csv(
        tables_dir / "intensities.csv",
        [
            "1,P0001,0,100.0,2.0,-12000.0",
            "1,P0001,1,50.0,2.0,-100.0",
            "1,P0001,2,100.0,2.0,12000.0",
        ],
    )
    output = generate_intensity_plots(
        tmp_path,
        output_path=tmp_path / "intensity_profiles.pdf",
        show=False,
    )

    assert output.n_plots == 1
    assert output.path.exists()


def test_cest_plots_normalize_against_reference_points(tmp_path) -> None:
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    _write_intensities_csv(
        tables_dir / "intensities.csv",
        [
            "1,P0001,0,100.0,2.0,-12000.0",
            "1,P0001,1,50.0,2.0,-100.0",
            "1,P0001,2,100.0,2.0,12000.0",
        ],
    )

    output = generate_cest_plots(
        tmp_path,
        output_path=tmp_path / "cest_profiles.pdf",
        show=False,
    )

    assert output.n_plots == 1
    assert output.path.exists()


def test_cest_transform_uses_auto_reference_offsets() -> None:
    data = prepare_cest_data(
        [
            (-12000.0, 100.0, 2.0),
            (-100.0, 50.0, 2.0),
            (12000.0, 100.0, 2.0),
        ],
        ref_points=[-1],
    )

    assert data is not None
    assert data["offset"].tolist() == [-100.0]
    assert data["intensity"].tolist() == [0.5]
    assert data["error"][0] > 0


def test_cpmg_plots_convert_intensities_to_r2eff(tmp_path) -> None:
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    _write_intensities_csv(
        tables_dir / "intensities.csv",
        [
            "1,P0001,0,100.0,2.0,0.0",
            "1,P0001,1,80.0,2.0,1.0",
            "1,P0001,2,70.0,2.0,2.0",
        ],
    )

    output = generate_cpmg_plots(
        tmp_path,
        time_t2=0.04,
        output_path=tmp_path / "cpmg_profiles.pdf",
        show=False,
    )

    assert output.n_plots == 1
    assert output.path.exists()


def test_cpmg_transform_is_deterministic() -> None:
    data = prepare_cpmg_data(
        [
            (0.0, 100.0, 2.0),
            (1.0, 80.0, 2.0),
            (2.0, 70.0, 2.0),
        ],
        time_t2=0.04,
    )

    assert data is not None
    assert data["nu_cpmg"].tolist() == [25.0, 50.0]
    assert data["r2eff"].tolist() == pytest.approx([-np.log(0.8) / 0.04, -np.log(0.7) / 0.04])
    assert data["error"].tolist() == pytest.approx(
        [
            0.8 * np.sqrt((2.0 / 80.0) ** 2 + (2.0 / 100.0) ** 2) / 0.04,
            0.7 * np.sqrt((2.0 / 70.0) ** 2 + (2.0 / 100.0) ** 2) / 0.04,
        ]
    )


def test_cpmg_transform_accepts_sign_consistent_negative_intensities() -> None:
    data = prepare_cpmg_data(
        [
            (0.0, -100.0, 2.0),
            (1.0, -80.0, 2.0),
            (2.0, -70.0, 2.0),
        ],
        time_t2=0.04,
    )

    assert data is not None
    assert data["nu_cpmg"].tolist() == [25.0, 50.0]
    assert data["r2eff"].tolist() == pytest.approx([-np.log(0.8) / 0.04, -np.log(0.7) / 0.04])


def test_cpmg_requires_positive_t2(tmp_path) -> None:
    with pytest.raises(ValueError, match="time_t2 must be greater than zero"):
        generate_cpmg_plots(tmp_path, time_t2=0.0)
