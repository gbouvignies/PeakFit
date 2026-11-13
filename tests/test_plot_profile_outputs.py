"""CLI profile plots: intensities from the table; CEST and CPMG ratios from the archive.

Ratios and R2eff need each peak's covariance between planes, which only the scientific
archive records (profile-uncertainty §4); the per-plane error column cannot carry it.
"""

import numpy as np
import pytest
from builders import completed_result

from peakfit.engine.algorithms.covariance_factors import peak_covariances_between_planes
from peakfit.engine.scientific.profiles import Normalized, normalize
from peakfit.gui.fit_export import export_fit_result
from peakfit.plot.outputs import (
    generate_cest_plots,
    generate_cpmg_plots,
    generate_intensity_plots,
    recorded_profiles,
)
from peakfit.plot.profile_data import RecordedProfile, prepare_cest_data, prepare_cpmg_data


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


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """One recorded run, exported once; its archive is what the CLI plots read."""
    _spectra, _config, _pipeline, result = completed_result(["csv"], False)
    return result, export_fit_result(result, tmp_path_factory.mktemp("run") / "results")


def _independent(x, intensity, sigma) -> RecordedProfile:
    intensity = np.asarray(intensity, dtype=float)
    return RecordedProfile(
        np.asarray(x, dtype=float), intensity, np.diag(np.broadcast_to(sigma, intensity.shape) ** 2)
    )


def test_recorded_profiles_are_the_archives_amplitudes_and_covariance(results) -> None:
    result, directory = results
    profiles = recorded_profiles(directory)
    planes = np.asarray(result.inputs.snapshot.plane_values, dtype=float)
    order = np.argsort(planes, kind="stable")  # plotted along the plane value
    names = set()
    for cluster in result.outcome.clusters:
        evaluation = cluster.analytical_evaluation
        blocks = peak_covariances_between_planes(
            evaluation.covariance_factors, evaluation.amplitudes
        )
        for peak, name in enumerate(cluster.peak_names):
            names.add(name)
            profile = profiles[name]
            np.testing.assert_array_equal(profile.x, planes[order])
            np.testing.assert_array_equal(profile.intensity, evaluation.amplitudes[peak, order])
            np.testing.assert_array_equal(profile.covariance, blocks[peak][np.ix_(order, order)])
    assert set(profiles) == names


def test_cest_and_cpmg_plots_read_the_archive(results, tmp_path) -> None:
    _result, directory = results
    cest = generate_cest_plots(directory, output_path=tmp_path / "cest.pdf", reference_indices=[0])
    assert cest.n_plots == len(recorded_profiles(directory))
    assert cest.path.exists()
    # The synthetic profiles change sign, so no ratio converts to R2eff; each says why.
    warnings: list[str] = []
    reporter = type(
        "Collect", (), {"info": lambda *_: None, "warning": lambda _s, m: warnings.append(m)}
    )()
    cpmg = generate_cpmg_plots(
        directory, time_t2=0.04, output_path=tmp_path / "cpmg.pdf", reporter=reporter
    )
    assert cpmg.n_plots + len(warnings) == cest.n_plots
    assert all(w.startswith("Skipped ") for w in warnings)


def test_profile_plots_without_an_archive_say_how_to_get_one(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match=r"No scientific\.npz"):
        generate_cest_plots(tmp_path, output_path=tmp_path / "cest.pdf")


def test_cest_transform_uses_auto_reference_offsets() -> None:
    data = prepare_cest_data(
        _independent([-12000.0, -100.0, 12000.0], [100.0, 50.0, 100.0], 2.0), ref_points=[-1]
    )

    assert not isinstance(data, str)
    assert data["offset"].tolist() == [-100.0]
    assert data["intensity"].tolist() == [0.5]
    assert data["error"][0] > 0


def test_cpmg_transform_without_covariance_between_planes_is_the_independent_one() -> None:
    data = prepare_cpmg_data(_independent([0.0, 1.0, 2.0], [100.0, 80.0, 70.0], 2.0), 0.04)

    assert not isinstance(data, str)
    assert data["nu_cpmg"].tolist() == [25.0, 50.0]
    assert data["r2eff"].tolist() == pytest.approx([-np.log(0.8) / 0.04, -np.log(0.7) / 0.04])
    # σ(R2eff) = σ_r / (r T) with σ_r / r = √((σ/I)² + (σ_ref/I_ref)²).
    assert data["error"].tolist() == pytest.approx(
        [
            np.sqrt((2.0 / 80.0) ** 2 + (2.0 / 100.0) ** 2) / 0.04,
            np.sqrt((2.0 / 70.0) ** 2 + (2.0 / 100.0) ** 2) / 0.04,
        ]
    )


def test_cpmg_errors_carry_the_covariance_with_the_reference() -> None:
    # A shared lineshape error moves every plane together: it cancels from the ratio.
    intensity = np.array([100.0, 80.0, 70.0])
    common = 1.5**2 * np.outer(intensity, intensity) / 100.0**2
    profile = RecordedProfile(np.array([0.0, 1.0, 2.0]), intensity, 4.0 * np.eye(3) + common)
    data = prepare_cpmg_data(profile, 0.04)
    independent = prepare_cpmg_data(_independent(profile.x, intensity, 2.0), 0.04)
    assert not isinstance(data, str)
    assert not isinstance(independent, str)
    np.testing.assert_allclose(data["error"], independent["error"], rtol=1e-12)


def test_cpmg_transform_accepts_sign_consistent_negative_intensities() -> None:
    data = prepare_cpmg_data(_independent([0.0, 1.0, 2.0], [-100.0, -80.0, -70.0], 2.0), 0.04)

    assert not isinstance(data, str)
    assert data["nu_cpmg"].tolist() == [25.0, 50.0]
    assert data["r2eff"].tolist() == pytest.approx([-np.log(0.8) / 0.04, -np.log(0.7) / 0.04])


def test_cpmg_requires_positive_t2(tmp_path) -> None:
    with pytest.raises(ValueError, match="time_t2 must be greater than zero"):
        generate_cpmg_plots(tmp_path, time_t2=0.0)


def test_cest_ratios_are_the_shared_profile_normalization() -> None:
    intensity = np.array([100.0, 50.0, 0.0, 98.0])
    rng = np.random.default_rng(3)
    modes = rng.standard_normal((4, 2))
    covariance = np.diag([4.0, 9.0, 9.0, 4.0]) + modes @ modes.T
    profile = RecordedProfile(np.array([-12000.0, -100.0, 0.0, 12000.0]), intensity, covariance)
    data = prepare_cest_data(profile, ref_points=[-1])
    expected = normalize(intensity, covariance, np.array([0, 3]))
    assert isinstance(expected, Normalized)
    assert not isinstance(data, str)
    np.testing.assert_array_equal(data["intensity"], expected.values[1:3])
    np.testing.assert_array_equal(data["error"], expected.errors[1:3])


@pytest.mark.parametrize(
    ("profile", "reason"),
    [
        (
            RecordedProfile(np.array([0.0, 1.0]), np.array([100.0, 80.0]), np.full((2, 2), np.nan)),
            "σ unavailable",
        ),
        (_independent([0.0, 1.0], [10.0, 8.0], 2.0), "σ_R/|R| = 0.20"),
    ],
)
def test_cpmg_says_why_a_profile_is_not_converted(profile, reason) -> None:
    data = prepare_cpmg_data(profile, time_t2=0.04)
    assert isinstance(data, str)
    assert reason in data
