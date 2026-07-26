"""Executable reproduction of the cluster point/series axis contract."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from pydantic import ValidationError

from peakfit.engine.algorithms.clustering import create_clusters
from peakfit.engine.algorithms.common import calculate_shape_heights, residuals
from peakfit.engine.algorithms.linear_algebra import calculate_amplitudes_with_uncertainty
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import FitConfig
from peakfit.engine.domain.params_vector import FitParameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters
from peakfit.engine.domain.state import FittingState
from peakfit.engine.fitting.simulation import simulate_data
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.engine.results import FitResult
from peakfit.fit.results import build_fit_results

if TYPE_CHECKING:
    from peakfit.engine.domain.params_scalar import Parameters

N_POINTS = 5
N_SERIES = 3
EXPECTED_CHI_SQUARED = 60.0


def _spectral_parameters(*, size: int, direct: bool) -> SpectralParameters:
    return SpectralParameters(
        size=size,
        sw=500.0,
        obs=500.0,
        car=0.0,
        aq_time=0.01,
        apocode=0.0,
        apodq1=0.0,
        apodq2=0.0,
        apodq3=0.0,
        p180=False,
        direct=direct,
        ft=True,
    )


def _axis_reproduction() -> tuple[Cluster, Spectra, Parameters]:
    spectra = Spectra(
        dic={},
        data=np.zeros((N_SERIES, N_POINTS), dtype=np.float64),
        z_values=np.arange(N_SERIES, dtype=np.float64),
        params=[
            _spectral_parameters(size=N_SERIES, direct=False),
            _spectral_parameters(size=N_POINTS, direct=True),
        ],
    )
    config = FitConfig(lineshape="gaussian")
    peak = Peak(
        name="P1",
        positions=np.array([0.0], dtype=np.float64),
        shapes=create_shapes(spectra, config, "P1", [0.0], ["gaussian"]),
    )
    params = peak.create_params()
    grid_indices = [np.arange(N_POINTS, dtype=np.intp)]

    empty_cluster = Cluster(
        cluster_id=1,
        peaks=[peak],
        grid_indices=grid_indices,
        data=np.zeros((N_POINTS, N_SERIES), dtype=np.float64),
    )
    shapes = empty_cluster.evaluate(params)

    residual_basis = np.array([1.0, -2.0, 3.0, -4.0, 2.0], dtype=np.float64)
    residual_basis -= shapes[0] * (shapes[0] @ residual_basis) / (shapes[0] @ shapes[0])
    residual_basis *= np.sqrt(EXPECTED_CHI_SQUARED / (N_SERIES * np.sum(residual_basis**2)))

    known_amplitudes = np.array([[10.0, 20.0, 30.0]], dtype=np.float64)
    data = (shapes.T @ known_amplitudes) + residual_basis[:, np.newaxis]
    cluster = Cluster(
        cluster_id=1,
        peaks=[peak],
        grid_indices=grid_indices,
        data=data,
    )
    return cluster, spectra, params


def test_cluster_counts_series_and_amplitudes_from_the_series_axis() -> None:
    cluster, _spectra, _params = _axis_reproduction()

    assert cluster.data.shape == (N_POINTS, N_SERIES)
    assert (cluster.n_series, cluster.n_amplitude_params) == (
        N_SERIES,
        len(cluster.peaks) * N_SERIES,
    )


@pytest.mark.parametrize(
    "data",
    [
        np.zeros(N_POINTS, dtype=np.float64),
        np.zeros((N_POINTS, N_SERIES, 1), dtype=np.float64),
    ],
)
def test_cluster_rejects_data_that_are_not_point_by_series(data: np.ndarray) -> None:
    cluster, _spectra, _params = _axis_reproduction()

    with pytest.raises(ValueError, match="two-dimensional"):
        Cluster(
            cluster_id=cluster.cluster_id,
            peaks=cluster.peaks,
            grid_indices=cluster.grid_indices,
            data=data,
        )


def test_cluster_rejects_spectral_grid_dimensions_with_different_point_counts() -> None:
    cluster, _spectra, _params = _axis_reproduction()

    with pytest.raises(ValueError, match="grid dimensions"):
        Cluster(
            cluster_id=cluster.cluster_id,
            peaks=cluster.peaks,
            grid_indices=[
                np.arange(N_POINTS, dtype=np.intp),
                np.arange(N_POINTS - 1, dtype=np.intp),
            ],
            data=cluster.data,
        )


def test_cluster_rejects_data_and_grid_point_count_mismatch() -> None:
    cluster, _spectra, _params = _axis_reproduction()

    with pytest.raises(ValueError, match="point count"):
        Cluster(
            cluster_id=cluster.cluster_id,
            peaks=cluster.peaks,
            grid_indices=cluster.grid_indices,
            data=np.zeros((N_POINTS - 1, N_SERIES), dtype=np.float64),
        )


def test_cluster_rejects_empty_point_or_series_axes() -> None:
    cluster, _spectra, _params = _axis_reproduction()

    with pytest.raises(ValueError, match="at least one point"):
        Cluster(
            cluster_id=cluster.cluster_id,
            peaks=cluster.peaks,
            grid_indices=[np.array([], dtype=np.intp)],
            data=np.zeros((0, N_SERIES), dtype=np.float64),
        )

    with pytest.raises(ValueError, match="at least one series"):
        Cluster(
            cluster_id=cluster.cluster_id,
            peaks=cluster.peaks,
            grid_indices=cluster.grid_indices,
            data=np.zeros((N_POINTS, 0), dtype=np.float64),
        )


def test_single_series_cluster_uses_an_explicit_series_axis() -> None:
    cluster, _spectra, _params = _axis_reproduction()
    single_series = Cluster(
        cluster_id=cluster.cluster_id,
        peaks=cluster.peaks,
        grid_indices=cluster.grid_indices,
        data=cluster.data[:, :1],
    )

    assert single_series.data.shape == (N_POINTS, 1)
    assert single_series.n_points == N_POINTS
    assert single_series.n_series == 1
    assert single_series.n_observations == N_POINTS


def test_cluster_creation_transposes_series_major_spectra_to_point_major_data() -> None:
    cluster, spectra, _params = _axis_reproduction()
    spectra.data = np.arange(1, N_POINTS * N_SERIES + 1, dtype=np.float64).reshape(
        N_SERIES,
        N_POINTS,
    )

    created = create_clusters(spectra, cluster.peaks, contour_level=0.5)

    assert len(created) == 1
    assert created[0].data == pytest.approx(spectra.data.T)
    assert created[0].n_points == N_POINTS
    assert created[0].n_series == N_SERIES


def test_multidimensional_grid_indices_are_paired_flattened_point_coordinates() -> None:
    n_series = 2
    axis_extents = (2, 3)
    n_points = int(np.prod(axis_extents))
    spectra = Spectra(
        dic={},
        data=np.ones((n_series, *axis_extents), dtype=np.float64),
        z_values=np.arange(n_series, dtype=np.float64),
        params=[
            _spectral_parameters(size=n_series, direct=False),
            _spectral_parameters(size=axis_extents[0], direct=False),
            _spectral_parameters(size=axis_extents[1], direct=True),
        ],
    )
    config = FitConfig(lineshape="gaussian")
    positions = [spectral_params.pts2ppm(0) for spectral_params in spectra.spectral_params]
    peak = Peak(
        name="P1",
        positions=np.asarray(positions, dtype=np.float64),
        shapes=create_shapes(
            spectra,
            config,
            "P1",
            positions,
            ["gaussian", "gaussian"],
        ),
    )

    [cluster] = create_clusters(spectra, [peak], contour_level=0.5)

    assert tuple(np.unique(indices).size for indices in cluster.grid_indices) == axis_extents
    assert tuple(indices.size for indices in cluster.grid_indices) == (n_points, n_points)
    assert set(zip(*cluster.grid_indices, strict=True)) == {
        (axis_0, axis_1) for axis_0 in range(axis_extents[0]) for axis_1 in range(axis_extents[1])
    }
    assert cluster.data.shape == (n_points, n_series)
    assert cluster.evaluate(peak.create_params()).shape == (1, n_points)

    merged = cluster + cluster

    assert tuple(indices.size for indices in merged.grid_indices) == (
        n_points * 2,
        n_points * 2,
    )
    assert merged.data.shape == (n_points * 2, n_series)
    assert merged.evaluate(peak.create_params()).shape == (2, n_points * 2)


def test_cluster_merge_concatenates_points_and_preserves_series() -> None:
    cluster, _spectra, _params = _axis_reproduction()

    merged = cluster + cluster

    assert merged.data.shape == (N_POINTS * 2, N_SERIES)
    assert merged.n_points == N_POINTS * 2
    assert merged.n_series == N_SERIES
    assert merged.n_observations == N_POINTS * N_SERIES * 2


def test_cluster_merge_rejects_different_series_counts() -> None:
    cluster, _spectra, _params = _axis_reproduction()
    single_series = Cluster(
        cluster_id=cluster.cluster_id,
        peaks=cluster.peaks,
        grid_indices=cluster.grid_indices,
        data=cluster.data[:, :1],
    )

    with pytest.raises(ValueError, match="same number of series"):
        _merged = cluster + single_series


def test_fit_result_uses_all_observations_and_one_amplitude_per_series() -> None:
    cluster, _spectra, params = _axis_reproduction()
    normalized_residuals = residuals(params, cluster, noise=1.0)
    fit_result = FitResult(
        params=params,
        residual=normalized_residuals,
        cost=EXPECTED_CHI_SQUARED / 2.0,
        n_amplitude_params=cluster.n_amplitude_params,
    )

    n_observations = N_POINTS * N_SERIES
    n_varied_lineshape_params = len(params.get_vary_names())
    n_fitted_params = n_varied_lineshape_params + len(cluster.peaks) * N_SERIES
    expected_dof = n_observations - n_fitted_params

    assert normalized_residuals.shape == (n_observations,)
    assert fit_result.chisqr == pytest.approx(EXPECTED_CHI_SQUARED)
    assert fit_result.redchi == pytest.approx(EXPECTED_CHI_SQUARED / expected_dof)


def test_persisted_statistics_and_uncertainty_scaling_use_the_series_axis() -> None:
    cluster, spectra, params = _axis_reproduction()
    state = FittingState(
        clusters=[cluster],
        params=FitParameters.from_parameters(params, cluster.peaks),
        scalar_params=params,
        noise=1.0,
    )

    fit_results = build_fit_results(state, spectra, config={}, input_files={})
    statistics = fit_results.statistics[0]

    n_observations = N_POINTS * N_SERIES
    n_fitted_params = len(params.get_vary_names()) + len(cluster.peaks) * N_SERIES
    expected_dof = n_observations - n_fitted_params
    expected_redchi = EXPECTED_CHI_SQUARED / expected_dof

    assert (
        statistics.n_data,
        statistics.n_params,
        statistics.dof,
        statistics.reduced_chi_squared,
    ) == pytest.approx(
        (
            n_observations,
            n_fitted_params,
            expected_dof,
            expected_redchi,
        )
    )
    assert len(fit_results.clusters[0].amplitudes) == N_SERIES


def test_persisted_uncertainties_use_the_series_axis_reduced_chi_squared() -> None:
    cluster, spectra, params = _axis_reproduction()
    state = FittingState(
        clusters=[cluster],
        params=FitParameters.from_parameters(params, cluster.peaks),
        scalar_params=params,
        noise=1.0,
    )

    fit_results = build_fit_results(state, spectra, config={}, input_files={})
    shapes = cluster.evaluate(params)
    _amplitudes, base_errors, _covariance = calculate_amplitudes_with_uncertainty(
        shapes,
        cluster.corrected_data,
        noise=1.0,
    )
    n_observations = N_POINTS * N_SERIES
    n_fitted_params = len(params.get_vary_names()) + len(cluster.peaks) * N_SERIES
    expected_redchi = EXPECTED_CHI_SQUARED / (n_observations - n_fitted_params)

    assert [estimate.std_error for estimate in fit_results.clusters[0].amplitudes] == pytest.approx(
        [float(base_errors[0]) * np.sqrt(expected_redchi)] * N_SERIES
    )


def test_simulation_transposes_point_major_cluster_data_back_to_series_major_spectra() -> None:
    cluster, spectra, params = _axis_reproduction()

    shapes, amplitudes = calculate_shape_heights(params, cluster)
    simulated = simulate_data(params, [cluster], spectra.data)

    assert amplitudes.shape == (len(cluster.peaks), N_SERIES)
    assert simulated.shape == (N_SERIES, N_POINTS)
    assert simulated == pytest.approx((shapes.T @ amplitudes).T)


def test_simulation_rejects_cluster_and_spectrum_series_mismatch() -> None:
    cluster, spectra, params = _axis_reproduction()
    single_series = Cluster(
        cluster_id=cluster.cluster_id,
        peaks=cluster.peaks,
        grid_indices=cluster.grid_indices,
        data=cluster.data[:, :1],
    )

    with pytest.raises(ValueError, match="series count"):
        simulate_data(params, [single_series], spectra.data)


def test_fitting_state_rejects_the_development_axis_contract_version() -> None:
    cluster, _spectra, params = _axis_reproduction()
    state = FittingState(
        clusters=[cluster],
        params=FitParameters.from_parameters(params, cluster.peaks),
        scalar_params=params,
        noise=1.0,
    )
    assert state.version == "2.0"

    state_data = state.model_dump()
    state_data["version"] = "1.1"
    with pytest.raises(ValidationError, match="version") as error:
        FittingState.model_validate(state_data)

    message = str(error.value)
    assert "1.1" in message
    assert "2.0" in message
