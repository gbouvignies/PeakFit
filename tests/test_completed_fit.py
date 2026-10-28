"""Whole-dataset review uses terminal inputs and never refits completed values."""

from dataclasses import replace

import numpy as np
import pytest

from peakfit.engine.algorithms.evaluation import classify_optimizer_result
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import FitConfig
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.params_vector import FitParameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters
from peakfit.engine.domain.state import FittingState
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.engine.results import FitResult
from peakfit.fit.completed import CompletedFit
from peakfit.fit.pipeline import CorrectionSnapshot, PipelineCompletion, complete_fit
from peakfit.fit.simulation import project_final_outcome, project_final_residual


@pytest.fixture
def completed():
    """Disjoint irregular masks with overlapping Lorentzian tails and stale neighbors."""
    shape = (12, 16)
    axes = [
        SpectralParameters(
            size=size,
            sw=500,
            obs=500,
            car=0,
            aq_time=0.01,
            apocode=0,
            apodq1=0,
            apodq2=0,
            apodq3=0,
            p180=False,
            direct=direct,
            ft=True,
        )
        for size, direct in ((4, False), (12, False), (16, True))
    ]
    spectra = Spectra(
        dic={},
        data=np.zeros((4, *shape)),
        z_values=np.array([20.0, -5.0, 20.0, 1.0]),
        params=axes,
    )
    centers = ((4.3, 5.2), (5.1, 6.1), (7.4, 8.3), (9.0, 12.0))
    peaks = []
    for i, center in enumerate(centers):
        positions = [
            float(axis.pts2ppm(point)) for axis, point in zip(axes[1:], center, strict=True)
        ]
        name = f"p_{i}"
        peaks.append(
            Peak(
                name,
                np.array(positions),
                create_shapes(
                    spectra,
                    FitConfig(lineshape="lorentzian"),
                    name,
                    positions,
                    ["lorentzian", "lorentzian"],
                ),
            )
        )
    for peak, cluster_id in zip(peaks, (11, 11, 37, 91), strict=True):
        peak.set_cluster_id(cluster_id)
    params = Parameters.from_peaks(peaks)
    for parameter in params.values():
        if parameter.name.endswith(".lw"):
            parameter.value = 100
    grid = tuple(index.ravel() for index in np.indices(shape))
    shapes = Cluster.evaluate_peaks(peaks, params, grid)
    amplitudes = np.array([[6, -4, 2, 5], [3, -2, 4, 1], [-3, 5, -2, 4], [1, 1, 1, 1]])
    # Include small deterministic structure so recorded fit residuals are nonzero.
    observed = shapes.T @ amplitudes + 0.01 * np.sin(np.arange(192))[:, None]
    spectra.data[:] = observed.T.reshape(spectra.data.shape)
    clusters, results, evaluations = [], [], []
    for cluster_id, members, chosen in (
        (11, peaks[:2], grid[1] <= 6),
        (37, peaks[2:3], (grid[1] > 6) & (grid[1] <= 11)),
        (91, peaks[3:], grid[1] > 11),
    ):
        # Retain a deliberately non-raster sample order, with holes in the masks.
        rows = np.flatnonzero(chosen & (np.arange(192) % 5 != 0))[::-1]
        cluster = Cluster(cluster_id, members, [index[rows] for index in grid], observed[rows])
        member_indices = [peaks.index(p) for p in members]
        external = [i for i in range(4) if i not in member_indices]
        # A terminal correction is a previous-pass estimate, NOT final neighbors.
        cluster.corrections[:] = 0.8 * shapes[external][:, rows].T @ amplitudes[external]
        terminal_params = params.copy(deep=True)
        if cluster_id == 91:
            terminal_params["p_3.F2.cs"].__dict__["value"] = np.nan
        result = FitResult(
            cluster_id,
            terminal_params,
            np.zeros(cluster.n_observations),
            0,
            correction_revision=7,
            optimizer_kind="varpro",
            noise=0.25,
            success=cluster_id == 11,
            message=f"terminal {cluster_id}",
            n_amplitude_params=cluster.n_amplitude_params,
        )
        clusters.append(cluster)
        results.append(result)
        evaluations.append(classify_optimizer_result(cluster=cluster, result=result, noise=0.25))
    state = FittingState(
        clusters=clusters,
        params=FitParameters.from_parameters(params, peaks),
        scalar_params=params,
        noise=0.25,
    )
    correction = CorrectionSnapshot.from_clusters(clusters, 7)
    result = complete_fit(
        PipelineCompletion(
            state,
            results[::-1],
            evaluations[::-1],
            correction,
            8,
            7,
        )
    )
    return result, spectra, grid, centers


def _expected_model(completed, coordinates, planes):
    """Independent Lorentzian formula, including periodic indirect offsets."""
    result, spectra, _, centers = completed
    total = np.zeros((len(coordinates[0]), len(planes)))
    for terminal in result.final_outcome.clusters:
        if not terminal.usable:
            continue
        for row, name in enumerate(terminal.peak_names):
            center = centers[int(name.removeprefix("p_"))]
            shape = np.ones(len(coordinates[0]))
            for point, axis, position in zip(
                coordinates, spectra.spectral_params, center, strict=True
            ):
                offset = point - position
                if not axis.direct:
                    offset = (offset + axis.size / 2) % axis.size - axis.size / 2
                hz = offset * axis.sw / axis.size
                shape *= 50**2 / (50**2 + hz**2)
            total += shape[:, None] * terminal.analytical_evaluation.amplitudes[row, planes]
    return total


def test_complete_projection_preserves_signed_contributions_and_plane_order(completed, monkeypatch):
    result, _, grid, _ = completed
    outcome, snapshot = result.final_outcome, result.simulation_snapshot
    assert result.completed_fit.outcome is outcome
    assert result.completed_fit.model_snapshot is snapshot
    assert [c.classification.value for c in outcome.clusters] == [
        "converged",
        "usable_non_converged",
        "unusable",
    ]

    def forbidden(*_args, **_kwargs):
        pytest.fail("Completed projection must not fit or solve amplitudes")

    monkeypatch.setattr("peakfit.engine.algorithms.linear_algebra.factorize_shapes", forbidden)
    monkeypatch.setattr("peakfit.fit.pipeline.fit_single_cluster_task", forbidden)
    planes = [3, 0, 3, 1]
    actual = project_final_outcome(outcome, snapshot, grid, planes=planes)
    assert actual is not None
    np.testing.assert_allclose(actual, _expected_model(completed, grid, planes), atol=1e-12)
    assert actual.min() < 0 < actual.max()
    fractional = (np.array([4.1, 6.3, 7.5]), np.array([5.6, 7.2, 9.1]))
    predicted = project_final_outcome(outcome, snapshot, fractional, planes=planes)
    assert predicted is not None
    np.testing.assert_allclose(
        predicted,
        _expected_model(completed, fractional, planes),
        atol=1e-12,
    )
    assert project_final_outcome(outcome, snapshot, grid, cluster_id=91) is None


def test_display_residual_is_not_the_recorded_corrected_problem(completed):
    result, spectra, grid, _ = completed
    outcome, snapshot = result.final_outcome, result.simulation_snapshot
    planes = [2, 0, 2, 3]
    observed = spectra.data.reshape(4, -1).T[:, planes]
    expected = observed - _expected_model(completed, grid, planes)
    display = project_final_residual(outcome, snapshot, grid, observed, planes=planes)
    assert display is not None
    np.testing.assert_allclose(display, expected, atol=1e-12)
    normalized = project_final_residual(
        outcome,
        snapshot,
        grid,
        observed,
        planes=planes,
        normalize=True,
    )
    assert normalized is not None
    np.testing.assert_allclose(normalized, expected / outcome.noise, atol=1e-12)
    for samples in result.completed_fit.samples[:2]:
        evaluation = outcome.cluster(samples.cluster_id).analytical_evaluation
        np.testing.assert_array_equal(
            samples.data - samples.neighbor_corrections - evaluation.model_values,
            evaluation.raw_residuals,
        )
        flat = np.ravel_multi_index(samples.grid_indices, spectra.data.shape[1:])
        assert not np.allclose(display[flat], evaluation.raw_residuals[:, planes])
        np.testing.assert_allclose(
            evaluation.normalized_residuals,
            evaluation.raw_residuals.ravel() / outcome.noise,
        )
        assert evaluation.statistics.chi_squared == pytest.approx(
            np.sum(evaluation.normalized_residuals**2)
        )
    # Unsampled points still have a diagnostic display residual.
    assert np.isfinite(display[0]).all()
    with pytest.raises(ValueError, match="point-by-plane"):
        project_final_residual(outcome, snapshot, grid, observed.T, planes=planes)
    invalid = observed.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        project_final_residual(outcome, snapshot, grid, invalid, planes=planes)


def test_retained_inputs_are_owned_and_survive_continuation_mutation(completed):
    result, _, grid, _ = completed
    retained = result.completed_fit
    before = project_final_outcome(retained.outcome, retained.model_snapshot, grid)
    original = [
        (s.data.copy(), s.neighbor_corrections.copy(), s.grid_indices[0].copy())
        for s in retained.samples
    ]
    for source, frozen, (data, correction, index) in zip(
        result.continuation_state.clusters,
        retained.samples,
        original,
        strict=True,
    ):
        assert np.any(frozen.neighbor_corrections != 0)
        for values in (frozen.data, frozen.neighbor_corrections, *frozen.grid_indices):
            assert not values.flags.writeable
            with pytest.raises(ValueError, match="read-only"):
                values.flat[0] = 0
        source.data[:] = 900
        source.corrections[:] = 800
        source.grid_indices[0][:] = 0
        source.peaks[0].shapes[0].center += 100
        source.peaks[0].shapes[0]._grid.spec_params.sw *= 2
        np.testing.assert_array_equal(frozen.data, data)
        np.testing.assert_array_equal(frozen.neighbor_corrections, correction)
        np.testing.assert_array_equal(frozen.grid_indices[0], index)
    np.testing.assert_array_equal(
        project_final_outcome(retained.outcome, retained.model_snapshot, grid),
        before,
    )


def test_capture_rejects_same_revision_with_wrong_terminal_numerical_inputs(completed):
    result, _, _, _ = completed
    state = result.continuation_state
    # Equal revision numbers are insufficient: even jointly changed state/snapshot
    # corrections must still describe the recorded analytical residuals.
    state.clusters[0].corrections += 1
    correction = CorrectionSnapshot.from_clusters(state.clusters, 7)
    with pytest.raises(ValueError, match="Recorded residuals"):
        CompletedFit.capture(result.final_outcome, state, correction)


def test_all_unusable_display_has_no_fabricated_zero_model(completed):
    result, spectra, grid, _ = completed
    failed = result.final_outcome.cluster(91)
    outcome = replace(
        result.final_outcome,
        clusters=(failed,),
        by_cluster_id={91: failed},
        final_nonlinear_parameters=(),
    )
    snapshot = replace(
        result.simulation_snapshot,
        clusters=(result.simulation_snapshot.clusters[-1],),
        parameter_values=(),
    )
    assert (
        project_final_residual(
            outcome,
            snapshot,
            grid,
            spectra.data.reshape(4, -1).T,
        )
        is None
    )
