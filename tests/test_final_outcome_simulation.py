"""Contracts for full-grid simulation from immutable final fit outcomes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

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
from peakfit.engine.fitting.simulation import simulate_data
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.engine.results import FitResult
from peakfit.fit.final_outcome import FinalFitOutcome, finalize_fit
from peakfit.fit.pipeline import CorrectionSnapshot, PipelineResult
from peakfit.fit.simulation import FinalModelSnapshot, simulate_final_outcome
from peakfit.io.writers.run_files import write_simulated_spectra

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.shared.typing import FloatArray


@dataclass(frozen=True)
class SimulationFixture:
    outcome: FinalFitOutcome
    snapshot: FinalModelSnapshot
    state: FittingState
    spectra: Spectra
    clusters: tuple[Cluster, ...]
    params: Parameters


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


def _fixture(
    states: dict[int, tuple[bool, bool]],
    *,
    grid_shape: tuple[int, ...] = (9,),
    n_series: int = 2,
    reverse_completion: bool = False,
) -> SimulationFixture:
    spectra = Spectra(
        dic={},
        data=np.zeros((n_series, *grid_shape), dtype=np.float64),
        z_values=np.arange(n_series, dtype=np.float64),
        params=[
            _spectral_parameters(size=n_series, direct=False),
            *(
                _spectral_parameters(size=size, direct=index == len(grid_shape) - 1)
                for index, size in enumerate(grid_shape)
            ),
        ],
    )
    coordinates = [axis.ravel() for axis in np.indices(grid_shape, dtype=np.intp)]
    clusters: list[Cluster] = []
    peaks: list[Peak] = []
    results: list[FitResult] = []
    evaluations = []
    for index, cluster_id in enumerate(states, start=1):
        positions = np.array(
            [
                spectra.spectral_params[dimension].pts2ppm(
                    min(index + 1, grid_shape[dimension] - 1)
                )
                for dimension in range(len(grid_shape))
            ],
            dtype=np.float64,
        )
        peak = Peak(
            name=f"P{cluster_id}",
            positions=positions,
            shapes=create_shapes(
                spectra,
                FitConfig(lineshape="gaussian"),
                f"P{cluster_id}",
                [float(position) for position in positions],
                ["gaussian"] * len(grid_shape),
            ),
        )
        peak.set_cluster_id(cluster_id)
        params = peak.create_params()
        empty = Cluster(
            cluster_id=cluster_id,
            peaks=[peak],
            grid_indices=coordinates,
            data=np.zeros((len(coordinates[0]), n_series), dtype=np.float64),
        )
        amplitudes = np.array([[2.0 * index, 3.0 * index]][:n_series], dtype=np.float64)
        if n_series != 2:
            amplitudes = np.arange(1, n_series + 1, dtype=np.float64)[None, :] * index
        cluster = Cluster(
            cluster_id=cluster_id,
            peaks=[peak],
            grid_indices=coordinates,
            data=empty.evaluate(params).T @ amplitudes,
        )
        converged, usable = states[cluster_id]
        result_params = params.copy(deep=True)
        if not usable:
            result_params[result_params.get_vary_names()[0]].__dict__["value"] = np.nan
        result = FitResult(
            cluster_id=cluster_id,
            params=result_params,
            residual=np.zeros(cluster.n_observations, dtype=np.float64),
            cost=0.0,
            n_amplitude_params=cluster.n_amplitude_params,
            correction_revision=3,
            success=converged,
            message=f"terminal {cluster_id}",
            optimizer_kind="varpro",
            noise=1.0,
        )
        clusters.append(cluster)
        peaks.append(peak)
        results.append(result)
        evaluations.append(classify_optimizer_result(cluster=cluster, result=result, noise=1.0))

    params = Parameters.from_peaks(peaks, fixed=False)
    state = FittingState(
        clusters=clusters,
        params=FitParameters.from_parameters(params, peaks),
        scalar_params=params,
        noise=1.0,
    )
    correction_snapshot = CorrectionSnapshot(
        revision=3,
        corrections=MappingProxyType(
            {cluster.cluster_id: cluster.corrections.copy() for cluster in clusters}
        ),
    )
    if reverse_completion:
        results.reverse()
        evaluations.reverse()
    outcome = finalize_fit(
        PipelineResult(
            state=state,
            results=results,
            evaluations=evaluations,
            correction_snapshot=correction_snapshot,
            n_optimizer_passes=1,
        )
    )
    return SimulationFixture(
        outcome=outcome,
        snapshot=FinalModelSnapshot.capture(outcome, state, correction_snapshot),
        state=state,
        spectra=spectra,
        clusters=tuple(clusters),
        params=params,
    )


def test_simulation_uses_the_final_outcome_amplitudes_without_resolving_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture({11: (True, True)})

    def fail_amplitude_solve(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Completed simulation must not solve amplitudes.")

    monkeypatch.setattr(
        "peakfit.engine.algorithms.linear_algebra.calculate_amplitudes_with_uncertainty",
        fail_amplitude_solve,
    )

    simulated = simulate_final_outcome(fixture.outcome, fixture.snapshot, fixture.spectra.data)

    evaluation = fixture.outcome.cluster(11).analytical_evaluation
    assert evaluation is not None
    assert simulated is not None
    np.testing.assert_allclose(simulated, (evaluation.shapes.T @ evaluation.amplitudes).T)


def test_simulation_preserves_the_previous_finite_full_grid_projection() -> None:
    fixture = _fixture({11: (True, True), 37: (False, True)})

    simulated = simulate_final_outcome(fixture.outcome, fixture.snapshot, fixture.spectra.data)
    previous = simulate_data(fixture.params, fixture.clusters, fixture.spectra.data)

    assert simulated is not None
    np.testing.assert_allclose(simulated, previous)


@pytest.mark.parametrize(
    ("states", "expected_ids"),
    [
        ({11: (True, True)}, [11]),
        ({37: (False, True)}, [37]),
        ({11: (False, False)}, []),
        ({91: (False, False), 11: (True, True), 37: (False, True)}, [11, 37]),
    ],
)
def test_simulation_uses_exactly_usable_nonconsecutive_outcomes(
    states: dict[int, tuple[bool, bool]], expected_ids: list[int]
) -> None:
    fixture = _fixture(states)

    simulated = simulate_final_outcome(fixture.outcome, fixture.snapshot, fixture.spectra.data)

    if not expected_ids:
        assert simulated is None
        return
    assert simulated is not None
    expected = np.zeros_like(simulated)
    for cluster_id in expected_ids:
        cluster = fixture.outcome.cluster(cluster_id)
        evaluation = cluster.analytical_evaluation
        assert evaluation is not None
        expected += (evaluation.shapes.T @ evaluation.amplitudes).T
    np.testing.assert_allclose(simulated, expected)


def test_simulation_is_deterministic_for_reverse_completion_order() -> None:
    fixture = _fixture({11: (True, True), 37: (False, True), 91: (False, False)})
    reversed_completion = _fixture(
        {11: (True, True), 37: (False, True), 91: (False, False)},
        reverse_completion=True,
    )

    first = simulate_final_outcome(fixture.outcome, fixture.snapshot, fixture.spectra.data)
    second = simulate_final_outcome(
        reversed_completion.outcome,
        reversed_completion.snapshot,
        reversed_completion.spectra.data,
    )

    assert first is not None
    assert second is not None
    np.testing.assert_allclose(first, second)


def test_simulation_supports_multidimensional_grid_indices_and_multiple_series() -> None:
    fixture = _fixture({11: (True, True)}, grid_shape=(3, 4), n_series=3)

    simulated = simulate_final_outcome(fixture.outcome, fixture.snapshot, fixture.spectra.data)

    assert simulated is not None
    assert simulated.shape == (3, 3, 4)


def test_simulation_rejects_output_series_and_amplitude_shape_mismatches() -> None:
    fixture = _fixture({11: (True, True)})
    bad_series = np.zeros((1, fixture.spectra.data.shape[1]), dtype=np.float64)

    with pytest.raises(ValueError, match="series count"):
        simulate_final_outcome(fixture.outcome, fixture.snapshot, bad_series)

    malformed = replace(
        fixture.snapshot.clusters[0],
        grid_index_shapes=((fixture.snapshot.clusters[0].n_points - 1,),),
    )
    malformed_snapshot = replace(fixture.snapshot, clusters=(malformed,))
    with pytest.raises(ValueError, match="grid-index shape"):
        simulate_final_outcome(fixture.outcome, malformed_snapshot, fixture.spectra.data)

    evaluation = fixture.outcome.cluster(11).analytical_evaluation
    assert evaluation is not None
    malformed_cluster = replace(
        fixture.outcome.cluster(11),
        analytical_evaluation=replace(
            evaluation,
            amplitudes=np.zeros((2, fixture.spectra.data.shape[0]), dtype=np.float64),
        ),
    )
    malformed_outcome = replace(
        fixture.outcome,
        clusters=(malformed_cluster,),
        by_cluster_id=MappingProxyType({11: malformed_cluster}),
    )
    with pytest.raises(ValueError, match="amplitude shape"):
        simulate_final_outcome(malformed_outcome, fixture.snapshot, fixture.spectra.data)


def test_simulation_rejects_stale_snapshot_and_leaves_outcome_arrays_immutable() -> None:
    fixture = _fixture({11: (True, True)})
    evaluation = fixture.outcome.cluster(11).analytical_evaluation
    assert evaluation is not None
    amplitudes = evaluation.amplitudes.copy()
    before = simulate_final_outcome(fixture.outcome, fixture.snapshot, fixture.spectra.data)
    fixture.state.scalar_params[fixture.state.scalar_params.get_vary_names()[0]].value += 0.01
    fixture.state.clusters[0].peaks[0].positions += 100.0

    simulated = simulate_final_outcome(fixture.outcome, fixture.snapshot, fixture.spectra.data)

    assert simulated is not None
    assert before is not None
    np.testing.assert_allclose(simulated, before)
    np.testing.assert_allclose(evaluation.amplitudes, amplitudes)
    assert evaluation.amplitudes.flags.writeable is False

    with pytest.raises(ValueError, match="nonlinear parameter"):
        FinalModelSnapshot.capture(
            fixture.outcome,
            fixture.state,
            CorrectionSnapshot(
                revision=3,
                corrections=MappingProxyType(
                    {
                        cluster.cluster_id: cluster.corrections.copy()
                        for cluster in fixture.state.clusters
                    }
                ),
            ),
        )


def test_simulation_rejects_snapshot_with_wrong_terminal_revision() -> None:
    fixture = _fixture({11: (True, True)})
    stale = FinalModelSnapshot(
        clusters=fixture.snapshot.clusters,
        noise=fixture.snapshot.noise,
        correction_revision=fixture.snapshot.correction_revision + 1,
    )

    with pytest.raises(ValueError, match="correction revision"):
        simulate_final_outcome(fixture.outcome, stale, fixture.spectra.data)


def test_simulation_rejects_grid_dimension_mismatch() -> None:
    fixture = _fixture({11: (True, True)})

    with pytest.raises(ValueError, match="grid dimension"):
        simulate_final_outcome(
            fixture.outcome,
            fixture.snapshot,
            np.zeros((2, 3, 3), dtype=np.float64),
        )


def test_simulated_writer_consumes_outcome_and_snapshot_without_legacy_reconstruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture({11: (True, True)})
    written: list[tuple[str, FloatArray]] = []

    class FakePipe:
        def write(self, path: str, _dic: object, data: FloatArray, *, overwrite: bool) -> None:
            assert overwrite is True
            written.append((path, data.copy()))

    class FakeNmrglue:
        pipe = FakePipe()

    def fail_legacy_reconstruction(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(
            "Completed simulation must not reconstruct FitResults from FittingState."
        )

    monkeypatch.setattr("peakfit.fit.results.build_fit_results", fail_legacy_reconstruction)
    monkeypatch.setattr("peakfit.io.writers.run_files.import_module", lambda _name: FakeNmrglue())
    simulated = simulate_final_outcome(fixture.outcome, fixture.snapshot, fixture.spectra.data)
    assert simulated is not None

    path = write_simulated_spectra(
        tmp_path,
        fixture.spectra,
        simulated,
    )

    assert path is not None
    assert written
