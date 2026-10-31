"""Desktop dataset-fit preparation agrees with the shared optimizer pipeline."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from peakfit.engine.algorithms.clustering import create_clusters
from peakfit.engine.algorithms.noise import estimate_noise, prepare_noise_level
from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.constraints import ParameterConstraint, PeakConstraints
from peakfit.engine.domain.fit_steps import FitStep
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.spectrum import get_shape_names
from peakfit.engine.results import FitResult
from peakfit.fit.pipeline import ClusterFitTask, run_pipeline_iter
from peakfit.gui.fitting import PreparedDatasetFit, prepare_dataset_fit
from peakfit.gui.peaks import WorkingPeak
from peakfit.gui.spectrum import load_peaks, load_spectrum
from peakfit.io.readers.peaks import read_list
from peakfit.io.readers.spectrum import read_spectra

DATA = Path(__file__).resolve().parents[1] / "examples" / "02-advanced-fitting" / "data"


def _prepare(spectra, peak, config, contour_level) -> PreparedDatasetFit:
    settings = config.model_copy(deep=True)
    settings.clustering.contour_level = contour_level
    return prepare_dataset_fit(spectra, (peak,), settings)


def test_preparation_matches_submitted_parameters_and_continuation(monkeypatch) -> None:  # noqa: PLR0915
    spectra = read_spectra(DATA / "pseudo3d.ft2")
    contour_level = 5 * prepare_noise_level(None, spectra)
    peak = WorkingPeak.new("same.assignment", (120.0, 8.5))
    # Test multiple existing model families, including model-specific parameters.
    for model in ("auto", "pvoigt", "lorentzian_doublet"):
        config = PeakFitConfig()
        config.fitting.lineshape = model
        config.parameters.defaults.patterns["*.*.lw"] = ParameterConstraint(
            value=35, min=10, max=80, vary=False
        )
        config.parameters.defaults.patterns["*.F2.cs"] = ParameterConstraint(vary=False)
        prepared = _prepare(spectra, peak, config, contour_level)
        (cluster,) = prepared.clusters
        assert not prepared.parameters[f"{prepared.identities[0].engine_name}.F2.cs"].vary
        widths = [
            p
            for p in prepared.parameters.values()
            if p.param_id is not None and p.param_id.label == "lw"
        ]
        assert all((p.value, p.min, p.max, p.vary) == (35, 10, 80, False) for p in widths)
        assert any(p.vary for p in prepared.parameters.values())
        submitted = []

        def capture(_function, tasks, submitted=submitted):
            submitted.extend(tasks)
            return iter(())

        # Capture the actual pipeline task before any numerical execution.
        with pytest.raises(ValueError, match="Missing optimizer result"):
            list(
                run_pipeline_iter(
                    config,
                    [cluster],
                    prepared.noise,
                    Parameters.from_peaks(cluster.peaks),
                    cluster.peaks,
                    executor=capture,
                )
            )
        task: ClusterFitTask = submitted[0]
        assert task.params.model_dump() == prepared.parameters.model_dump()
        assert task.correction_revision == 0
        assert not np.any(task.cluster.corrections)
        if model == "pvoigt":
            assert any(
                p.param_id is not None and p.param_id.label == "eta"
                for p in prepared.parameters.values()
            )
        if model.endswith("doublet"):
            assert any(
                p.param_id is not None and p.param_id.label == "j"
                for p in prepared.parameters.values()
            )

    # Engine-safe renaming must never silently omit assignment-based settings,
    # including rules that would accidentally match generated names or files.
    unsupported = []
    per_peak = PeakFitConfig()
    per_peak.parameters.peaks[peak.name] = PeakConstraints(position_window=0.01)
    unsupported.append(per_peak)
    for pattern in ("same.assignment.*.lw", "*assignment.*.lw", "p_0.*.lw"):
        targeted = PeakFitConfig()
        targeted.parameters.defaults.patterns[pattern] = ParameterConstraint(vary=False)
        unsupported.append(targeted)
    for step in (FitStep(fix=[f"{peak.name}.*.cs"]), FitStep(vary=[f"{peak.name}.*.cs"])):
        targeted_step = PeakFitConfig()
        targeted_step.fitting.steps = [step]
        unsupported.append(targeted_step)
    from_file = PeakFitConfig()
    from_file.parameters.from_file = DATA / "previous-fit.json"
    unsupported.append(from_file)
    for settings in unsupported:
        with pytest.raises(ValueError, match=r"assignment.*not supported.*wildcard"):
            _prepare(spectra, peak, settings, contour_level)

    # A configured start must be used once; explicit later release continues from
    # the preceding fit rather than resetting to 35 Hz. No optimizer is executed.
    config.fitting.lineshape = "gaussian"
    prepared = _prepare(spectra, peak, config, contour_level)
    (cluster,) = prepared.clusters
    config.fitting.steps = [FitStep(name="start"), FitStep(name="release", vary=["*.*.lw"])]
    captured = []

    def advance(task):
        captured.append(task.params.copy(deep=True))
        for parameter in task.params.values():
            if parameter.param_id.label == "lw":
                parameter.value = 42

        return FitResult(
            cluster_id=task.cluster_id,
            params=task.params,
            residual=np.zeros(task.cluster.n_observations),
            cost=0,
            nfev=1,
            success=True,
            message="test capture",
            n_amplitude_params=task.cluster.n_amplitude_params,
        )

    monkeypatch.setattr("peakfit.fit.pipeline.fit_single_cluster_task", advance)
    # Finalization is irrelevant here; stop just after both tasks are submitted.
    items = run_pipeline_iter(
        config,
        [cluster],
        prepared.noise,
        Parameters.from_peaks(cluster.peaks),
        cluster.peaks,
    )
    while len(captured) < 2:
        next(items)
    assert [
        (p.value, p.vary)
        for p in captured[1].values()
        if p.param_id is not None and p.param_id.label == "lw"
    ] == [(42, True), (42, True)]
    assert all(
        p.value == 35 and not p.vary
        for p in captured[0].values()
        if p.param_id is not None and p.param_id.label == "lw"
    )


def test_dataset_cluster_identity_aliasing_and_direct_axis_guard() -> None:
    spectra = load_spectrum(DATA / "pseudo3d.ft2", DATA / "b1_offsets.txt")
    peaks = tuple(load_peaks(DATA / "pseudo3d.list", spectra))
    config = PeakFitConfig()
    # Use the existing reader + engine clustering as the reference, independently
    # of the GUI's identity conversion. Select a real overlapping group.
    reference_peaks = read_list(
        DATA / "pseudo3d.list", spectra, get_shape_names(config.fitting, spectra), config.fitting
    )
    noise = prepare_noise_level(None, spectra)
    contour_level = 5 * estimate_noise(spectra.data[0])
    groups = create_clusters(spectra, reference_peaks, contour_level)
    high_groups = create_clusters(spectra, reference_peaks, 1000 * estimate_noise(spectra.data[0]))
    high_sizes = {peak.name: len(group.peaks) for group in high_groups for peak in group.peaks}
    reference = next(
        group for group in groups if high_sizes[group.peaks[0].name] < len(group.peaks)
    )
    selected = next(peak for peak in peaks if peak.name == reference.peaks[0].name)
    expected_ids = {peak.id for peak in peaks if peak.name in {p.name for p in reference.peaks}}
    # Rename two members to the same assignment, preserving separate identities;
    # unfold one indirect position by exactly a spectral width.
    y_axis, x_axis = spectra.spectral_params
    duplicate_id = next(identity for identity in expected_ids if identity != selected.id)
    peaks = tuple(
        replace(
            peak,
            name="Duplicate",
            positions=(peak.positions[0] + y_axis.delta * y_axis.size, peak.positions[1]),
        )
        if peak.id == selected.id
        else replace(peak, name="Duplicate")
        if peak.id == duplicate_id
        else peak
        for peak in peaks
    )
    config.clustering.contour_level = contour_level
    prepared = prepare_dataset_fit(spectra, peaks, config)
    assert {item.working_id for item in prepared.identities} == {peak.id for peak in peaks}
    assert len({item.engine_name for item in prepared.identities}) == len(peaks)
    assert sum(item.assignment == "Duplicate" for item in prepared.identities) == 2
    identity = next(item for item in prepared.identities if item.working_id == selected.id)
    cluster = next(
        c for c in prepared.clusters if identity.engine_name in {p.name for p in c.peaks}
    )
    engine_ids = {item.engine_name: item.working_id for item in prepared.identities}
    assert {engine_ids[peak.name] for peak in cluster.peaks} == expected_ids
    for actual, expected in zip(cluster.grid_indices, reference.grid_indices, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(cluster.data, reference.data)
    assert cluster.n_series == spectra.data.shape[0]
    assert prepared.noise == noise
    assert prepared.contour_level == contour_level
    assert prepared.snapshot.contour_level == contour_level
    scientific = next(peak for peak in cluster.peaks if peak.name == identity.engine_name)
    working = next(peak for peak in peaks if peak.id == selected.id)
    np.testing.assert_array_equal(scientific.positions, working.positions)
    # The engine's indirect alias sign must be retained as well as folded indices.
    original = reference.peaks[0].shapes[0]
    offsets, signs = scientific.shapes[0].dim_ctx.compute_offsets(
        np.array([0, y_axis.size - 1]), np.array([working.positions[0]])
    )
    old_offsets, old_signs = original.dim_ctx.compute_offsets(
        np.array([0, y_axis.size - 1]), np.array([selected.positions[0]])
    )
    np.testing.assert_allclose(offsets, old_offsets)
    np.testing.assert_array_equal(signs, -old_signs if y_axis.p180 else old_signs)
    outside = replace(peaks[0], positions=(peaks[0].positions[0], x_axis.pts2ppm(x_axis.size)))
    with pytest.raises(ValueError, match="direct-axis position"):
        prepare_dataset_fit(spectra, (outside, *peaks[1:]), config)
