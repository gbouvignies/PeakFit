"""Shared optimizer preparation agrees with the local scientific preview."""

from pathlib import Path

import numpy as np
import pytest

from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.constraints import ParameterConstraint, PeakConstraints
from peakfit.engine.domain.fit_steps import FitStep
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.results import FitResult
from peakfit.fit.pipeline import ClusterFitTask, run_pipeline_iter
from peakfit.gui.fitting import prepare_local_fit
from peakfit.gui.peaks import WorkingPeak
from peakfit.io.readers.spectrum import read_spectra

DATA = Path(__file__).resolve().parents[1] / "examples" / "02-advanced-fitting" / "data"


def test_preview_matches_submitted_parameters_and_continuation(monkeypatch) -> None:  # noqa: PLR0915
    spectra = read_spectra(DATA / "pseudo3d.ft2")
    peak = WorkingPeak.new("same.assignment", (120.0, 8.5))
    # Test multiple existing model families, including model-specific parameters.
    for model in ("auto", "pvoigt", "lorentzian_doublet"):
        config = PeakFitConfig()
        config.fitting.lineshape = model
        config.parameters.defaults.patterns["*.*.lw"] = ParameterConstraint(
            value=35, min=10, max=80, vary=False
        )
        config.parameters.defaults.patterns["*.F2.cs"] = ParameterConstraint(vary=False)
        prepared = prepare_local_fit(spectra, (peak,), peak.id, config)
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
                    [prepared.cluster],
                    prepared.noise,
                    Parameters.from_peaks(prepared.cluster.peaks),
                    prepared.cluster.peaks,
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
            prepare_local_fit(spectra, (peak,), peak.id, settings)

    # A configured start must be used once; explicit later release continues from
    # the preceding fit rather than resetting to 35 Hz. No optimizer is executed.
    config.fitting.lineshape = "gaussian"
    prepared = prepare_local_fit(spectra, (peak,), peak.id, config)
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
        [prepared.cluster],
        prepared.noise,
        Parameters.from_peaks(prepared.cluster.peaks),
        prepared.cluster.peaks,
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
    with pytest.raises(ValueError, match="one isolated"):
        prepare_local_fit(spectra, (peak,), peak.id, config)
