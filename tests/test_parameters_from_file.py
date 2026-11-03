"""Starting values from a previous fit (``[parameters].from_file``)."""

import json

import pytest
from builders import two_cluster_dataset

from peakfit.engine.domain.constraints import ParameterConstraint
from peakfit.fit.pipeline import run_pipeline, run_pipeline_iter
from peakfit.io.output_metadata import RunMetadata
from peakfit.io.writers.json import write_final_outcome_summary


def _pipeline(config, prepared, executor=None):
    return run_pipeline_iter(
        config,
        prepared.clusters,
        prepared.noise,
        prepared.parameters,
        [p for c in prepared.clusters for p in c.peaks],
        executor=executor,
    )


def _first_pass_starts(config, prepared):
    """The nonlinear values submitted to the optimizer in the first pass."""
    submitted = []

    def capture(_function, tasks):
        submitted.extend(tasks)
        return iter(())

    # Stop before any optimizer runs: only the submitted starts matter here.
    with pytest.raises(ValueError, match="Missing optimizer result"):
        list(_pipeline(config, prepared, capture))
    return {name: p.value for task in submitted for name, p in task.params.items()}


def _previous_fit(tmp_path):
    spectra, _peaks, config, prepared = two_cluster_dataset()
    pipeline = run_pipeline(
        config,
        prepared.clusters,
        prepared.noise,
        prepared.parameters,
        [p for c in prepared.clusters for p in c.peaks],
    )
    path = write_final_outcome_summary(
        pipeline.final_outcome,
        metadata=RunMetadata(),
        z_values=spectra.z_values,
        path=tmp_path / "summary" / "fit.json",
    )
    return path, config, prepared


def test_previous_fitted_nonlinear_values_become_the_starts(tmp_path) -> None:
    path, config, prepared = _previous_fit(tmp_path)
    fitted = {
        p["name"]: p["value"]
        for p in json.loads(path.read_text(encoding="utf-8"))["final_nonlinear_parameters"]
    }
    defaults = _first_pass_starts(config, prepared)
    assert set(fitted) == set(defaults)
    # The test is meaningful only if the previous fit moved away from the defaults.
    assert any(abs(fitted[name] - defaults[name]) > 1e-6 for name in fitted)

    config.parameters.from_file = path
    assert _first_pass_starts(config, prepared) == fitted


def test_configured_values_take_precedence_over_previous_values(tmp_path) -> None:
    path, config, prepared = _previous_fit(tmp_path)
    config.parameters.from_file = path
    config.parameters.defaults.patterns["*.*.lw"] = ParameterConstraint(value=30.0)
    widths = [v for name, v in _first_pass_starts(config, prepared).items() if name.endswith(".lw")]
    assert widths
    assert all(value == 30.0 for value in widths)


def test_unusable_previous_summary_fails_with_actionable_message(tmp_path) -> None:
    path, config, prepared = _previous_fit(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))

    # The pre-4.0 layout this option used to read is rejected, not silently ignored.
    old = tmp_path / "old.json"
    old.write_text(
        json.dumps({"clusters": [{"parameters": [{"name": "p_0.F2.cs", "value": 1.0}]}]}),
        encoding="utf-8",
    )
    config.parameters.from_file = old
    with pytest.raises(ValueError, match=r"(?s)not a PeakFit fit summary.*summary/fit\.json"):
        next(_pipeline(config, prepared))

    for parameters in (
        document["final_nonlinear_parameters"],
        *(cluster["final_nonlinear_parameters"] for cluster in document["clusters"]),
    ):
        for parameter in parameters:
            parameter["name"] = f"other.{parameter['name']}"
    unrelated = tmp_path / "unrelated.json"
    unrelated.write_text(json.dumps(document), encoding="utf-8")
    config.parameters.from_file = unrelated
    with pytest.raises(ValueError, match="shares no parameter name"):
        next(_pipeline(config, prepared))

    config.parameters.from_file = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match=r"parameters\.from_file not found"):
        next(_pipeline(config, prepared))
