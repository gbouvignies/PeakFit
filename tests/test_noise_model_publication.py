"""Published sandwich errors and their noise-model record: docs/correlated-noise.md §5.

Fitted values never depend on the noise model; only the published errors do. The
record travels with every format and reopens exactly as written.
"""

import importlib
import json
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from peakfit.engine.algorithms.linear_algebra import (
    correlated_amplitude_covariance,
    factorize_shapes,
)
from peakfit.engine.algorithms.noise_correlation import (
    AxisProcessing,
    cluster_correlation,
    not_modelled,
)
from peakfit.engine.algorithms.varpro import VarProOptimizer, _calculate_errors
from peakfit.engine.domain.config import FitConfig, PeakFitConfig
from peakfit.engine.scientific.sampling import sampling_problem
from peakfit.engine.scientific.summary import RunSummary
from peakfit.fit.pipeline import run_pipeline
from peakfit.io.output_metadata import RunMetadata
from peakfit.io.readers.spectrum import read_noise_model
from peakfit.io.schemas import FitSummarySchema
from peakfit.io.scientific import read_scientific, write_scientific
from peakfit.io.writers.config import WriterConfig
from peakfit.io.writers.orchestrator import write_fit_outputs
from peakfit.shared.exceptions import ScientificIntegrityError


@pytest.fixture(scope="module")
def tool():
    path = str(Path(__file__).resolve().parents[1] / "tools")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(path)
        yield importlib.import_module("validate_sandwich_errors")


def _fit(tool, case, noise_model=None):
    spectra, cluster, truth, _ = tool.build(case, np.random.default_rng(11))
    model = read_noise_model(spectra.dic, spectra.data.shape)
    config = PeakFitConfig(fitting=FitConfig(lineshape="sp2"))
    completed = run_pipeline(
        config, [cluster], tool.SIGMA, truth.copy(), noise_model=noise_model or model
    ).completed_fit
    return completed, model


@pytest.fixture(scope="module")
def modelled(tool):
    return _fit(tool, "overlapping pair")


@pytest.fixture(scope="module")
def independent(tool):
    return _fit(tool, "overlapping pair", not_modelled("synthetic comparison"))


def test_a_modelled_fit_publishes_the_sandwich(modelled):
    completed, model = modelled
    assert model.modelled
    assert completed.outcome.noise_model == model
    problem = sampling_problem(completed, 1)
    correlation = cluster_correlation(model, problem.cluster.grid_indices)
    optimizer = VarProOptimizer(
        problem.cluster, list(problem.names), problem.parameters, completed.outcome.noise
    )
    published = np.array([problem.parameters[n].stderr for n in problem.names])
    np.testing.assert_allclose(
        published, _calculate_errors(optimizer, problem.start, correlation), rtol=1e-12
    )
    evaluation = completed.outcome.cluster(1).analytical_evaluation
    projection = factorize_shapes(problem.cluster.evaluate(problem.parameters))
    sandwich = completed.outcome.noise**2 * correlated_amplitude_covariance(projection, correlation)
    np.testing.assert_allclose(evaluation.amplitude_covariance, sandwich, rtol=1e-12)
    np.testing.assert_allclose(
        evaluation.amplitude_standard_errors, np.sqrt(np.diag(sandwich)), rtol=1e-12
    )


def test_the_noise_model_changes_errors_but_never_fitted_values(modelled, independent):
    a, b = modelled[0].outcome, independent[0].outcome
    for p, q in zip(a.final_nonlinear_parameters, b.final_nonlinear_parameters, strict=True):
        assert p.value == q.value
        assert p.standard_error > q.standard_error  # zero fill ×4 and a sine bell
    ea, eb = a.cluster(1).analytical_evaluation, b.cluster(1).analytical_evaluation
    np.testing.assert_array_equal(ea.amplitudes, eb.amplitudes)
    assert ea.statistics == eb.statistics  # χ²ᵣ keeps its definition
    assert np.all(ea.amplitude_standard_errors > eb.amplitude_standard_errors)


def test_processing_that_leaves_points_independent_publishes_today_s_errors(tool):
    # No zero fill, no apodization: the header models R = I exactly.
    completed, model = _fit(tool, "white-noise control (R = I)")
    assert model.modelled
    assert all(a.window == "none" and a.acquired == a.transform for a in model.axes)
    reference, _ = _fit(tool, "white-noise control (R = I)", not_modelled("comparison"))
    for p, q in zip(
        completed.outcome.final_nonlinear_parameters,
        reference.outcome.final_nonlinear_parameters,
        strict=True,
    ):
        assert p.value == q.value
        assert p.standard_error == pytest.approx(q.standard_error, rel=1e-10)
    np.testing.assert_allclose(
        completed.outcome.cluster(1).analytical_evaluation.amplitude_standard_errors,
        reference.outcome.cluster(1).analytical_evaluation.amplitude_standard_errors,
        rtol=1e-10,
    )


@pytest.mark.parametrize("which", ["modelled", "independent"])
def test_the_archive_keeps_the_noise_model_and_errors_exactly(which, request, tmp_path):
    completed, _ = request.getfixturevalue(which)
    path = write_scientific(tmp_path / "scientific.npz", completed.outcome, completed.samples)
    outcome, _, _ = read_scientific(path)
    assert outcome.noise_model == completed.outcome.noise_model
    for a, b in zip(completed.outcome.clusters, outcome.clusters, strict=True):
        assert a.final_nonlinear_parameters == b.final_nonlinear_parameters
        np.testing.assert_array_equal(
            a.analytical_evaluation.amplitude_covariance,
            b.analytical_evaluation.amplitude_covariance,
        )


def _rewrite(source: Path, target: Path, change) -> Path:
    with ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    document = json.loads(members["metadata.json"])
    change(document["outcome"]["noise_model"])
    members["metadata.json"] = json.dumps(document).encode()
    with ZipFile(target, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return target


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.update(reason="contradicts the axes"),  # modelled and a reason
        lambda m: m.update(axes=m["axes"][:1]),  # one axis for 2D samples
        lambda m: m.update(axes=[]),  # neither axes nor a reason
        lambda m: m["axes"][0].update(window="exponential"),
        lambda m: m["axes"][0].update(extra=1.0),
        lambda m: m["axes"][0].pop("filter_delay"),  # no default for recorded provenance
        lambda m: m.pop("axes"),
    ],
)
def test_a_malformed_noise_model_is_refused(modelled, tmp_path, change):
    completed, _ = modelled
    path = write_scientific(tmp_path / "scientific.npz", completed.outcome, completed.samples)
    damaged = _rewrite(path, tmp_path / "damaged.npz", change)
    with pytest.raises(ScientificIntegrityError):
        read_scientific(damaged)


def test_exports_state_the_noise_model(modelled, independent, tmp_path):
    for (completed, _), label in ((modelled, "modelled"), (independent, "not modelled")):
        written = write_fit_outputs(
            completed.outcome,
            tmp_path / label,
            WriterConfig(formats=("csv", "json", "txt")),
            metadata=RunMetadata(),
            summary=RunSummary.from_outcome(completed.outcome),
        )
        document = json.loads(written["summary_fit"].read_text())
        record = completed.outcome.noise_model
        assert document["noise_model"]["modelled"] == record.modelled
        assert document["noise_model"]["reason"] == record.reason
        assert len(document["noise_model"]["axes"]) == len(record.axes)
        FitSummarySchema.model_validate(document)
        for table in ("parameters", "intensities"):
            rows = written[table].read_text().splitlines()
            column = rows[0].split(",").index("noise_correlation")
            assert {row.split(",")[column] for row in rows[1:]} == {label}
        report = written["report"].read_text()
        assert record.description in report


def test_the_json_schema_refuses_an_inconsistent_noise_model(modelled, tmp_path):
    written = write_fit_outputs(
        modelled[0].outcome,
        tmp_path,
        WriterConfig(formats=("json",)),
        metadata=RunMetadata(),
    )
    document = json.loads(written["summary_fit"].read_text())
    document["noise_model"]["reason"] = "contradicts the axes"
    with pytest.raises(ValueError, match="modelled with axes"):
        FitSummarySchema.model_validate(document)


def test_an_axis_count_disagreeing_with_the_samples_is_refused(modelled, tmp_path):
    completed, model = modelled
    one_axis = replace(model, axes=model.axes[:1])
    outcome = replace(completed.outcome, noise_model=one_axis)
    with pytest.raises(ValueError, match="sample coordinates"):
        write_scientific(tmp_path / "scientific.npz", outcome, completed.samples)


def test_descriptions_name_the_axes_or_the_reason():
    axis = AxisProcessing(92, 256, "sine_bell", 0.4, 0.98, 2.0)
    from peakfit.engine.algorithms.noise_correlation import NoiseModel  # noqa: PLC0415

    assert NoiseModel(axes=(axis, axis)).description == (
        "Noise correlation modelled from the recorded processing "
        "(F2: sine bell, 92 → 256 points; F3: sine bell, 92 → 256 points)."
    )
    assert "can be too small" in not_modelled("axis F2: x").description
