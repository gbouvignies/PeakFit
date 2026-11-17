"""The Model page's settings fit exactly as the CLI fits the equivalent configuration.

A desktop fit that excludes planes, sets the statistical noise, sets an absolute
clustering level or gives peaks constraints of their own must give the clusters and
outcome of ``peakfit fit`` with the same settings, bit for bit: the GUI prepares the
same problem through the same functions, only with stable peak identities in place
of assignments.
"""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from builders import two_cluster_dataset

from peakfit.engine.domain.config import PeakFitConfig
from peakfit.fit import fitting
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.peaks import ParameterLimit

PLANE_VALUES = np.array([2.0, 0.0, 2.0, 1.0, 3.0, 0.5])  # repeated: identity is the index
EXCLUDED = [1, 4]


@pytest.fixture(scope="module")
def dataset():
    """Six planes of the two-cluster dataset; the planes to exclude corrupted."""
    amplitudes = np.array([[5.0, -3.0, 2.0, 4.0, 1.0, 3.0], [3.0, -2.0, 4.0, 1.0, 2.5, -1.0]])
    remote = np.array([4.0, -2.0, 3.0, 2.0, 1.0, 5.0])
    spectra, peaks, _config, _prepared = two_cluster_dataset(amplitudes, PLANE_VALUES, remote)
    spectra.data[EXCLUDED] *= 40.0  # anything leaking from an excluded plane would show
    return spectra, peaks


def _settings(**changes) -> PeakFitConfig:
    """The settings both fits use, as the TOML would give them."""
    base = {
        "fitting": {"family": "gaussian", "refine_iterations": 2},
        "clustering": {"contour_level": 0.1},
    }
    return PeakFitConfig.model_validate(base | changes)


def _cli(directory, monkeypatch, spectra, peaks, config):
    """``peakfit fit`` through ``load_data`` and ``run_fit``, on a peak list of ``peaks``."""
    directory.mkdir()
    source, peak_list = directory / "spectrum.ft2", directory / "peaks.list"
    source.write_bytes(b"synthetic spectrum, read through the patched reader")
    peak_list.write_text(
        "".join(f"{i}N-H {p.positions[0]!r} {p.positions[1]!r}\n" for i, p in enumerate(peaks))
    )
    monkeypatch.setattr(fitting, "read_spectra", lambda *_: spectra.model_copy(deep=True))
    data = fitting.load_data(source, peak_list, None, config)
    return data, fitting.run_fit(data, config, directory / "results", workers=1)


def _gui(spectra, peaks, config):
    prepared = prepare_dataset_fit(spectra, peaks, config)
    return prepared, finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))


def _assert_identical(gui, cli) -> None:
    """Every recorded number of two outcomes, bit for bit, peak i against peak i."""
    a, b = gui.outcome, cli.outcome
    assert a.excluded_planes == b.excluded_planes
    assert a.noise == b.noise
    assert a.noise_model == b.noise_model
    assert a.statistics == b.statistics
    assert [c.cluster_id for c in a.clusters] == [c.cluster_id for c in b.clusters]
    rename = {f"p_{i}": f"{i}N-H" for i in range(len(gui.inputs.identities))}
    for ca, cb in zip(a.clusters, b.clusters, strict=True):
        assert [rename[n] for n in ca.peak_names] == list(cb.peak_names)
        assert ca.classification == cb.classification
        for pa, pb in zip(
            ca.final_nonlinear_parameters, cb.final_nonlinear_parameters, strict=True
        ):
            peak, _, suffix = pa.name.partition(".")
            assert f"{rename[peak]}.{suffix}" == pb.name
            assert (pa.value, pa.standard_error, pa.min, pa.max, pa.vary) == (
                pb.value,
                pb.standard_error,
                pb.min,
                pb.max,
                pb.vary,
            )
        ea, eb = ca.analytical_evaluation, cb.analytical_evaluation
        assert ea is not None
        assert eb is not None
        for field in (
            "amplitudes",
            "amplitude_standard_errors",
            "amplitude_covariance",
            "model_values",
        ):
            np.testing.assert_array_equal(getattr(ea, field), getattr(eb, field), strict=True)
        assert ea.statistics == eb.statistics


def test_a_desktop_fit_excluding_planes_equals_peakfit_fit_exclude(dataset, tmp_path, monkeypatch):
    spectra, peaks = dataset
    config = _settings(exclude_planes=EXCLUDED)
    prepared, gui = _gui(spectra, peaks, config)
    _data, cli = _cli(tmp_path / "cli", monkeypatch, spectra, peaks, config)
    _assert_identical(gui, cli)
    # The run records the whole spectrum and which of its planes were fitted.
    snapshot = prepared.snapshot
    assert snapshot.data_shape == spectra.data.shape
    assert snapshot.plane_values == tuple(PLANE_VALUES)
    assert snapshot.source_plane_indices == (0, 2, 3, 5)
    assert gui.outcome.excluded_planes == (1, 4)
    assert all(
        c.analytical_evaluation.amplitudes.shape[1] == 4
        for c in gui.outcome.clusters
        if c.analytical_evaluation is not None
    )


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"noise_level": 0.0031}, id="manual-noise"),
        pytest.param({"clustering": {"contour_level": 0.35}}, id="absolute-level"),
        pytest.param(
            {"noise_level": 0.0031, "clustering": {"contour_factor": 30.0}},
            id="factor-of-manual-noise",
        ),
    ],
)
def test_manual_noise_and_an_absolute_level_cluster_and_fit_as_the_toml(
    dataset, tmp_path, monkeypatch, changes
):
    spectra, peaks = dataset
    config = _settings(exclude_planes=EXCLUDED, **changes)
    prepared, gui = _gui(spectra, peaks, config)
    data, cli = _cli(tmp_path / "cli", monkeypatch, spectra, peaks, config)
    assert (prepared.noise, prepared.contour_level) == (data.noise, data.contour_level)
    rename = {f"p_{i}": f"{i}N-H" for i in range(len(peaks))}
    assert [[rename[p.name] for p in c.peaks] for c in prepared.clusters] == [
        [p.name for p in c.peaks] for c in data.clusters
    ]
    for ours, theirs in zip(prepared.clusters, data.clusters, strict=True):
        for index, other in zip(ours.grid_indices, theirs.grid_indices, strict=True):
            np.testing.assert_array_equal(index, other)
    _assert_identical(gui, cli)


def test_the_estimate_stays_known_beside_a_manual_noise(dataset):
    spectra, peaks = dataset
    estimated = prepare_dataset_fit(spectra, peaks, _settings(exclude_planes=EXCLUDED))
    manual = prepare_dataset_fit(
        spectra, peaks, _settings(exclude_planes=EXCLUDED, noise_level=0.0031)
    )
    assert manual.noise == 0.0031
    assert manual.estimated_noise == estimated.noise == estimated.estimated_noise


# Each peak's own constraints, as ``peakfit fit`` reads them from the TOML. The two
# overlapping peaks share one assignment in the desktop ("Duplicate"); their limits stay
# their own because they belong to the working peak, not to its assignment.
OWN_TOML = {
    "0N-H": {"position_windows": {"F2": 0.01}, "F3.lw": {"min": 20.0, "max": 30.0}},
    "1N-H": {"F2.lw": {"value": 28.0, "vary": False}},
}
OWN = {
    0: (
        ("F2.cs", ParameterLimit(window=0.01)),
        ("F3.lw", ParameterLimit(lower=20.0, upper=30.0)),
    ),
    1: (("F2.lw", ParameterLimit(start=28.0, vary=False)),),
}


def test_a_desktop_fit_applies_each_peaks_own_constraints_as_peakfit_fit(
    dataset, tmp_path, monkeypatch
):
    spectra, peaks = dataset
    rule = {"defaults": {"*.*.lw": {"min": 5.0, "max": 100.0}}}
    config = _settings(exclude_planes=EXCLUDED, parameters=rule)
    constrained = tuple(replace(p, limits=OWN.get(i, ())) for i, p in enumerate(peaks))
    assert constrained[0].name == constrained[1].name  # one assignment, two peaks
    prepared, gui = _gui(spectra, constrained, config)
    cli_config = _settings(exclude_planes=EXCLUDED, parameters=rule | {"peaks": OWN_TOML})
    _data, cli = _cli(tmp_path / "cli", monkeypatch, spectra, peaks, cli_config)
    # The engine fits the configuration of the equivalent TOML, peak for peak.
    engine = prepared.engine_configuration
    rename = {f"p_{i}": f"{i}N-H" for i in range(len(peaks))}
    assert {rename[name]: c for name, c in engine.parameters.peaks.items()} == (
        cli_config.parameters.peaks
    )
    assert (
        engine.model_copy(update={"parameters": engine.parameters.model_copy(update={"peaks": {}})})
        == config
    )
    assert not prepared.problems
    _assert_identical(gui, cli)
    # And the constraints act: the held width stays at its start, the bounded one inside.
    final = {p.name: p for c in gui.outcome.clusters for p in c.final_nonlinear_parameters}
    held = final["p_1.F2.lw"]
    assert (held.value, held.vary) == (28.0, False)
    assert (final["p_0.F3.lw"].min, final["p_0.F3.lw"].max) == (20.0, 30.0)
    start = constrained[0].positions[0]
    assert final["p_0.F2.cs"].min == pytest.approx(start - 0.01)
    assert final["p_0.F2.cs"].max == pytest.approx(start + 0.01)
    # The record keeps the working settings and the limits on their peaks.
    assert PeakFitConfig.model_validate_json(prepared.snapshot.configuration) == config
    assert prepared.snapshot.peaks == constrained


def test_a_limit_that_cannot_be_fitted_as_given_blocks_the_fit(dataset):
    spectra, peaks = dataset
    rule = {"defaults": {"*.*.lw": {"min": 5.0, "max": 100.0}}}
    config = _settings(parameters=rule)
    outside = replace(peaks[0], limits=(("F3.lw", ParameterLimit(start=150.0)),))
    missing = replace(peaks[1], limits=(("F3.eta", ParameterLimit(vary=False)),))
    prepared = prepare_dataset_fit(spectra, (outside, missing, peaks[2]), config)
    reasons = {(p.working_id, p.key): p.reason for p in prepared.problems}
    assert reasons[outside.id, "F3.lw"].startswith("Start 150 Hz is outside its range 5–100 Hz.")
    assert "model has no such parameter" in reasons[missing.id, "F3.eta"]
    # ``peakfit fit`` would move the start to its bound without saying so; the desktop
    # names the peak and refuses to fit.
    assert prepared.problem_text().startswith(f"{outside.name}: LW F3: Start 150 Hz")
    with pytest.raises(ValueError, match="outside its range"):
        compute_dataset_fit(prepared)
