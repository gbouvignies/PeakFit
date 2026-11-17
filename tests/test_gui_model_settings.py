"""The Model page's settings fit exactly as the CLI fits the equivalent configuration.

A desktop fit that excludes planes, sets the statistical noise or sets an absolute
clustering level must give the clusters and outcome of ``peakfit fit`` with the same
settings, bit for bit: the GUI prepares the same problem through the same functions,
only with stable peak identities in place of assignments.
"""

from copy import deepcopy

import numpy as np
import pytest
from builders import two_cluster_dataset

from peakfit.engine.domain.config import PeakFitConfig
from peakfit.fit import fitting
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.fitting import prepare_dataset_fit

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
