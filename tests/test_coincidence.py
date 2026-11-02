"""Near-collinear fitted shapes: diagnosis, retained positions and initialization.

Bundled cases and controls are those of docs/validation/repeat-fit-collapse.md:
three near-collinear pairs on the real CEST data, an exact duplicate assignment,
and a real opposite-sign peak injected 0.5 and 0.25 linewidth from 136N-H.
"""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from peakfit.engine.diagnostics.coincidence import COLLINEARITY_LIMIT, collinear_pairs
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import FitConfig, PeakFitConfig
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.fit_export import export_fit_result
from peakfit.gui.fit_positions import continuation_source, dataset_completion
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.peaks import WorkingPeak
from peakfit.gui.spectrum import read_dataset
from peakfit.io.scientific import read_scientific

DATA = Path(__file__).resolve().parents[1] / "examples/02-advanced-fitting/data"
PAIRS = {("36N-H", "93N-H"), ("5N-H", "76N-H"), ("53N-H", "147N-H")}


# The diagnostic ------------------------------------------------------------------


def _lorentzian(x, center, width):
    return 1 / (1 + ((x - center) / (width / 2)) ** 2)


def _gaussian(x, center, width):
    return np.exp(-4 * np.log(2) * ((x - center) / width) ** 2)


@pytest.mark.parametrize("shape", [_lorentzian, _gaussian])
def test_flags_equal_shapes_within_about_one_percent_of_a_linewidth(shape):
    x = np.linspace(-10, 10, 801)
    for separation, flagged in ((0.0, True), (0.004, True), (0.05, False), (0.5, False)):
        shapes = np.stack([shape(x, 0, 2), shape(x, 2 * separation, 2)])
        assert bool(collinear_pairs(shapes)) is flagged, (shape.__name__, separation)


def test_dissimilarity_stays_accurate_near_collinearity():
    x = np.linspace(-10, 10, 2001)
    sigma, delta = 1.0, 1e-5
    shapes = np.stack(
        [np.exp(-(x**2) / (2 * sigma**2)), np.exp(-((x - delta) ** 2) / (2 * sigma**2))]
    )
    (pair,) = collinear_pairs(shapes)
    # For Gaussians, 1 − cos = 1 − exp(−δ²/4σ²) ≈ δ²/4σ² = 2.5×10⁻¹¹, far below
    # what 1 − (u·v) resolves in double precision.
    assert pair.dissimilarity == pytest.approx(delta**2 / (4 * sigma**2), rel=1e-3)


def test_opposite_sign_shapes_count_by_magnitude_and_report_cancellation():
    x = np.linspace(-10, 10, 401)
    s = _lorentzian(x, 0, 2)
    pair = collinear_pairs(np.stack([s, -s]), np.array([[1e6, 2e6], [1e6 - 1, 2e6 - 3]]))[0]
    assert pair.dissimilarity == 0
    assert pair.cancellation > 1e5


def test_zero_or_nonfinite_shapes_are_never_paired():
    x = np.linspace(-10, 10, 101)
    s = _lorentzian(x, 0, 2)
    bad = s.copy()
    bad[3] = np.nan
    assert collinear_pairs(np.stack([s, np.zeros_like(s)])) == ()
    (pair,) = collinear_pairs(np.stack([s, bad, s * 2]))
    assert (pair.first, pair.second) == (0, 2)
    assert collinear_pairs(np.stack([s])) == ()
    assert collinear_pairs(np.zeros((0, 5))) == ()


def test_clusters_with_more_peaks_report_each_collinear_pair():
    x = np.linspace(-10, 10, 401)
    a, b, c = _lorentzian(x, 0, 2), _lorentzian(x, 0.001, 2), _lorentzian(x, 3, 2)
    assert [(p.first, p.second) for p in collinear_pairs(np.stack([a, c, b]))] == [(0, 2)]
    assert [(p.first, p.second) for p in collinear_pairs(np.stack([a, b, a]))] == [
        (0, 1),
        (0, 2),
        (1, 2),
    ]
    assert COLLINEARITY_LIMIT == 1e-4


# The bundled data and its controls ---------------------------------------------------


def _fit(spectra, peaks, settings, completion=None):
    source = continuation_source(completion) if completion is not None else None
    prepared = prepare_dataset_fit(spectra, tuple(peaks), settings, source)
    result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    return dataset_completion(
        result, result.inputs.snapshot, submitted_revision=0, manual_revision=0
    )


def _cluster(completion, *members):
    names = {i.engine_name: i.assignment for i in completion.result.inputs.identities}
    return next(
        c
        for c in completion.result.outcome.clusters
        if set(members) <= {names[p] for p in c.peak_names}
    )


def _ids(dataset):
    return {p.name: p.id for p in dataset.peaks}


@pytest.fixture(scope="module")
def bundled():
    dataset = read_dataset(DATA / "pseudo3d.ft2", DATA / "pseudo3d.list", DATA / "b1_offsets.txt")
    settings = PeakFitConfig(fitting=FitConfig(refine_iterations=2))
    first = _fit(dataset.spectra, dataset.peaks, settings)
    second = _fit(dataset.spectra, first.working_peaks, settings, first)
    third = _fit(dataset.spectra, second.working_peaks, settings, second)
    return dataset, settings, first, second, third


@pytest.mark.slow
def test_bundled_pairs_are_retained_with_their_fit_unchanged(bundled, tmp_path):
    dataset, _settings, first, _second, _third = bundled
    ids, names = _ids(dataset), {p.id: p.name for p in dataset.peaks}
    retained = dict(first.retained)
    assert {names[i] for i in retained} == {n for pair in PAIRS for n in pair}
    for a, b in PAIRS:
        assert retained[ids[a]] == f"near-collinear fitted shapes with {b}"
        assert retained[ids[b]] == f"near-collinear fitted shapes with {a}"
    seeds = {p.id: p.positions for p in dataset.peaks}
    working = {p.id: p.positions for p in first.working_peaks}
    for identity in retained:
        assert working[identity] == seeds[identity]  # previous working coordinates
    # Healthy members of the same clusters adopt as usual.
    for name in ("75N-H", "10N-H", "105N-H"):
        assert ids[name] in first.adopted
        assert working[ids[name]] != seeds[ids[name]]
    # The fitted result is untouched: the recorded collapsed values stay.
    cluster = _cluster(first, "53N-H", "147N-H")
    assert cluster.analytical_evaluation.uncertainty_warning is not None
    cs = [p.value for p in cluster.final_nonlinear_parameters if p.name.endswith("F2.cs")]
    assert abs(cs[0] - cs[1]) < 1e-3
    # Exports keep the fitted coordinates and state why they were not adopted.
    path = export_fit_result(first.result, tmp_path / "result", completion=first)
    _, _, context = read_scientific(path / "scientific.npz")
    row = context["coordinate_table"]["peaks"][ids["147N-H"]]
    assert row["adoption_status"] == "retained_initial"
    assert row["adoption_reason"].startswith("near-collinear fitted shapes with p_")
    assert row["fitted_unfolded_ppm"] is not None
    np.testing.assert_array_equal(row["adopted_unfolded_ppm"], seeds[ids["147N-H"]])


@pytest.mark.slow
def test_retained_peaks_restart_from_defaults_and_unedited_refits_settle(bundled):
    dataset, _settings, first, second, third = bundled
    ids, names = _ids(dataset), {p.id: p.name for p in dataset.peaks}
    continuation = second.result.inputs.continuation
    assert continuation is not None
    run = first.result.run_id[:8]
    reasons = dict(continuation.reasons)
    assert {names[i] for i in continuation.initialized} == {n for pair in PAIRS for n in pair}
    assert reasons[ids["53N-H"]] == (
        f"not adopted by run {run}: near-collinear fitted shapes with 147N-H"
    )
    assert len(continuation.continued) == len(dataset.peaks) - 6
    # The identifiability cases re-collapse in place and stay retained.
    for pair in (("53N-H", "147N-H"), ("36N-H", "93N-H")):
        before = _cluster(first, *pair).analytical_evaluation.statistics.reduced_chi_squared
        after = _cluster(second, *pair).analytical_evaluation.statistics.reduced_chi_squared
        assert after == pytest.approx(before, rel=1e-5)
        assert {names[i] for i in dict(second.retained)} >= set(pair)
    # The optimizer-failure case reaches its better resolved fit from defaults.
    five = _cluster(second, "5N-H", "76N-H").analytical_evaluation
    assert five.uncertainty_warning is None
    assert (
        five.statistics.chi_squared
        < _cluster(first, "5N-H", "76N-H").analytical_evaluation.statistics.chi_squared
    )
    assert _cluster(second, "52N-H", "102N-H").analytical_evaluation.uncertainty_warning is None
    # From then on an unedited refit reproduces every cluster and every position.
    for a, b in zip(second.working_peaks, third.working_peaks, strict=True):
        assert a.id == b.id
        np.testing.assert_allclose(a.positions, b.positions, atol=1e-4)
    for cluster in third.result.outcome.clusters:
        names3 = {i.engine_name: i.assignment for i in third.result.inputs.identities}
        members = {names3[p] for p in cluster.peak_names}
        previous = _cluster(second, *members).analytical_evaluation
        assert cluster.analytical_evaluation.statistics.reduced_chi_squared == pytest.approx(
            previous.statistics.reduced_chi_squared, rel=1e-5
        ), members
        assert bool(cluster.analytical_evaluation.uncertainty_warning) == bool(
            previous.uncertainty_warning
        )
    assert dict(third.retained).keys() == dict(second.retained).keys()


@pytest.mark.slow
def test_editing_a_retained_peak_restores_ordinary_initialization(bundled):
    dataset, settings, first, _second, _third = bundled
    ids = _ids(dataset)
    moved = ids["53N-H"]
    peaks = tuple(
        replace(p, positions=(p.positions[0] + 0.01, p.positions[1])) if p.id == moved else p
        for p in first.working_peaks
    )
    prepared = prepare_dataset_fit(dataset.spectra, peaks, settings, continuation_source(first))
    assert prepared.continuation is not None
    reasons = dict(prepared.continuation.reasons)
    run = first.result.run_id[:8]
    assert reasons[moved] == f"moved since run {run}"
    assert reasons[ids["147N-H"]].startswith(f"not adopted by run {run}")


def _inject(dataset, settings, fraction):
    """A real negative resonance `fraction` of a linewidth from 136N-H, as in #89."""
    first = _fit(dataset.spectra, dataset.peaks, settings)
    cluster = _cluster(first, "136N-H")
    fitted = {p.name.partition(".")[2]: p.value for p in cluster.final_nonlinear_parameters}
    amplitude = np.asarray(cluster.analytical_evaluation.amplitudes, float)[0]
    y_axis, x_axis = dataset.spectra.params[1], dataset.spectra.params[2]
    position = (
        fitted["F2.cs"] + fraction * fitted["F2.lw"] / y_axis.obs,
        fitted["F3.cs"] + fraction * fitted["F3.lw"] / x_axis.obs,
    )
    spectra = dataset.spectra.model_copy(update={"data": dataset.spectra.data.copy()})
    shapes = list(first.result.inputs.shape_names)
    peak = Peak(
        "neg", np.array(position), create_shapes(spectra, settings.fitting, "neg", position, shapes)
    )
    params = Parameters.from_peaks([peak])
    params["neg.F2.lw"].value, params["neg.F3.lw"].value = fitted["F2.lw"], fitted["F3.lw"]
    rows, columns = np.indices(spectra.data.shape[1:])
    values = Cluster.evaluate_peaks([peak], params, [rows.ravel(), columns.ravel()])[0]
    spectra.data += (-0.6 * amplitude)[:, None, None] * values.reshape(rows.shape)
    neg = WorkingPeak("neg", "NEG", position)
    return neg, _fit(spectra, (*dataset.peaks, neg), settings)


@pytest.mark.slow
def test_controls_flag_only_near_collinear_decompositions(bundled):
    dataset, settings, _first, _second, _third = bundled
    ids = _ids(dataset)
    seed = next(p for p in dataset.peaks if p.name == "136N-H")
    # (a) An exact duplicate assignment collapses into a cancelling pair: retained.
    duplicate = WorkingPeak("dup", "DUP", seed.positions)
    completion = _fit(dataset.spectra, (*dataset.peaks, duplicate), settings)
    assert {"dup", ids["136N-H"]} <= dict(completion.retained).keys()
    # (b) A real opposite-sign peak 0.5 linewidth away is resolved and adopted.
    neg, completion = _inject(dataset, settings, 0.5)
    assert {"neg", ids["136N-H"]} <= completion.adopted
    # (c) At 0.25 linewidth the fit collapses although two resonances are present:
    # both stable identities keep their own submitted coordinates; nothing is
    # merged, deleted or assigned.
    neg, completion = _inject(dataset, settings, 0.25)
    retained = dict(completion.retained)
    assert retained[ids["136N-H"]] == "near-collinear fitted shapes with NEG"
    assert retained["neg"] == "near-collinear fitted shapes with 136N-H"
    working = {p.id: p for p in completion.working_peaks}
    assert working["neg"] == neg
    assert working[ids["136N-H"]].positions == seed.positions
    assert len(completion.working_peaks) == len(dataset.peaks) + 1
