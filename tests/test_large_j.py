"""Large J couplings: the fitting region covers both doublet components (#214).

A doublet's position is the multiplet centre (``GLOSSARY.md``), so its components
sit at cs ± J/2. Clustering marks each peak's footprint before labelling the
contour-connected segments; a doublet's footprint spans cs ± J_max/2 on its axis,
J_max being the upper bound the fit gives J.
"""

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from peakfit.engine.algorithms.clustering import create_clusters, prepare_clusters
from peakfit.engine.algorithms.covariance_factors import lineshape_order
from peakfit.engine.algorithms.noise_correlation import not_modelled
from peakfit.engine.algorithms.varpro import (
    VarProOptimizer,
    fit_cluster,
    identifiable_covariance,
)
from peakfit.engine.diagnostics.couplings import CouplingQualification, coupling_qualifications
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import (
    AxisModel,
    Doublet,
    FitConfig,
    LineshapeFamily,
    PeakFitConfig,
)
from peakfit.engine.domain.constraints import ParameterConfig
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters, axis_lineshapes
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.engine.scientific.final_outcome import FinalParameter
from peakfit.engine.scientific.summary import RunSummary
from peakfit.fit.pipeline import run_pipeline
from peakfit.io.writers.markdown import write_final_outcome_report

if TYPE_CHECKING:
    from peakfit.engine.scientific.completed import CompletedFit

N15_SIZE, H1_SIZE = 256, 256
NOISE = 1.0
HEIGHT = 100.0  # first-plane peak height, in noise units
PROFILE = np.array([1.0, 0.6, 0.3])
TRUE_J = 92.0
TRUE_LW = {"F2": 20.0, "F3": 25.0}


def _axis(
    size: int, sw: float, obs: float, aq: float, *, direct: bool, label: str, apodized=False
) -> SpectralParameters:
    """An axis acquired for ``aq`` s, zero-filled to ``size``; ``apodized``: SP1 0.35 0.98."""
    return SpectralParameters(
        size=size,
        sw=sw,
        obs=obs,
        car=0.0,
        aq_time=aq,
        apocode=1.0 if apodized else 0.0,
        apodq1=0.35 if apodized else 0.0,
        apodq2=0.98 if apodized else 0.0,
        apodq3=1.0 if apodized else 0.0,
        p180=False,
        direct=direct,
        ft=True,
        label=label,
    )


def _spectra(*, apodized: bool = False) -> Spectra:
    """A pseudo-3D: three planes, ¹⁵N F2 (7.8 Hz per point, 32 ms), ¹H F3 (9.4 Hz, 50 ms)."""
    params = [
        _axis(3, 1.0, 1.0, 1.0, direct=False, label="F1"),
        _axis(N15_SIZE, 2000.0, 60.8, 0.032, direct=False, label="F2", apodized=apodized),
        _axis(H1_SIZE, 2400.0, 600.0, 0.05, direct=True, label="F3", apodized=apodized),
    ]
    return Spectra(
        dic={}, data=np.zeros((3, N15_SIZE, H1_SIZE)), z_values=np.arange(3.0), params=params
    )


def _doublet_peak(
    spectra: Spectra,
    doublet: Doublet,
    points=(128.3, 120.6),
    *,
    family: LineshapeFamily = "lorentzian",
    phase: bool = False,
) -> Peak:
    """Peak A: a doublet on ¹⁵N F2 (its position the multiplet centre), a ¹H singlet."""
    config = FitConfig(family=family, axes={"F2": AxisModel(doublet=doublet, phase=phase)})
    position = [
        float(axis.pts2ppm(p)) for axis, p in zip(spectra.spectral_params, points, strict=True)
    ]
    shapes = create_shapes(spectra, axis_lineshapes(config, spectra), "A", position)
    return Peak("A", np.array(position), shapes)


def _with_signal(
    spectra: Spectra,
    peak: Peak,
    seed: int = 7,
    *,
    j_hz: float = TRUE_J,
    widths: dict[str, float] = TRUE_LW,
    phase: float | None = None,
) -> Parameters:
    """Draw peak A with the given J, widths and F2 phase; return the true parameters."""
    truth = Parameters.from_peaks([peak])
    truth["A.F2.j"].value = j_hz
    for axis, width in widths.items():
        truth[f"A.{axis}.lw"].value = width
    if phase is not None:
        truth[next(name for name in truth if name.endswith("F2.phase"))].value = phase
    grid = [index.ravel() for index in np.indices((N15_SIZE, H1_SIZE))]
    shape = Cluster.evaluate_peaks([peak], truth, grid)[0]
    clean = PROFILE[:, None] * shape / np.max(np.abs(shape)) * HEIGHT * NOISE
    rng = np.random.default_rng(seed)
    spectra.data[:] = clean.reshape(3, N15_SIZE, H1_SIZE) + rng.normal(
        scale=NOISE, size=spectra.data.shape
    )
    return truth


def _fitted(spectra: Spectra, peak: Peak) -> CompletedFit:
    """Prepare and fit peak A as the CLI and GUI do."""
    config = PeakFitConfig(noise_level=NOISE)
    config.clustering.contour_level = 0.15 * HEIGHT * NOISE
    _, _, clusters = prepare_clusters(spectra, [peak], config)
    return run_pipeline(
        config,
        clusters,
        NOISE,
        Parameters.from_peaks([peak]),
        noise_model=not_modelled("synthetic"),
    ).completed_fit


def _component_points(spectra: Spectra, truth: Parameters, j_hz: float) -> list[tuple[int, int]]:
    """The grid indices of the two components' maxima, at cs ± J/2 on F2."""
    n15, h1 = spectra.spectral_params
    half = n15.hz2ppm(0.5 * j_hz)
    column = h1.ppm2pt_i(truth["A.F3.cs"].value)
    centre = truth["A.F2.cs"].value
    return [(n15.ppm2pt_i(centre - half), column), (n15.ppm2pt_i(centre + half), column)]


def _region(cluster: Cluster) -> set[tuple[int, int]]:
    return set(zip(*(index.tolist() for index in cluster.grid_indices), strict=True))


@pytest.mark.parametrize("j_hz", [TRUE_J, 46.0, 184.0])
def test_both_doublet_components_lie_in_the_fitting_region(j_hz) -> None:
    """At a contour of 15 % of the peak, each component is its own contour island.

    Before #214 the cluster was the 3 × 3 block at the multiplet centre, which holds
    no signal. Every J within the bounds [46, 184] Hz keeps both components inside.
    """
    spectra = _spectra()
    peak = _doublet_peak(spectra, Doublet(j=TRUE_J))
    truth = _with_signal(spectra, peak)
    [cluster] = create_clusters(spectra, [peak], contour_level=0.15 * HEIGHT * NOISE)
    region = _region(cluster)
    for point in _component_points(spectra, truth, j_hz):
        assert point in region


def test_the_footprint_follows_the_constrained_j_bound() -> None:
    """A ``*.F2.j`` rule applies after the axis model, so it sets the footprint too."""
    spectra = _spectra()
    peak = _doublet_peak(spectra, Doublet(j=TRUE_J, min=80.0, max=100.0))
    truth = _with_signal(spectra, peak)
    config = PeakFitConfig(
        noise_level=NOISE,
        parameters=ParameterConfig.model_validate({"defaults": {"*.F2.j": {"max": 250.0}}}),
    )
    config.clustering.contour_level = 0.15 * HEIGHT * NOISE
    _, _, [cluster] = prepare_clusters(spectra, [peak], config)
    region = _region(cluster)
    for point in _component_points(spectra, truth, 250.0):
        assert point in region
    narrow = create_clusters(spectra, [peak], contour_level=0.15 * HEIGHT * NOISE)
    assert not set(_component_points(spectra, truth, 250.0)) <= _region(narrow[0])


def test_a_footprint_across_the_spectral_edge_wraps() -> None:
    """A component past the edge aliases; its footprint continues from the other side."""
    spectra = _spectra()
    peak = _doublet_peak(spectra, Doublet(j=TRUE_J), points=(2.0, 60.6))
    footprint_f2, footprint_f3 = peak.footprint_i(Parameters.from_peaks([peak]))
    half = 0.5 * 184.0 / (2000.0 / N15_SIZE)  # J_max / 2, in points
    expected = {k % N15_SIZE for k in range(round(2.0 - half), round(2.0 + half) + 1)}
    assert set(footprint_f2.tolist()) == expected
    assert {0, N15_SIZE - 1} <= expected
    assert footprint_f3.tolist() == [61]


def test_a_singlet_footprint_is_its_centre_point() -> None:
    spectra = _spectra()
    config = FitConfig(family="lorentzian")
    position = [
        float(axis.pts2ppm(p))
        for axis, p in zip(spectra.spectral_params, (128.3, 120.6), strict=True)
    ]
    peak = Peak(
        "S",
        np.array(position),
        create_shapes(spectra, axis_lineshapes(config, spectra), "S", position),
    )
    footprint = peak.footprint_i(Parameters.from_peaks([peak]))
    assert [axis.tolist() for axis in footprint] == [[128], [121]]


def test_a_one_bond_doublet_is_recovered_by_the_fit() -> None:
    """J starts 20 % low; J, both positions and both widths land near the truth.

    A regression guard at 3σ on one realization; ``tools/validate_large_j.py`` holds
    the calibration evidence (±1σ and ±1.96σ coverage over many realizations).
    """
    spectra = _spectra()
    peak = _doublet_peak(spectra, Doublet(j=0.8 * TRUE_J))
    truth = _with_signal(spectra, peak, seed=11)
    config = PeakFitConfig(noise_level=NOISE)
    config.clustering.contour_level = 0.15 * HEIGHT * NOISE
    _, _, [cluster] = prepare_clusters(spectra, [peak], config)
    peak.set_cluster_id(cluster.cluster_id)
    params = Parameters.from_peaks([peak])
    result = fit_cluster(params, cluster, noise=NOISE, correlation=None)
    for name in ("A.F2.j", "A.F2.cs", "A.F3.cs", "A.F2.lw", "A.F3.lw"):
        fitted = result.params[name]
        assert np.isfinite(fitted.stderr), name
        assert abs(fitted.value - truth[name].value) <= 3 * fitted.stderr, name


def _qualification(completed: CompletedFit) -> CouplingQualification:
    """Peak A's J as a reader sees it: recorded values, θ named from the recorded model."""
    return completed.couplings()["A.F2.j"]


def test_a_coupling_below_the_linewidth_reports_its_linewidth_dependence() -> None:
    """J = ½ LW: the components overlap at half height, so J and LW trade off."""
    spectra = _spectra()
    peak = _doublet_peak(spectra, Doublet(j=10.0))
    _with_signal(spectra, peak, j_hz=10.0, widths={"F2": 20.0, "F3": 25.0})
    qualification = _qualification(_fitted(spectra, peak))
    assert qualification.below_linewidth
    assert qualification.linewidth_correlation is not None
    assert qualification.linewidth_correlation < -0.5


def test_a_one_bond_coupling_is_not_qualified() -> None:
    spectra = _spectra()
    peak = _doublet_peak(spectra, Doublet(j=0.8 * TRUE_J))
    _with_signal(spectra, peak)
    qualification = _qualification(_fitted(spectra, peak))
    assert not qualification.below_linewidth
    assert qualification.linewidth_correlation is not None
    assert abs(qualification.linewidth_correlation) < 0.3


def test_a_phased_doublet_recovers_j_and_phase_and_reports_their_correlation() -> None:
    """An SP1 doublet with a 10° phase on F2: J, the phase and ρ(J, phase).

    J's derivative is even about the multiplet centre and the phase's (dispersive) one
    is odd, so a resolved doublet's ρ(J, phase) is near zero; the phase trades off
    with the position instead.
    """
    spectra = _spectra(apodized=True)
    peak = _doublet_peak(spectra, Doublet(j=0.8 * TRUE_J), family="sp1", phase=True)
    _with_signal(spectra, peak, widths={"F2": 10.0, "F3": 20.0}, phase=10.0)
    completed = _fitted(spectra, peak)
    [cluster] = completed.outcome.clusters
    fitted = {p.name: p for p in cluster.final_nonlinear_parameters}
    phase = next(name for name in fitted if name.endswith("F2.phase"))
    for name, true in (("A.F2.j", TRUE_J), (phase, 10.0)):
        assert np.isfinite(fitted[name].standard_error), name
        assert abs(fitted[name].value - true) <= 3 * fitted[name].standard_error, name
    qualification = _qualification(completed)
    assert qualification.phase == phase
    assert qualification.phase_correlation is not None
    assert abs(qualification.phase_correlation) < 0.2


def test_a_withheld_covariance_leaves_only_the_linewidth_comparison() -> None:
    parameters = (
        FinalParameter("A.F2.cs", 120.0, 119.0, 121.0, True, "ppm", 0.01),
        FinalParameter("A.F2.lw", 20.0, 0.1, 200.0, True, "Hz", math.nan),
        FinalParameter("A.F2.j", 5.0, 2.5, 10.0, True, "Hz", math.nan),
    )
    [(name, qualification)] = coupling_qualifications(
        parameters, ["A.F2.cs", "A.F2.lw", "A.F2.j"], None
    ).items()
    assert name == "A.F2.j"
    assert qualification.below_linewidth
    assert qualification.linewidth == "A.F2.lw"
    assert qualification.linewidth_correlation is None
    assert qualification.phase is None


@pytest.mark.parametrize(
    ("family", "axes", "labels"),
    [
        ("pvoigt", {"F2": AxisModel(doublet=Doublet(j=92.0)), "F3": AxisModel()}, None),
        (
            "sp1",
            {
                "F2": AxisModel(doublet=Doublet(j=92.0), phase=True),
                "F3": AxisModel(doublet=Doublet(j=7.0), phase=True),
            },
            None,
        ),
        # Labels that sort against the dimension order: θ follows the dimensions.
        ("sp1", {"F2": AxisModel(phase=True), "F3": AxisModel(phase=True)}, ("F3", "F2")),
    ],
)
def test_lineshape_order_is_the_order_parameters_are_created(family, axes, labels) -> None:
    """The recorded parameters are sorted by name; θ follows the shapes' own order."""
    spectra = _spectra(apodized=True)
    if labels is not None:
        for axis, label in zip(spectra.spectral_params, labels, strict=True):
            axis.label = label
    lineshapes = axis_lineshapes(FitConfig(family=family, axes=axes), spectra)
    peaks = []
    for name, points in (("Z", (120.0, 60.0)), ("B", (124.0, 64.0))):
        position = [
            float(axis.pts2ppm(p)) for axis, p in zip(spectra.spectral_params, points, strict=True)
        ]
        peak = Peak(name, np.array(position), create_shapes(spectra, lineshapes, name, position))
        peak.set_cluster_id(3)
        peaks.append(peak)
    assert lineshape_order(peaks) == list(Parameters.from_peaks(peaks))


def test_the_recorded_correlations_name_the_right_rows() -> None:
    """ρ read back from the record equals the fit's own first-order values.

    The record sorts parameters by name while θ follows the shapes (cs, lw, j, phase);
    an SP1 doublet below its linewidth, with its phase fitted, has a strong ρ(J, LW)
    that any mix-up of rows would lose.
    """
    spectra = _spectra(apodized=True)
    peak = _doublet_peak(spectra, Doublet(j=15.0), family="sp1", phase=True)
    _with_signal(spectra, peak, j_hz=15.0, widths={"F2": 20.0, "F3": 20.0}, phase=10.0)
    config = PeakFitConfig(noise_level=NOISE)
    config.clustering.contour_level = 0.15 * HEIGHT * NOISE
    _, _, [cluster] = prepare_clusters(spectra, [peak], config)
    peak.set_cluster_id(cluster.cluster_id)
    result = fit_cluster(Parameters.from_peaks([peak]), cluster, noise=NOISE, correlation=None)
    names = result.params.get_vary_names()
    optimizer = VarProOptimizer(
        cluster=cluster, names=names, params_template=result.params, noise=NOISE
    )
    covariance = identifiable_covariance(
        optimizer.compute_jacobian(result.params.get_vary_values()), None
    )
    scale = np.sqrt(np.diag(covariance))
    direct = covariance / np.outer(scale, scale)
    phase = next(name for name in names if name.endswith("F2.phase"))
    j = names.index("A.F2.j")
    recorded = _qualification(_fitted(spectra, peak))
    assert direct[j, names.index("A.F2.lw")] < -0.5
    assert abs(direct[j, names.index(phase)]) > 0.3
    assert recorded.linewidth_correlation == pytest.approx(
        direct[j, names.index("A.F2.lw")], abs=0.02
    )
    assert recorded.phase_correlation == pytest.approx(direct[j, names.index(phase)], abs=0.02)


def test_the_report_lists_each_coupling_and_its_qualification(tmp_path) -> None:
    spectra = _spectra()
    peak = _doublet_peak(spectra, Doublet(j=10.0))
    _with_signal(spectra, peak, j_hz=10.0, widths={"F2": 20.0, "F3": 25.0})
    completed = _fitted(spectra, peak)
    qualification = completed.couplings()["A.F2.j"]
    summary = RunSummary.from_outcome(completed.outcome)
    report = write_final_outcome_report(
        completed.outcome, tmp_path / "report.md", summary, couplings=completed.couplings()
    ).read_text()
    assert "## J Couplings" in report
    row = next(line for line in report.splitlines() if line.startswith("| A.F2.j |"))
    assert "| yes |" in row
    assert f"| {qualification.linewidth_correlation:.2f} | N/A |" in row
    plain = write_final_outcome_report(completed.outcome, tmp_path / "plain.md", summary)
    assert "J Couplings" not in plain.read_text()
