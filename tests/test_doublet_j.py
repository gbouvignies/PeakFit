"""Doublet lineshapes differentiate with respect to J, and the fit uses it.

A doublet is k(Δω₊) + k(Δω₋) with its components at cs ± J/2, so ∂/∂J is
½(s₋k′(Δω₋) − s₊k′(Δω₊)). Each check compares the analytic derivative with central
finite differences of the evaluated lineshape or of the VarPro residual itself.
"""

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from peakfit.engine.algorithms.covariance_factors import profile_standard_errors
from peakfit.engine.algorithms.evaluation import (
    AnalyticalModelEvaluation,
    evaluate_analytical_model,
)
from peakfit.engine.algorithms.noise_correlation import not_modelled
from peakfit.engine.algorithms.varpro import VarProOptimizer, _calculate_errors, fit_cluster
from peakfit.engine.domain.cluster import Cluster, MissingDerivativeError
from peakfit.engine.domain.config import PHASED_FAMILIES, AxisModel, Doublet, FitConfig
from peakfit.engine.domain.param_map import ParameterMap
from peakfit.engine.domain.params_scalar import Parameter, Parameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import (
    AxisLineshape,
    Spectra,
    SpectralParameters,
    axis_lineshapes,
)
from peakfit.engine.lineshapes import models
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.engine.types import ClusterParameters

DOUBLETS = {
    "gaussian": models.GaussianDoublet,
    "lorentzian": models.LorentzianDoublet,
    "pvoigt": models.PseudoVoigtDoublet,
    "no_apod": models.NoApodDoublet,
    "sp1": models.SP1Doublet,
    "sp2": models.SP2Doublet,
}
PHASED = PHASED_FAMILIES


def _axis(size, *, direct, obs, sw, p180=False, label="F2"):
    """A frequency axis with sine-bell apodization parameters for the phased kernels."""
    return SpectralParameters(
        size=size,
        sw=sw,
        obs=obs,
        car=0.0,
        aq_time=size / sw,
        apocode=1.0,
        apodq1=0.35,
        apodq2=0.98,
        apodq3=1.0,
        p180=p180,
        direct=direct,
        ft=True,
        label=label,
    )


# A ¹H-like direct axis (5.9 Hz per point) and a ¹⁵N-like indirect axis with P180
# sign alternation, where one component of the larger couplings aliases.
AXES = {
    "direct": _axis(512, direct=True, obs=600.0, sw=3000.0),
    "indirect": _axis(128, direct=False, obs=60.8, sw=1500.0, p180=True),
}


def _spectra(axis: SpectralParameters, n_planes: int = 1) -> Spectra:
    planes = SpectralParameters(
        size=n_planes,
        sw=1.0,
        obs=1.0,
        car=0.0,
        aq_time=0.1,
        apocode=0.0,
        apodq1=0.0,
        apodq2=0.0,
        apodq3=0.0,
        p180=False,
        direct=False,
        ft=False,
        label="F1",
    )
    return Spectra(
        dic={},
        data=np.zeros((n_planes, axis.size)),
        z_values=np.arange(float(n_planes)),
        params=[planes, axis],
    )


def _evaluate(shape, x, extras, *, derivs):
    return shape.evaluate_cluster(x, ClusterParameters(extras={**extras}), compute_derivs=derivs)


# Real-valued doublets have no phase; phased ones are checked unphased, at 0° and at 12°.
SHAPE_PHASES = [
    (lineshape, phase)
    for lineshape in sorted(DOUBLETS)
    for phase in ((None, 0.0, 12.0) if lineshape in PHASED else (None,))
]


@pytest.mark.parametrize("axis_name", sorted(AXES))
@pytest.mark.parametrize("j_hz", [2.0, 7.0, 35.0, 92.0])
@pytest.mark.parametrize(("lineshape", "phase"), SHAPE_PHASES)
def test_j_derivative_matches_finite_differences(lineshape, phase, j_hz, axis_name) -> None:
    axis = AXES[axis_name]
    spectra = _spectra(axis)
    model = AxisLineshape(axis.label, lineshape, Doublet(j=5.0), phase=phase is not None)
    shape = DOUBLETS[lineshape]("P", 0.0, spectra, 1, model)
    # Two peaks at once: the derivative is evaluated per peak, vectorised.
    centre = axis.pts2ppm(np.array([0.30, 0.62]) * axis.size)
    extras = {
        "cs": centre,
        "lw": np.array([4.0, 22.0]),
        "j": np.array([j_hz, 1.3 * j_hz]),
        "eta": np.array([0.3, 0.8]),
    }
    if phase is not None:
        extras["phase"] = np.array([phase, -phase])
    x = np.arange(axis.size, dtype=np.float64)

    analytic = _evaluate(shape, x, extras, derivs=True).derivatives["j"]
    step = 1e-3  # Hz: curvature follows the linewidth, not J
    plus = _evaluate(shape, x, {**extras, "j": extras["j"] + step}, derivs=False).values
    minus = _evaluate(shape, x, {**extras, "j": extras["j"] - step}, derivs=False).values
    numerical = (plus - minus) / (2 * step)

    assert analytic.shape == numerical.shape
    scale = np.max(np.abs(numerical), axis=0)
    assert np.all(scale > 0)
    np.testing.assert_allclose(analytic / scale, numerical / scale, rtol=0, atol=1e-6)


def _doublet_cluster(lineshape, *, j_hz, lw_hz, noise, seed=7, fit_phase=False):
    """One 1D doublet over three planes, its truth and a J range wide enough for ¹J."""
    axis = AXES["direct"]
    spectra = _spectra(axis, n_planes=3)
    centre = float(axis.pts2ppm(axis.size / 2))
    model = AxisModel(doublet=Doublet(j=5.0), phase=fit_phase)
    config = FitConfig(family=lineshape, axes={axis.label: model})
    peak = Peak(
        "A",
        np.array([centre]),
        create_shapes(spectra, axis_lineshapes(config, spectra), "A", [centre]),
    )
    peak.set_cluster_id(1)
    truth = Parameters.from_peaks([peak])
    for name, parameter in truth.items():
        if name.endswith(".j"):
            parameter.min, parameter.max, parameter.value = 0.5, 150.0, j_hz
        elif name.endswith(".lw"):
            parameter.value = lw_hz
        elif name.endswith(".phase"):
            parameter.value = 6.0
    grid = [np.arange(axis.size)]
    cluster = Cluster(1, [peak], grid, np.zeros((axis.size, 3)))
    amplitudes = np.array([[100.0, 60.0, 30.0]])
    cluster.data = cluster.evaluate(truth).T @ amplitudes
    cluster.data += noise * np.random.default_rng(seed).normal(size=cluster.data.shape)
    return cluster, truth


@pytest.mark.parametrize("lineshape", sorted(DOUBLETS))
def test_varpro_jacobian_differentiates_the_objective_with_respect_to_j(lineshape) -> None:
    cluster, params = _doublet_cluster(
        lineshape, j_hz=35.0, lw_hz=8.0, noise=0.5, fit_phase=lineshape in PHASED
    )
    names = params.get_vary_names()
    assert any(name.endswith(".j") for name in names)
    optimizer = VarProOptimizer(cluster, names, params, 0.5)
    x = params.get_vary_values()
    analytical = optimizer.compute_jacobian(x)
    steps = np.array([1e-7 if name.endswith(".cs") else 1e-4 for name in names])
    numerical = approx_derivative(optimizer.compute_residuals, x, method="3-point", abs_step=steps)
    j = names.index(next(name for name in names if name.endswith(".j")))
    assert np.linalg.norm(analytical[:, j]) > 0
    scale = np.max(np.abs(numerical), axis=0)
    np.testing.assert_allclose(analytical / scale, numerical / scale, rtol=0, atol=2e-5)


def _drop_j_derivative(monkeypatch):
    original = models.LorentzianDoublet.evaluate_cluster

    def without_j(self, x_grid, cluster_params, compute_derivs=False):
        result = original(self, x_grid, cluster_params, compute_derivs)
        result.derivatives.pop("j", None)
        return result

    monkeypatch.setattr(models.LorentzianDoublet, "evaluate_cluster", without_j)


def test_a_varied_parameter_without_a_derivative_fails_fast(monkeypatch) -> None:
    cluster, params = _doublet_cluster("lorentzian", j_hz=35.0, lw_hz=8.0, noise=0.0)
    _drop_j_derivative(monkeypatch)
    names = params.get_vary_names()
    d_matrix = np.zeros((len(names), cluster.n_points))
    with pytest.raises(MissingDerivativeError, match=r"A\.F2\.j.*no derivative"):
        cluster.evaluate(
            params,
            cluster.grid_indices,
            d_matrix=d_matrix,
            param_map=ParameterMap.from_names(names),
        )


def test_a_fixed_parameter_needs_no_derivative(monkeypatch) -> None:
    cluster, params = _doublet_cluster("lorentzian", j_hz=35.0, lw_hz=8.0, noise=0.0)
    _drop_j_derivative(monkeypatch)
    for name, parameter in params.items():
        if name.endswith(".j"):
            parameter.vary = False
    names = params.get_vary_names()
    d_matrix = np.zeros((len(names), cluster.n_points))
    cluster.evaluate(
        params, cluster.grid_indices, d_matrix=d_matrix, param_map=ParameterMap.from_names(names)
    )
    assert np.all(np.isfinite(d_matrix))


@pytest.mark.parametrize("lineshape", sorted(DOUBLETS))
@pytest.mark.parametrize(
    ("j_hz", "lw_hz"),
    [(40.0, 8.0), (12.0, 10.0)],
    ids=["resolved", "partly-resolved"],
)
def test_fit_recovers_j(lineshape, j_hz, lw_hz) -> None:
    cluster, truth = _doublet_cluster(
        lineshape, j_hz=j_hz, lw_hz=lw_hz, noise=0.2, fit_phase=lineshape in PHASED
    )
    params = truth.copy(deep=True)
    j_name = next(name for name in params if name.endswith(".j"))
    params[j_name].value = 0.75 * j_hz
    result = fit_cluster(params, cluster, noise=0.2, correlation=None)
    assert result.success
    fitted = result.params[j_name]
    assert fitted.value != pytest.approx(0.75 * j_hz, rel=1e-3)
    assert np.isfinite(fitted.stderr)
    assert fitted.stderr > 0
    assert abs(fitted.value - j_hz) < 4 * fitted.stderr


def _errors_at_truth(j_hz, lw_hz, noise):
    cluster, params = _doublet_cluster("lorentzian", j_hz=j_hz, lw_hz=lw_hz, noise=noise)
    names = params.get_vary_names()
    optimizer = VarProOptimizer(cluster, names, params, noise)
    return dict(
        zip(names, _calculate_errors(optimizer, params.get_vary_values(), None), strict=True)
    )


def test_an_unresolved_coupling_has_no_local_uncertainty() -> None:
    # J = ¼ linewidth: σ(J)/J ≈ 0.34 here. The doublet is even in J, so near J = 0 it
    # depends on J² and a first-order σ(J) is valid only when σ(J) ≪ J.
    errors = _errors_at_truth(5.0, 20.0, noise=5.0)
    assert np.isnan(errors["A.F2.j"])
    assert np.isfinite(errors["A.F2.lw"])
    assert np.isfinite(errors["A.F2.cs"])


@pytest.mark.parametrize(
    ("j_hz", "lw_hz", "noise"),
    [(40.0, 8.0, 1.0), (12.0, 10.0, 5.0), (92.0, 15.0, 5.0), (5.0, 20.0, 1.0)],
    ids=["resolved", "partly-resolved", "one-bond", "unresolved-high-snr"],
)
def test_a_determined_coupling_keeps_its_uncertainty(j_hz, lw_hz, noise) -> None:
    errors = _errors_at_truth(j_hz, lw_hz, noise)
    assert 0 < errors["A.F2.j"] <= 0.1 * j_hz


def _evaluated(j_hz, lw_hz, noise) -> AnalyticalModelEvaluation:
    cluster, params = _doublet_cluster("lorentzian", j_hz=j_hz, lw_hz=lw_hz, noise=noise)
    fit_cluster(params, cluster, noise, correlation=None)
    evaluation = evaluate_analytical_model(cluster, params, noise, not_modelled("synthetic"))
    assert isinstance(evaluation, AnalyticalModelEvaluation)
    return evaluation


def test_an_unresolved_coupling_withholds_the_profile_uncertainty() -> None:
    # The profile σ propagates the same first-order J uncertainty, so it goes too.
    evaluation = _evaluated(5.0, 20.0, noise=5.0)
    assert evaluation.uncertainty_warning is not None
    assert not evaluation.covariance_factors.available
    assert np.isnan(
        profile_standard_errors(evaluation.covariance_factors, evaluation.amplitudes)
    ).all()
    assert np.isfinite(evaluation.amplitude_standard_errors).all()


def test_a_determined_coupling_keeps_the_profile_uncertainty() -> None:
    evaluation = _evaluated(40.0, 8.0, noise=1.0)
    assert evaluation.uncertainty_warning is None
    assert evaluation.covariance_factors.available
    assert np.isfinite(
        profile_standard_errors(evaluation.covariance_factors, evaluation.amplitudes)
    ).all()


def test_a_missing_derivative_is_not_absorbed_by_the_evaluation(monkeypatch) -> None:
    # The profile-σ path withholds on lineshape errors; a missing derivative is not one.
    cluster, params = _doublet_cluster("lorentzian", j_hz=35.0, lw_hz=8.0, noise=0.5)
    _drop_j_derivative(monkeypatch)
    with pytest.raises(MissingDerivativeError):
        evaluate_analytical_model(cluster, params, 0.5, not_modelled("synthetic"))


def test_a_coupling_collapsed_onto_its_bound_withholds_the_profile_uncertainty() -> None:
    # The low-SNR failure of an unresolved doublet: J at its lower bound has no σ, and
    # the profile σ would be conditional on that arbitrary bound.
    cluster, params = _doublet_cluster("lorentzian", j_hz=5.0, lw_hz=20.0, noise=5.0)
    j_name = next(name for name in params if name.endswith(".j"))
    params[j_name].value = params[j_name].min
    params[j_name].stderr = np.nan
    evaluation = evaluate_analytical_model(cluster, params, 5.0, not_modelled("synthetic"))
    assert isinstance(evaluation, AnalyticalModelEvaluation)
    assert not evaluation.covariance_factors.available
    assert evaluation.uncertainty_warning is not None


def test_a_coupling_is_recognised_without_a_structured_identifier() -> None:
    # Parameters rebuilt by name (sampling, continuation, reopened projects) carry none.
    assert Parameter(name="A.F2.j", value=5.0).is_coupling
    assert Parameter(name="cluster_1.F2.phase", value=0.0).is_coupling is False
    assert Parameter(name="A.F2.lw", value=5.0).is_coupling is False
