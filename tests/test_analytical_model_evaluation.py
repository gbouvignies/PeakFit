from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest

from peakfit.engine.algorithms.evaluation import (
    AnalyticalEvaluationFailure,
    AnalyticalModelEvaluation,
    FitOutcomeClassification,
    classify_optimizer_result,
    evaluate_analytical_model,
)
from peakfit.engine.algorithms.linear_algebra import (
    calculate_amplitude_covariance,
    calculate_amplitudes_with_uncertainty,
)
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import BasinHoppingConfig, FitConfig, VarProConfig
from peakfit.engine.domain.param_id import PSEUDO_AXIS, ParameterId
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters
from peakfit.engine.fitting.optimizers import fit_with_optimizer
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.engine.results import FitResult

if TYPE_CHECKING:
    from peakfit.engine.domain.params_scalar import Parameters


@dataclass(frozen=True)
class GaussianFit:
    cluster: Cluster
    params: Parameters
    expected_amplitudes: np.ndarray


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


@pytest.fixture
def gaussian_fit() -> GaussianFit:
    n_points = 9
    n_series = 2
    spectra = Spectra(
        dic={},
        data=np.zeros((n_series, n_points), dtype=np.float64),
        z_values=np.array([0.0, 1.0], dtype=np.float64),
        params=[
            _spectral_parameters(size=n_series, direct=False),
            _spectral_parameters(size=n_points, direct=True),
        ],
    )
    center = float(spectra.spectral_params[0].pts2ppm(4.0))
    peak = Peak(
        name="P1",
        positions=np.array([center], dtype=np.float64),
        shapes=create_shapes(
            spectra,
            FitConfig(lineshape="gaussian"),
            "P1",
            [center],
            ["gaussian"],
        ),
    )
    peak.set_cluster_id(37)
    params = peak.create_params()
    grid_indices = [np.arange(n_points, dtype=np.intp)]
    empty_cluster = Cluster(
        cluster_id=37,
        peaks=[peak],
        grid_indices=grid_indices,
        data=np.zeros((n_points, n_series), dtype=np.float64),
    )
    expected_amplitudes = np.array([[2.0, 3.0]], dtype=np.float64)
    data = empty_cluster.evaluate(params).T @ expected_amplitudes
    return GaussianFit(
        cluster=Cluster(
            cluster_id=37,
            peaks=[peak],
            grid_indices=grid_indices,
            data=data,
        ),
        params=params,
        expected_amplitudes=expected_amplitudes,
    )


def test_analytical_model_evaluation_solves_a_real_gaussian_cluster(
    gaussian_fit: GaussianFit,
) -> None:
    evaluation = evaluate_analytical_model(
        gaussian_fit.cluster,
        gaussian_fit.params,
        noise=1.0,
    )

    assert isinstance(evaluation, AnalyticalModelEvaluation)
    np.testing.assert_allclose(evaluation.amplitudes, gaussian_fit.expected_amplitudes)
    np.testing.assert_allclose(evaluation.model_values, gaussian_fit.cluster.corrected_data)
    np.testing.assert_allclose(evaluation.raw_residuals, 0.0, atol=1e-12)
    assert evaluation.statistics.chi_squared == pytest.approx(0.0, abs=1e-24)
    assert evaluation.statistics.n_observations == 18
    assert evaluation.statistics.n_nonlinear_parameters == 2
    assert evaluation.statistics.n_amplitude_parameters == 2
    assert evaluation.statistics.n_fitted_parameters == 4
    assert evaluation.statistics.degrees_of_freedom == 14
    assert evaluation.statistics.reduced_chi_squared == pytest.approx(
        0.0,
        abs=1e-24,
    )
    assert evaluation.statistics.amplitude_uncertainty_scale == 1.0
    np.testing.assert_allclose(
        evaluation.scaled_amplitude_standard_errors,
        evaluation.amplitude_standard_errors,
    )
    assert np.isfinite(evaluation.statistics.aic)
    assert np.isfinite(evaluation.statistics.bic)
    assert np.isfinite(evaluation.statistics.log_likelihood)


@pytest.mark.parametrize(
    ("success", "message", "expected_classification"),
    [
        (True, "optimizer converged", FitOutcomeClassification.CONVERGED),
        (
            False,
            "requested number of iterations completed successfully",
            FitOutcomeClassification.USABLE_NON_CONVERGED,
        ),
    ],
)
def test_numerical_usability_is_independent_of_optimizer_convergence(
    gaussian_fit: GaussianFit,
    success: bool,
    message: str,
    expected_classification: FitOutcomeClassification,
) -> None:
    analytical = evaluate_analytical_model(
        gaussian_fit.cluster,
        gaussian_fit.params,
        noise=1.0,
    )
    assert isinstance(analytical, AnalyticalModelEvaluation)
    result = FitResult(
        cluster_id=gaussian_fit.cluster.cluster_id,
        params=gaussian_fit.params,
        residual=analytical.normalized_residuals,
        cost=0.5 * analytical.statistics.chi_squared,
        success=success,
        message=message,
        n_amplitude_params=gaussian_fit.cluster.n_amplitude_params,
    )

    evaluated = classify_optimizer_result(
        cluster=gaussian_fit.cluster,
        result=result,
        noise=1.0,
    )

    assert evaluated.classification is expected_classification
    assert evaluated.usable is True
    assert evaluated.analytical is not None
    assert evaluated.unusable_reason is None


def test_numerical_usability_rejects_nonfinite_nonlinear_parameters(
    gaussian_fit: GaussianFit,
) -> None:
    params = gaussian_fit.params.copy(deep=True)
    parameter_name = params.get_vary_names()[0]
    params.params[parameter_name] = params[parameter_name].model_copy(update={"value": np.nan})
    result = FitResult(
        cluster_id=gaussian_fit.cluster.cluster_id,
        params=params,
        residual=np.zeros(gaussian_fit.cluster.n_observations),
        cost=0.0,
        success=True,
        message="optimizer claimed convergence",
        n_amplitude_params=gaussian_fit.cluster.n_amplitude_params,
    )

    evaluated = classify_optimizer_result(
        cluster=gaussian_fit.cluster,
        result=result,
        noise=1.0,
    )

    assert evaluated.classification is FitOutcomeClassification.UNUSABLE
    assert evaluated.usable is False
    assert evaluated.analytical is None
    assert evaluated.unusable_reason == (f"non-finite nonlinear parameters: {parameter_name}")


def test_analytical_model_evaluation_rejects_nonfinite_amplitudes(
    gaussian_fit: GaussianFit,
) -> None:
    params = gaussian_fit.params.copy(deep=True)
    linewidth_name = next(name for name in params if name.endswith(".lw"))
    params[linewidth_name].value = 3.5
    cluster = Cluster(
        cluster_id=gaussian_fit.cluster.cluster_id,
        peaks=gaussian_fit.cluster.peaks,
        grid_indices=[np.array([3, 5], dtype=np.intp)],
        data=np.full((2, 1), 1e308, dtype=np.float64),
    )

    evaluation = evaluate_analytical_model(cluster, params, noise=1.0)

    assert isinstance(evaluation, AnalyticalEvaluationFailure)
    assert evaluation.reason == "non-finite analytical amplitudes"


def test_amplitude_solver_skips_covariance_after_nonfinite_amplitudes() -> None:
    shapes = np.array([[1e-200]], dtype=np.float64)
    data = np.array([[1e308]], dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        amplitudes, errors, covariance = calculate_amplitudes_with_uncertainty(
            shapes,
            data,
            noise=1.0,
        )

    assert not np.isfinite(amplitudes).all()
    assert np.isnan(errors).all()
    assert np.isnan(covariance).all()


def test_evaluator_keeps_nonfinite_covariance_as_uncertainty_failure(
    gaussian_fit: GaussianFit,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_peaks = len(gaussian_fit.cluster.peaks)
    monkeypatch.setattr(
        "peakfit.engine.algorithms.evaluation.calculate_amplitudes_with_uncertainty",
        lambda *_args: (
            np.ones((n_peaks, gaussian_fit.cluster.n_series)),
            np.full((n_peaks,), np.inf),
            np.full((n_peaks, n_peaks), np.inf),
        ),
    )

    evaluation = evaluate_analytical_model(gaussian_fit.cluster, gaussian_fit.params, noise=1.0)

    assert isinstance(evaluation, AnalyticalEvaluationFailure)
    assert evaluation.reason == "non-finite amplitude uncertainty inputs"


def test_amplitude_solver_preserves_nonfinite_covariance_for_finite_amplitudes() -> None:
    shapes = np.array([[1e-200]], dtype=np.float64)
    data = np.array([[1e-200]], dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        amplitudes, errors, covariance = calculate_amplitudes_with_uncertainty(
            shapes,
            data,
            noise=1.0,
        )

    assert np.isfinite(amplitudes).all()
    assert not np.isfinite(errors).all()
    assert not np.isfinite(covariance).all()


def test_amplitude_calculations_preserve_ordinary_finite_results() -> None:
    shapes = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float64)
    data = np.array([[4.0, 6.0], [9.0, 12.0]], dtype=np.float64)

    amplitudes, errors, covariance = calculate_amplitudes_with_uncertainty(
        shapes,
        data,
        noise=2.0,
    )

    np.testing.assert_allclose(amplitudes, [[2.0, 3.0], [3.0, 4.0]])
    np.testing.assert_allclose(errors, [1.0, 2.0 / 3.0])
    np.testing.assert_allclose(covariance, [[1.0, 0.0], [0.0, 4.0 / 9.0]])
    np.testing.assert_allclose(calculate_amplitude_covariance(shapes, noise=2.0), covariance)


def test_amplitude_covariance_suppresses_floating_point_overflow_warning() -> None:
    shapes = np.array([[1e-200]], dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        covariance = calculate_amplitude_covariance(shapes, noise=1.0)

    assert not np.isfinite(covariance).all()


@pytest.mark.parametrize(
    ("nonfinite_field", "expected_reason"),
    [
        ("residual", "non-finite optimizer residuals"),
        ("cost", "non-finite optimizer cost"),
    ],
)
def test_numerical_usability_rejects_nonfinite_optimizer_numerics(
    gaussian_fit: GaussianFit,
    nonfinite_field: str,
    expected_reason: str,
) -> None:
    analytical = evaluate_analytical_model(
        gaussian_fit.cluster,
        gaussian_fit.params,
        noise=1.0,
    )
    assert isinstance(analytical, AnalyticalModelEvaluation)
    residual = analytical.normalized_residuals.copy()
    cost = 0.5 * analytical.statistics.chi_squared
    if nonfinite_field == "residual":
        residual[0] = np.nan
    else:
        cost = np.inf
    result = FitResult(
        cluster_id=gaussian_fit.cluster.cluster_id,
        params=gaussian_fit.params,
        residual=residual,
        cost=cost,
        success=True,
        n_amplitude_params=gaussian_fit.cluster.n_amplitude_params,
    )

    evaluated = classify_optimizer_result(
        cluster=gaussian_fit.cluster,
        result=result,
        noise=1.0,
    )

    assert evaluated.classification is FitOutcomeClassification.UNUSABLE
    assert evaluated.analytical is None
    assert evaluated.unusable_reason == expected_reason


@pytest.mark.parametrize(
    ("cluster_id", "residual_size", "n_amplitude_params", "expected_reason"),
    [
        (91, 18, 2, "cluster_id mismatch: expected 37, got 91"),
        (37, 17, 2, "optimizer residual shape mismatch: expected (18,), got (17,)"),
        (
            37,
            18,
            1,
            "optimizer amplitude parameter count mismatch: expected 2, got 1",
        ),
    ],
)
def test_numerical_usability_rejects_incompatible_optimizer_result(
    gaussian_fit: GaussianFit,
    cluster_id: int,
    residual_size: int,
    n_amplitude_params: int,
    expected_reason: str,
) -> None:
    result = FitResult(
        cluster_id=cluster_id,
        params=gaussian_fit.params,
        residual=np.zeros(residual_size),
        cost=0.0,
        success=True,
        n_amplitude_params=n_amplitude_params,
    )

    evaluated = classify_optimizer_result(
        cluster=gaussian_fit.cluster,
        result=result,
        noise=1.0,
    )

    assert evaluated.classification is FitOutcomeClassification.UNUSABLE
    assert evaluated.unusable_reason == expected_reason


def test_analytical_model_evaluation_rejects_incompatible_design_shape(
    gaussian_fit: GaussianFit,
) -> None:
    cluster = Cluster(
        cluster_id=gaussian_fit.cluster.cluster_id,
        peaks=gaussian_fit.cluster.peaks,
        grid_indices=gaussian_fit.cluster.grid_indices,
        data=gaussian_fit.cluster.data,
    )
    cluster.data = cluster.data[:-1]
    cluster.corrections = np.zeros_like(cluster.data)

    evaluation = evaluate_analytical_model(cluster, gaussian_fit.params, noise=1.0)

    assert isinstance(evaluation, AnalyticalEvaluationFailure)
    assert evaluation.reason == ("lineshape values shape mismatch: expected (1, 8), got (1, 9)")


def test_analytical_model_evaluation_reports_lineshape_failure(
    gaussian_fit: GaussianFit,
) -> None:
    gaussian_fit.cluster.peaks[0].shapes.clear()

    evaluation = evaluate_analytical_model(
        gaussian_fit.cluster,
        gaussian_fit.params,
        noise=1.0,
    )

    assert isinstance(evaluation, AnalyticalEvaluationFailure)
    assert evaluation.reason.startswith("lineshape evaluation failed: IndexError:")


def test_analytical_model_evaluation_ignores_stale_injected_amplitudes(
    gaussian_fit: GaussianFit,
) -> None:
    params = gaussian_fit.params.copy(deep=True)
    for series_index in range(gaussian_fit.cluster.n_series):
        params.add(
            ParameterId(
                peak_name=gaussian_fit.cluster.peaks[0].name,
                axis=PSEUDO_AXIS,
                label="I",
                index=series_index,
            ),
            value=10_000.0 + series_index,
            vary=False,
            computed=True,
        )

    evaluation = evaluate_analytical_model(gaussian_fit.cluster, params, noise=1.0)

    assert isinstance(evaluation, AnalyticalModelEvaluation)
    np.testing.assert_allclose(evaluation.amplitudes, gaussian_fit.expected_amplitudes)


def test_analytical_model_evaluation_preserves_rank_deficient_fallback(
    gaussian_fit: GaussianFit,
) -> None:
    cluster = Cluster(
        cluster_id=gaussian_fit.cluster.cluster_id,
        peaks=[gaussian_fit.cluster.peaks[0], gaussian_fit.cluster.peaks[0]],
        grid_indices=gaussian_fit.cluster.grid_indices,
        data=gaussian_fit.cluster.data[:, :1],
    )

    with pytest.warns(RuntimeWarning, match="Rank-deficient design matrix"):
        evaluation = evaluate_analytical_model(cluster, gaussian_fit.params, noise=1.0)

    assert isinstance(evaluation, AnalyticalModelEvaluation)
    assert np.isfinite(evaluation.amplitudes).all()
    assert np.isfinite(evaluation.amplitude_covariance).all()
    np.testing.assert_allclose(evaluation.model_values, cluster.corrected_data)


@pytest.mark.parametrize(
    ("optimizer", "config"),
    [
        ("varpro", VarProConfig(max_nfev=25)),
        ("basin_hopping", BasinHoppingConfig(n_iterations=1, seed=23)),
    ],
)
def test_shared_evaluation_matches_optimizer_terminal_residual(
    gaussian_fit: GaussianFit,
    optimizer: str,
    config: VarProConfig | BasinHoppingConfig,
) -> None:
    result = fit_with_optimizer(
        optimizer,
        gaussian_fit.params.copy(deep=True),
        gaussian_fit.cluster,
        noise=1.0,
        config=config,
    )

    evaluated = classify_optimizer_result(
        cluster=gaussian_fit.cluster,
        result=result,
        noise=1.0,
    )

    assert evaluated.usable
    assert evaluated.analytical is not None
    np.testing.assert_allclose(
        evaluated.analytical.normalized_residuals,
        result.residual,
        rtol=1e-8,
        atol=1e-12,
    )
    assert evaluated.analytical.statistics.chi_squared == pytest.approx(
        float(np.sum(result.residual**2)),
        rel=1e-8,
        abs=1e-20,
    )
