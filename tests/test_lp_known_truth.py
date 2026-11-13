"""The building blocks of the T7 known-truth study (tools/profile_uncertainty/lp_known_truth.py)."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def study():
    path = str(Path(__file__).resolve().parents[1] / "tools" / "profile_uncertainty")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(path)
        yield importlib.import_module("lp_known_truth")


def test_linear_prediction_extends_decaying_exponentials_exactly(study):
    t = np.arange(92)
    full = 1.0 * np.exp(2j * np.pi * 0.21 * t - 0.02 * t) + 0.6 * np.exp(
        2j * np.pi * 0.37 * t - 0.05 * t
    )
    extended = study.lp_extend(full[:47], 45)
    np.testing.assert_allclose(extended, full, rtol=0, atol=1e-9)


@pytest.mark.parametrize("estimator", ["OLS", "GLS"])
def test_plane_fit_recovers_the_truth_from_noise_free_data(study, estimator):
    chain = study.Chain(92, 256, 0.4)
    theta = np.array([[98.5, 1.3], [101.5, 1.3]])
    amplitudes = np.array([[1.0, 0.7], [0.8, 0.3], [0.5, 0.1]])
    index = np.arange(91, 110)
    y = chain.process(study.planes_fid(theta, amplitudes, chain.n))[:, index]
    r = study.lag_matrix(chain.rho(), index)
    whiten = study.whitening(np.eye(index.size) if estimator == "OLS" else r)
    fit = study.fit_planes(y, chain, index, theta + np.array([[0.3, 0.2], [-0.2, -0.1]]), whiten)
    np.testing.assert_allclose(fit.theta, theta, atol=1e-6)
    np.testing.assert_allclose(fit.amplitudes, amplitudes, atol=1e-6)


def test_free_scale_decay_fit_returns_the_rate_of_an_exact_profile(study):
    tau = np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert study.decay_rate(tau, 2.5 * np.exp(-3.0 * tau)) == pytest.approx(3.0, abs=1e-8)
