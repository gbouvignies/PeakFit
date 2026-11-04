"""The ensemble sampler and its diagnostics: docs/mcmc-formulation.md §7, V4, V8, V10."""

from concurrent.futures import CancelledError
from dataclasses import replace

import emcee
import numpy as np
import pytest
from builders import peak_pair
from scipy import stats

from peakfit.engine.diagnostics import ensemble
from peakfit.engine.domain.config import FitConfig, PeakFitConfig
from peakfit.engine.scientific.sampler import (
    LogPosterior,
    PosteriorSample,
    SamplerSettings,
    diagnose,
    exchanges,
    run_ensemble,
    start_positions,
)
from peakfit.engine.scientific.sampling import sampling_problem
from peakfit.fit.pipeline import run_pipeline

INF = np.inf


def _gaussian(covariance):
    precision = np.linalg.inv(covariance)

    def log_probability(x):
        return -0.5 * x @ precision @ x

    return log_probability


def _run(log_probability, start, *, lower=None, upper=None, walkers=16, steps=3000, seed=1):
    start = np.asarray(start, dtype=float)
    lower = np.full(start.size, -INF) if lower is None else np.asarray(lower, dtype=float)
    upper = np.full(start.size, INF) if upper is None else np.asarray(upper, dtype=float)
    run = run_ensemble(
        log_probability,
        start,
        np.full(start.size, 0.1),
        lower,
        upper,
        walkers=walkers,
        steps=steps,
        seed=np.random.SeedSequence(seed),
    )
    return run, lower, upper


def test_v4_a_correlated_gaussian_is_recovered_within_monte_carlo_error():
    covariance = np.array([[1.0, 1.6], [1.6, 4.0]])  # sd 1 and 2, ρ = 0.8
    run, lower, upper = _run(_gaussian(covariance), [0.0, 0.0])
    diagnosis = diagnose(run, ("x", "y"), lower, upper)
    assert diagnosis.reasons == ()
    assert diagnosis.tau_reliable
    for p, sd in zip(diagnosis.parameters, (1.0, 2.0), strict=True):
        assert abs(p.mean) < 5 * p.mean_error
        # The standard deviation's own error is about sd / sqrt(2 ESS).
        assert abs(p.sd - sd) < 5 * sd / np.sqrt(2 * p.ess)
    assert abs(diagnosis.correlation[0, 1] - 0.8) < 0.05


def test_v4_a_truncated_gaussian_keeps_its_bound_and_moments():
    a = 0.5

    def log_probability(x):
        return -0.5 * float(x[0]) ** 2 if x[0] >= a else -INF

    run, lower, upper = _run(log_probability, [1.0], lower=[a], walkers=8)
    diagnosis = diagnose(run, ("x",), lower, upper)
    (p,) = diagnosis.parameters
    expected = stats.truncnorm(a, INF)
    assert p.quantiles[0] >= a
    assert abs(p.mean - expected.mean()) < 5 * p.mean_error
    assert abs(p.sd - expected.std()) < 5 * expected.std() / np.sqrt(2 * p.ess)


def test_v4_equal_seeds_give_identical_chains():
    first, _, _ = _run(_gaussian(np.eye(2)), [0.0, 0.0], steps=200, seed=7)
    second, _, _ = _run(_gaussian(np.eye(2)), [0.0, 0.0], steps=200, seed=7)
    other, _, _ = _run(_gaussian(np.eye(2)), [0.0, 0.0], steps=200, seed=8)
    np.testing.assert_array_equal(first.chain, second.chain)
    assert not np.array_equal(first.chain, other.chain)
    assert (first.entropy, first.spawn_key) == (7, ())


def test_v4_a_short_run_is_provisional_not_trusted():
    run, lower, upper = _run(_gaussian(np.eye(2)), [0.0, 0.0], steps=60)
    diagnosis = diagnose(run, ("x", "y"), lower, upper, burn_in=0)
    assert not diagnosis.tau_reliable
    assert any("shorter than 50 τ_max" in reason for reason in diagnosis.reasons)


def test_v4_a_drifting_ensemble_fails_the_stationarity_check():
    # Walkers start far from the mode and nothing is discarded; the first half of the
    # chain is still on its way down.
    run, lower, upper = _run(_gaussian(np.eye(1)), [1e4], walkers=8, steps=160)
    diagnosis = diagnose(run, ("x",), lower, upper, burn_in=0)
    assert not diagnosis.parameters[0].stationary
    assert any("two halves" in reason for reason in diagnosis.reasons)


def test_a_burn_in_that_leaves_no_chain_is_refused():
    run, lower, upper = _run(_gaussian(np.eye(1)), [0.0], walkers=8, steps=50)
    with pytest.raises(ValueError, match="run longer"):
        diagnose(run, ("x",), lower, upper, burn_in=48)


def test_an_automatic_burn_in_longer_than_the_chain_is_provisional(monkeypatch):
    # emcee's windowed estimator keeps 5 τ below the chain length in practice, so the
    # whole-chain estimate is pinned to a poorly mixing value: τ as long as the chain.
    run, lower, upper = _run(_gaussian(np.eye(1)), [0.0], walkers=8, steps=40)
    estimate = emcee.autocorr.integrated_time

    def whole_chain_tau(chain, tol=50):
        if tol == 0:
            return np.full(chain.shape[2], float(chain.shape[0]))
        return estimate(chain, tol=tol)

    monkeypatch.setattr(emcee.autocorr, "integrated_time", whole_chain_tau)
    diagnosis = diagnose(run, ("x",), lower, upper)
    assert diagnosis.burn_in == 20  # the second half, summarized only as an indication
    assert diagnosis.reasons[0] == (
        "the chain (40 steps) is shorter than its automatic burn-in (5 τ_max = 200 "
        "steps); only its second half is summarized, as an indication"
    )
    # An explicit burn-in that leaves too little chain is still the user's error.
    with pytest.raises(ValueError, match="run longer"):
        diagnose(run, ("x",), lower, upper, burn_in=38)


def test_serial_and_pooled_runs_give_identical_chains(close_pair):
    log_posterior = LogPosterior(close_pair)
    scale = np.full(close_pair.start.size, 1e-3)

    def run(workers):
        return run_ensemble(
            log_posterior,
            close_pair.start,
            scale,
            close_pair.lower,
            close_pair.upper,
            walkers=8,
            steps=30,
            seed=np.random.SeedSequence(11),
            workers=workers,
        )

    serial, pooled = run(1), run(2)
    np.testing.assert_array_equal(serial.chain, pooled.chain)
    np.testing.assert_array_equal(serial.log_probability, pooled.log_probability)
    np.testing.assert_array_equal(serial.blobs, pooled.blobs)


def test_too_few_walkers_are_refused_not_silently_raised():
    with pytest.raises(ValueError, match="use at least 6"):
        _run(_gaussian(np.eye(3)), [0.0, 0.0, 0.0], walkers=4, steps=10)


def test_sampling_reports_progress_and_can_be_canceled():
    seen = []

    def canceled():
        return len(seen) >= 5

    with pytest.raises(CancelledError):
        run_ensemble(
            _gaussian(np.eye(1)),
            np.zeros(1),
            np.ones(1),
            np.full(1, -INF),
            np.full(1, INF),
            walkers=8,
            steps=100,
            seed=np.random.SeedSequence(0),
            progress=seen.append,
            canceled=canceled,
        )
    assert [p.step for p in seen] == [1, 2, 3, 4, 5]
    assert all(p.steps == 100 and 0 <= p.acceptance <= 1 for p in seen)


def test_v8_walkers_start_strictly_inside_the_bounds():
    # θ̂ on its lower bound: half of the draws fall outside and are drawn again.
    positions = start_positions(
        _gaussian(np.eye(2)),
        np.array([0.0, 0.0]),
        np.array([0.5, 0.5]),
        np.array([0.0, -INF]),
        np.array([1.0, INF]),
        64,
        np.random.default_rng(0),
    )
    assert np.all(positions[:, 0] > 0.0)
    assert np.all(positions[:, 0] < 1.0)


def test_v8_pile_up_follows_the_at_bound_rule():
    samples = np.array([[0.0, 5.0], [0.5, 5.0], [1e-7, 5.0], [1.0, 5.0]])
    at_lower, at_upper = ensemble.bound_pileup(samples, np.array([0.0, 0.0]), np.array([1.0, INF]))
    np.testing.assert_allclose(at_lower, [0.5, 0.0])  # 0 and 1e-7 are within 1e-6 of 1
    np.testing.assert_allclose(at_upper, [0.25, 0.0])  # one-sided bounds never pile up


def test_undefined_correlations_stay_undefined():
    samples = np.column_stack([np.arange(10.0), np.ones(10)])
    matrix = ensemble.correlation(samples)
    assert np.isnan(matrix[0, 1])
    assert np.isnan(matrix[1, 1])
    assert matrix[0, 0] == 1.0


@pytest.fixture(scope="module")
def close_pair():
    cluster, params = peak_pair(2.0, noise=0.05)
    config = PeakFitConfig(fitting=FitConfig(lineshape="gaussian"))
    completed = run_pipeline(config, [cluster], 0.05, params).completed_fit
    return sampling_problem(completed, 1)


def test_v10_an_exchange_of_two_peaks_is_reported_and_withholds_both(close_pair):
    problem = close_pair
    a, b = problem.names.index("A.F2.cs"), problem.names.index("B.F2.cs")
    flat = np.tile(problem.start, (100, 1))
    assert exchanges(problem, flat) == ()
    flat[:7, [a, b]] = flat[:7, [b, a]]  # 7% of the samples swap the two positions
    (exchange,) = exchanges(problem, flat)
    assert (exchange.first, exchange.second, exchange.axis) == ("A", "B", "F2")
    assert exchange.fraction == pytest.approx(0.07)
    run, lower, upper = _run(_gaussian(np.eye(1)), [0.0], walkers=8)
    posterior = PosteriorSample(
        cluster_id=1,
        names=problem.names,
        peak_names=("A", "B"),
        settings=SamplerSettings(),
        run=run,
        diagnosis=replace(diagnose(run, ("x",), lower, upper), reasons=()),
        exchanges=(exchange,),
        determined=np.ones(2),
    )
    assert posterior.provisional
    assert posterior.reasons == ("peaks A and B exchange order along F2 in 7.0% of samples",)
    assert posterior.withheld("A.F2.lw") == "exchanges identity with B"
    assert posterior.withheld("B.F2.cs") == "exchanges identity with A"
