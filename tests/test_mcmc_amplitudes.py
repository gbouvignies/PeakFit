"""MCMC amplitude summaries: docs/mcmc-formulation.md §5-§6 and validation V5."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from builders import completed_result, peak_pair

from peakfit.engine.domain.config import FitConfig, PeakFitConfig
from peakfit.engine.scientific.amplitudes import balanced_samples, summarize_amplitudes
from peakfit.engine.scientific.sampler import SamplerSettings, sample_posterior
from peakfit.engine.scientific.sampling import conditional_amplitudes, sampling_problem
from peakfit.fit.pipeline import run_pipeline

if TYPE_CHECKING:
    from peakfit.engine.scientific.sampling import SamplingProblem


@pytest.fixture(scope="module")
def isolated():
    """The single, well-resolved peak of the two-cluster dataset, at high SNR."""
    completed = completed_result([], False, True)[2].completed_fit
    cluster = next(c for c in completed.outcome.clusters if len(c.peak_names) == 1)
    problem = sampling_problem(completed, cluster.cluster_id)
    posterior = sample_posterior(
        problem, SamplerSettings(walkers=16, steps=3000), np.random.SeedSequence(2026)
    )
    return completed, problem, posterior


def _joint_inverse_fisher(problem):
    """σ² (JᵀJ)⁻¹ over (θ, amplitudes), with ∂Φ/∂θ by central differences."""
    theta = problem.start
    amplitudes = problem.conditional(theta).mean  # (peaks, planes)
    columns = []
    for j, name in enumerate(problem.names):
        step = 1e-4 * problem.parameters[name].stderr
        shifted = []
        for sign in (1.0, -1.0):
            moved = theta.copy()
            moved[j] += sign * step
            parameters = problem.parameters.copy()
            for n, value in zip(problem.names, moved, strict=True):
                parameters[n].value = float(value)
            shifted.append(problem.cluster.evaluate(parameters).T @ amplitudes)
        columns.append(((shifted[0] - shifted[1]) / (2 * step)).ravel())
    shapes = problem.cluster.evaluate(problem.parameters)  # (peaks, points)
    planes = amplitudes.shape[1]
    for k in range(shapes.shape[0]):
        for plane in range(planes):
            column = np.zeros((shapes.shape[1], planes))
            column[:, plane] = shapes[k]
            columns.append(column.ravel())
    jacobian = np.column_stack(columns)
    return problem.noise**2 * np.linalg.inv(jacobian.T @ jacobian)


def test_v5_the_linear_regime_matches_the_joint_inverse_fisher_matrix(isolated):
    _, problem, posterior = isolated
    assert not posterior.provisional, posterior.reasons
    covariance = _joint_inverse_fisher(problem)
    n_theta = len(problem.names)
    for j, p in enumerate(posterior.diagnosis.parameters):
        # θ: VarPro's identifiable covariance and the joint inverse Fisher matrix agree.
        np.testing.assert_allclose(p.sd, np.sqrt(covariance[j, j]), rtol=0.1)
        np.testing.assert_allclose(p.sd, problem.parameters[p.name].stderr, rtol=0.1)
    conditional = problem.conditional(problem.start)
    for summary in posterior.amplitudes.amplitudes:
        joint = np.sqrt(covariance[n_theta + summary.plane, n_theta + summary.plane])
        np.testing.assert_allclose(summary.sd, joint, rtol=0.1)
        # In this regime only: wider than the analytical error conditional on θ̂.
        assert summary.sd > np.sqrt(conditional.scale @ conditional.scale.T)[0, 0]


def test_amplitude_moments_and_draws_agree(isolated):
    _, _, posterior = isolated
    for summary in posterior.amplitudes.amplitudes:
        low95, low68, median, high68, high95 = summary.quantiles
        # Gaussian in this regime: the draws' quantiles match the total-variance moments.
        assert abs(median - summary.mean) < 5 * summary.mean_error
        np.testing.assert_allclose((high68 - low68) / 2, summary.sd, rtol=0.1)
        np.testing.assert_allclose((high95 - low95) / 2, 1.96 * summary.sd, rtol=0.1)


def test_amplitude_draws_are_reproducible_from_the_seed(isolated):
    _, problem, posterior = isolated
    again = sample_posterior(
        problem, SamplerSettings(walkers=16, steps=3000), np.random.SeedSequence(2026)
    )
    assert again.amplitudes == posterior.amplitudes


@pytest.mark.parametrize(("separation", "flagged"), [(2.0, True), (8.0, False)])
def test_strongly_correlated_amplitudes_are_flagged_and_summed(separation, flagged):
    cluster, params = peak_pair(separation, noise=0.05)
    config = PeakFitConfig(fitting=FitConfig(lineshape="gaussian"))
    completed = run_pipeline(config, [cluster], 0.05, params).completed_fit
    posterior = sample_posterior(
        sampling_problem(completed, 1),
        SamplerSettings(walkers=16, steps=2000),
        np.random.SeedSequence(5),
    )
    amplitudes = posterior.amplitudes
    assert bool(amplitudes.correlated) is flagged
    assert bool(amplitudes.sums) is flagged
    # Positions and linewidths stay reported, with a caution, unless peaks exchange.
    assert bool(posterior.caution("A")) is flagged
    if flagged:
        assert posterior.caution("B") == "not separately resolved from A (amplitude |ρ| ≥ 0.9)"
    if flagged:
        assert {(c.first, c.second) for c in amplitudes.correlated} == {("A", "B")}
        assert all(abs(c.rho) >= 0.9 for c in amplitudes.correlated)
        for total in amplitudes.sums:
            parts = [a for a in amplitudes.amplitudes if a.plane == total.plane]
            assert total.peaks == ("A", "B")
            # Anticorrelated members: their sum is far better determined than either.
            assert total.sd < min(a.sd for a in parts)


@dataclass(frozen=True)
class _Peak:
    name: str


@dataclass(frozen=True)
class _Cluster:
    peaks: tuple[_Peak, ...]


@dataclass(frozen=True)
class _Fixed:
    """A posterior whose shapes do not depend on θ: two duplicated peaks and one apart."""

    shapes: np.ndarray
    data: np.ndarray
    cluster: _Cluster = _Cluster((_Peak("A"), _Peak("B"), _Peak("C")))

    def conditional(self, _theta):
        return conditional_amplitudes(self.shapes, self.data, 0.01)


def test_duplicated_peaks_are_withheld_but_their_sum_is_reported():
    x = np.arange(60.0)
    shapes = np.exp(-0.5 * ((x[None, :] - np.array([[20.0], [20.0], [40.0]])) / 3.0) ** 2)
    truth = np.array([[2.0, 1.0], [1.0, 3.0], [3.0, 0.5]])
    data = shapes.T @ truth
    summaries = summarize_amplitudes(
        cast("SamplingProblem", _Fixed(shapes, data)),
        np.zeros((500, 1, 1)),
        np.array([0.0, 0.0, 1.0]),
        np.random.SeedSequence(0),
        500.0,
    )
    assert summaries.undetermined == (("A", 1.0), ("B", 1.0))
    assert {a.peaks for a in summaries.amplitudes} == {("C",)}
    assert {s.peaks for s in summaries.sums} == {("A", "B")}
    for total in summaries.sums:
        np.testing.assert_allclose(total.mean, truth[:2, total.plane].sum(), rtol=1e-10)
        # The sum's conditional error: σ / ‖φ‖ for the shared shape.
        np.testing.assert_allclose(total.sd, 0.01 / np.linalg.norm(shapes[0]), rtol=1e-6)


@dataclass(frozen=True)
class _Scaled:
    """One peak whose amplitude is proportional to θ, so each walker's samples differ."""

    shapes: np.ndarray
    data: np.ndarray
    cluster: _Cluster = _Cluster((_Peak("A"),))

    def conditional(self, theta):
        return conditional_amplitudes(self.shapes, self.data * theta[0], 0.01)


def test_amplitude_samples_cover_every_walker_and_the_whole_chain():
    steps, walkers = 2000, 16
    pairs = balanced_samples(steps, walkers)
    assert len(pairs) <= 4000
    counts = np.bincount(pairs[:, 1], minlength=walkers)
    assert counts.min() == counts.max() == 250
    for walker in range(walkers):  # each walker spans the whole chain, not a stretch
        times = pairs[pairs[:, 1] == walker, 0]
        assert (times.min(), times.max()) == (0, steps - 1)
    # Walker w holds θ = w + 1 throughout; the amplitude is 2θ. Equal representation of
    # the walkers gives a mean of 2 × 8.5. Indices spaced evenly through the flattened
    # (steps, walkers) chain gave walker 7 no sample, walker 15 one, each other walker
    # one stretch of about 285 steps, and a mean of 2 × 8.0.
    x = np.arange(60.0)
    shapes = np.exp(-0.5 * ((x - 30.0) / 3.0) ** 2)[None, :]
    kept = np.broadcast_to(np.arange(1.0, walkers + 1)[None, :, None], (steps, walkers, 1))
    summaries = summarize_amplitudes(
        cast("SamplingProblem", _Scaled(shapes, shapes.T * 2.0)),
        kept,
        np.ones(1),
        np.random.SeedSequence(0),
        4000.0,
    )
    (amplitude,) = summaries.amplitudes
    np.testing.assert_allclose(amplitude.mean, 2.0 * 8.5, rtol=1e-12)
