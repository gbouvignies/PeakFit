"""Ensemble sampling of one cluster's posterior.

See ``docs/mcmc-formulation.md`` §4, §7 and §8. ``run_ensemble`` drives emcee from
one recorded ``SeedSequence``, checks cancellation and reports progress every step.
``diagnose`` applies the ensemble diagnostics to a run. ``sample_posterior`` does
both for a ``SamplingProblem``, adding label switching and per-sample amplitude
identifiability.
"""

import os
from concurrent.futures import CancelledError
from dataclasses import dataclass
from math import ceil
from multiprocessing import get_context
from typing import TYPE_CHECKING

import emcee
import numpy as np
from threadpoolctl import threadpool_limits

from peakfit.engine.diagnostics import ensemble
from peakfit.engine.scientific.amplitudes import AmplitudeSummaries, summarize_amplitudes
from peakfit.shared.constants import MCMC_N_STEPS, MCMC_N_WALKERS

if TYPE_CHECKING:
    from collections.abc import Callable

    from peakfit.engine.scientific.sampling import SamplingProblem
    from peakfit.shared.typing import FloatArray

# The version of docs/mcmc-formulation.md this code implements; it changes whenever
# sampling, diagnosis or summaries change what a run produces (§8).
FORMULATION = "1.0"
# Each stream ``run_ensemble`` spawns from a run's SeedSequence, by child index path,
# and its bit generator.
STREAMS = {
    "0/0 walker starts": "PCG64",
    "0/1 emcee moves": "MT19937",
    "1 amplitude draws": "PCG64",
}
# An automatic burn-in discards this many τ_max.
BURN_IN_TAUS = 5
# Reported quantiles: the central 95% and 68% intervals and the median.
QUANTILES = (0.025, 0.16, 0.5, 0.84, 0.975)
# Walkers start this many fit standard errors around θ̂.
_START_SPREAD = 0.1
# Redraws allowed for one walker's start before giving up.
_START_ATTEMPTS = 1000
_NAME_PARTS = 3
# Fewest kept steps that can be summarized and split into halves.
_MIN_KEPT = 4


@dataclass(frozen=True)
class SamplerSettings:
    """What a run does."""

    walkers: int = MCMC_N_WALKERS
    steps: int = MCMC_N_STEPS
    burn_in: int | None = None  # steps discarded; None: BURN_IN_TAUS × τ_max


@dataclass(frozen=True)
class SamplerProgress:
    """One completed step and the ensemble's acceptance fraction so far."""

    step: int
    steps: int
    acceptance: float


@dataclass(frozen=True)
class LogPosterior:
    """Picklable log-posterior of a problem, returning per-peak identifiability.

    Outside the prior's support it is −∞, with no amplitude determined.
    """

    problem: SamplingProblem

    def __call__(self, theta: FloatArray) -> tuple[float, np.ndarray]:
        """Return log π(θ | Y) up to a constant, and which amplitudes are determined at θ."""
        conditional = self.problem.conditional(theta)
        if conditional is None:
            return -np.inf, np.zeros(len(self.problem.cluster.peaks), dtype=bool)
        return conditional.log_likelihood, conditional.determined


# A pool worker's log-posterior, installed once when the worker starts so that steps
# do not pickle the problem again.
_WORKER: dict[str, Callable] = {}


def _install(log_probability: Callable) -> None:
    _WORKER["log_probability"] = log_probability


def _pooled(theta: FloatArray):
    return _WORKER["log_probability"](theta)


@dataclass(frozen=True)
class EnsembleRun:
    """Everything the sampler produced, burn-in included."""

    chain: FloatArray  # (steps, walkers, parameters)
    log_probability: FloatArray  # (steps, walkers)
    blobs: np.ndarray | None  # (steps, walkers, ...) as the log-probability returned
    acceptance: FloatArray  # (walkers,)
    entropy: int | tuple[int, ...]  # with ``spawn_key``, the SeedSequence that reproduces it
    spawn_key: tuple[int, ...]


def start_positions(
    log_probability: Callable,
    start: FloatArray,
    scale: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    walkers: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Walkers drawn from N(start, scale²), strictly inside the bounds.

    A draw outside the open interval, or with a non-finite log-probability, is
    drawn again; nothing is clipped onto a bound.
    """
    positions = np.empty((walkers, start.size))
    for walker in range(walkers):
        for _ in range(_START_ATTEMPTS):
            theta = start + scale * rng.standard_normal(start.size)
            if np.all((lower < theta) & (theta < upper)) and np.isfinite(
                _value(log_probability(theta))
            ):
                positions[walker] = theta
                break
        else:
            raise ValueError(
                "Cannot place MCMC walkers inside the parameter bounds around the fitted "
                "values; widen the bounds or refit."
            )
    return positions


def _value(result) -> float:
    return float(result[0] if isinstance(result, tuple) else result)


def run_ensemble(
    log_probability: Callable,
    start: FloatArray,
    scale: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    *,
    walkers: int,
    steps: int,
    seed: np.random.SeedSequence,
    progress: Callable[[SamplerProgress], None] | None = None,
    canceled: Callable[[], bool] | None = None,
    workers: int = 1,
) -> EnsembleRun:
    """Run emcee from one ``SeedSequence``.

    The sequence spawns a sampler stream and an amplitude-draw stream (§8). The
    sampler stream seeds both the starting positions and emcee's moves, so an equal
    sequence gives identical chains. Serial unless ``workers`` asks for a pool
    (−1: every CPU); pool workers receive the log-probability once.
    """
    dimensions = start.size
    if walkers < 2 * dimensions:
        raise ValueError(
            f"{walkers} walkers cannot sample {dimensions} parameters; "
            f"use at least {2 * dimensions}."
        )
    sampler_stream, _draw_stream = seed.spawn(2)
    start_stream, move_stream = sampler_stream.spawn(2)
    positions = start_positions(
        log_probability,
        start,
        scale,
        lower,
        upper,
        walkers,
        np.random.default_rng(start_stream),
    )
    moves = [(emcee.moves.StretchMove(), 0.8), (emcee.moves.DEMove(), 0.2)]

    def run(function, pool) -> emcee.EnsembleSampler:
        sampler = emcee.EnsembleSampler(walkers, dimensions, function, pool=pool, moves=moves)
        sampler.random_state = np.random.RandomState(np.random.MT19937(move_stream)).get_state()
        for step, _ in enumerate(sampler.sample(positions, iterations=steps), start=1):
            if canceled is not None and canceled():
                raise CancelledError("MCMC sampling canceled")
            if progress is not None:
                progress(SamplerProgress(step, steps, float(np.mean(sampler.acceptance_fraction))))
        return sampler

    if workers == 1:
        sampler = run(log_probability, None)
    else:
        processes = workers if workers > 0 else (os.cpu_count() or 1)
        with (
            threadpool_limits(limits=1, user_api="blas"),
            get_context("spawn").Pool(
                processes, initializer=_install, initargs=(log_probability,)
            ) as pool,
        ):
            sampler = run(_pooled, pool)
    blobs = sampler.get_blobs()
    return EnsembleRun(
        chain=sampler.get_chain(),
        log_probability=sampler.get_log_prob(),
        blobs=None if blobs is None else np.asarray(blobs),
        acceptance=np.array(sampler.acceptance_fraction),
        entropy=(
            tuple(int(e) for e in seed.entropy)  # ty: ignore[not-iterable]
            if isinstance(seed.entropy, (list, tuple, np.ndarray))
            else int(seed.entropy)  # ty: ignore[invalid-argument-type]
        ),
        spawn_key=tuple(seed.spawn_key),
    )


@dataclass(frozen=True)
class ParameterSummary:
    """One sampled parameter over the kept chain."""

    name: str
    mean: float
    sd: float
    quantiles: tuple[float, ...]  # at QUANTILES
    mean_error: float  # Monte Carlo standard error of the mean
    quantile_errors: tuple[float, ...]
    tau: float
    ess: float
    stationary: bool
    at_lower: float  # fraction of samples At bound
    at_upper: float


@dataclass(frozen=True)
class Diagnosis:
    """The kept chain's summaries, and every reason a run is provisional (§7)."""

    burn_in: int
    tau: FloatArray
    tau_reliable: bool
    parameters: tuple[ParameterSummary, ...]
    correlation: FloatArray
    stuck: tuple[int, ...]
    reasons: tuple[str, ...]


def diagnose(
    run: EnsembleRun,
    names: tuple[str, ...],
    lower: FloatArray,
    upper: FloatArray,
    burn_in: int | None = None,
) -> Diagnosis:
    """Discard burn-in once, then summarize and check the kept chain.

    Without ``burn_in`` it is ``BURN_IN_TAUS`` × τ_max of the whole chain. When that
    leaves too little chain (a poorly mixing run), the second half is summarized
    and the run is provisional, saying why. An explicit ``burn_in`` that leaves too
    little chain is an error.
    """
    steps = run.chain.shape[0]
    longest = steps - _MIN_KEPT
    too_short = None
    if burn_in is None:
        with np.errstate(invalid="ignore", divide="ignore"):
            whole = emcee.autocorr.integrated_time(run.chain, tol=0)
        finite = whole[np.isfinite(whole)]
        burn_in = ceil(BURN_IN_TAUS * float(np.max(finite))) if finite.size else 0
        if burn_in > longest:
            too_short = (
                f"the chain ({steps} steps) is shorter than its automatic burn-in "
                f"({BURN_IN_TAUS} τ_max = {burn_in} steps); only its second half is "
                "summarized, as an indication"
            )
            burn_in = steps // 2
    elif not 0 <= burn_in <= longest:
        raise ValueError(
            f"A burn-in of {burn_in} steps leaves fewer than {_MIN_KEPT} of the {steps} "
            "steps; run longer."
        )
    if burn_in > longest:
        raise ValueError(f"A chain of {steps} steps is too short to summarize; run longer.")
    kept = run.chain[burn_in:]
    tau, reliable = ensemble.autocorrelation_times(kept)
    ess = ensemble.effective_sample_size(kept, tau)
    flat = kept.reshape(-1, kept.shape[2])
    quantiles = np.quantile(flat, QUANTILES, axis=0)
    errors = ensemble.quantile_errors(flat, np.array(QUANTILES), ess)
    sd = np.std(flat, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_error = sd / np.sqrt(ess)
    stationary = ensemble.stationary(kept, tau)
    at_lower, at_upper = ensemble.bound_pileup(flat, lower, upper)
    stuck = ensemble.stuck_walkers(run.acceptance)
    parameters = tuple(
        ParameterSummary(
            name=name,
            mean=float(np.mean(flat[:, j])),
            sd=float(sd[j]),
            quantiles=tuple(float(q) for q in quantiles[:, j]),
            mean_error=float(mean_error[j]),
            quantile_errors=tuple(float(e) for e in errors[:, j]),
            tau=float(tau[j]),
            ess=float(ess[j]),
            stationary=bool(stationary[j]),
            at_lower=float(at_lower[j]),
            at_upper=float(at_upper[j]),
        )
        for j, name in enumerate(names)
    )
    reasons = [too_short] if too_short else []
    undefined = [name for name, t in zip(names, tau, strict=True) if not np.isfinite(t)]
    if undefined:
        reasons.append(f"τ is undefined for {', '.join(undefined)}")
    elif not reliable:
        needed = ceil(ensemble.MIN_TAUS * float(np.max(tau)))
        reasons.append(
            f"the kept chain ({kept.shape[0]} steps) is shorter than "
            f"{ensemble.MIN_TAUS} τ_max ({needed} steps)"
        )
    drifting = [name for name, ok in zip(names, stationary, strict=True) if not ok]
    if drifting:
        reasons.append(f"the two halves of the kept chain disagree for {', '.join(drifting)}")
    if stuck.size:
        reasons.append(
            f"{stuck.size} of {run.acceptance.size} walkers are stuck "
            f"(acceptance under {ensemble.STUCK_FRACTION:.0%} of the median)"
        )
    return Diagnosis(
        burn_in=burn_in,
        tau=tau,
        tau_reliable=reliable,
        parameters=parameters,
        correlation=ensemble.correlation(flat),
        stuck=tuple(int(i) for i in stuck),
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class Exchange:
    """Two peaks whose order along their best-separated axis changes in some samples."""

    first: str
    second: str
    axis: str
    fraction: float


@dataclass(frozen=True)
class PosteriorSample:
    """One cluster's sampled posterior: the run, its diagnosis and its status."""

    cluster_id: int
    names: tuple[str, ...]
    peak_names: tuple[str, ...]
    settings: SamplerSettings
    run: EnsembleRun
    diagnosis: Diagnosis
    exchanges: tuple[Exchange, ...]
    determined: FloatArray  # (peaks,) fraction of kept samples with a determined amplitude
    amplitudes: AmplitudeSummaries

    @property
    def reasons(self) -> tuple[str, ...]:
        """Why the run is provisional; empty when it passed the convergence checks."""
        return self.diagnosis.reasons + tuple(
            f"peaks {e.first} and {e.second} exchange order along {e.axis} "
            f"in {e.fraction:.1%} of samples"
            for e in self.exchanges
        )

    @property
    def provisional(self) -> bool:
        """Whether any check failed; a provisional run gives no final intervals."""
        return bool(self.reasons)

    def withheld(self, name: str) -> str | None:
        """Why a parameter's interval is withheld whatever the status, or None."""
        peak = name.rsplit(".", 2)[0]
        for e in self.exchanges:
            if peak in (e.first, e.second):
                other = e.second if peak == e.first else e.first
                return f"exchanges identity with {other}"
        return None

    def caution(self, peak: str) -> str | None:
        """Why a peak's intervals need care although they are reported, or None.

        A peak whose amplitude is strongly correlated with another's (§6) is not
        separately resolved from it. That alone does not show its position or
        linewidth to be unidentifiable, so their intervals stay visible with this
        caution; withholding is reserved for exchanges (``withheld``).
        """
        correlated = self.amplitudes.correlated
        partners = sorted(
            {c.second for c in correlated if c.first == peak}
            | {c.first for c in correlated if c.second == peak}
        )
        if not partners:
            return None
        return f"not separately resolved from {', '.join(partners)} (amplitude |ρ| ≥ 0.9)"


def _positions(problem: SamplingProblem) -> dict[str, dict[str, str]]:
    """Each peak's chemical-shift parameter per axis, by name."""
    peaks: dict[str, dict[str, str]] = {}
    for name in problem.parameters:
        parts = name.rsplit(".", 2)  # {peak}.{axis}.{label}; a peak name may hold dots
        if len(parts) == _NAME_PARTS and parts[2] == "cs":
            peaks.setdefault(parts[0], {})[parts[1]] = name
    return peaks


def exchanges(problem: SamplingProblem, flat: FloatArray) -> tuple[Exchange, ...]:
    """Pairs of peaks with overlapping position bounds that exchange order (§7)."""
    index = {name: j for j, name in enumerate(problem.names)}

    def values(name: str) -> FloatArray:
        if name in index:
            return flat[:, index[name]]
        return np.full(flat.shape[0], problem.parameters[name].value)

    peaks = _positions(problem)
    found = []
    ordered = [p.name for p in problem.cluster.peaks if p.name in peaks]
    for i, first in enumerate(ordered):
        for second in ordered[i + 1 :]:
            axes = sorted(peaks[first].keys() & peaks[second].keys())
            a = [problem.parameters[peaks[first][axis]] for axis in axes]
            b = [problem.parameters[peaks[second][axis]] for axis in axes]
            if not axes or not all(
                p.min <= q.max and q.min <= p.max for p, q in zip(a, b, strict=True)
            ):
                continue
            separation = [
                abs(q.value - p.value) / (p.max - p.min) if p.max > p.min else 0.0
                for p, q in zip(a, b, strict=True)
            ]
            k = int(np.argmax(separation))
            reference = b[k].value - a[k].value
            fraction = ensemble.exchange_fraction(
                values(peaks[first][axes[k]]), values(peaks[second][axes[k]]), reference
            )
            if fraction > 0:
                found.append(Exchange(first, second, axes[k], fraction))
    return tuple(found)


def sample_posterior(
    problem: SamplingProblem,
    settings: SamplerSettings,
    seed: np.random.SeedSequence,
    *,
    progress: Callable[[SamplerProgress], None] | None = None,
    canceled: Callable[[], bool] | None = None,
    workers: int = 1,
) -> PosteriorSample:
    """Sample one cluster's posterior and check the result (§7)."""
    standard_errors = np.array([problem.parameters[n].stderr for n in problem.names])
    span = problem.upper - problem.lower
    fallback = np.where(
        np.isfinite(span), 1e-3 * span, np.maximum(1e-4 * np.abs(problem.start), 1e-6)
    )
    usable = np.isfinite(standard_errors) & (standard_errors > 0)
    scale = np.where(usable, _START_SPREAD * standard_errors, fallback)
    run = run_ensemble(
        LogPosterior(problem),
        problem.start,
        scale,
        problem.lower,
        problem.upper,
        walkers=settings.walkers,
        steps=settings.steps,
        seed=seed,
        progress=progress,
        canceled=canceled,
        workers=workers,
    )
    diagnosis = diagnose(run, problem.names, problem.lower, problem.upper, settings.burn_in)
    kept = run.chain[diagnosis.burn_in :]
    flat = kept.reshape(-1, kept.shape[2])
    assert run.blobs is not None
    # emcee squeezes a one-peak blob, so the peak axis is restored by count.
    peaks = len(problem.cluster.peaks)
    determined = run.blobs[diagnosis.burn_in :].reshape(-1, peaks).mean(axis=0)
    return PosteriorSample(
        cluster_id=problem.cluster.cluster_id,
        names=problem.names,
        peak_names=tuple(p.name for p in problem.cluster.peaks),
        settings=settings,
        run=run,
        diagnosis=diagnosis,
        exchanges=exchanges(problem, flat),
        determined=determined,
        amplitudes=_amplitudes(problem, run, diagnosis, determined),
    )


def _amplitudes(problem, run: EnsembleRun, diagnosis: Diagnosis, determined) -> AmplitudeSummaries:
    # The amplitude-draw stream is the run's second child sequence (``run_ensemble``).
    draws = np.random.SeedSequence(run.entropy, spawn_key=(*run.spawn_key, 1))
    ess = np.array([p.ess for p in diagnosis.parameters])
    effective = float(np.min(ess)) if np.isfinite(ess).all() else float("nan")
    kept = run.chain[diagnosis.burn_in :]
    return summarize_amplitudes(problem, kept, determined, draws, effective)


def regenerate_amplitudes(
    problem: SamplingProblem, posterior: PosteriorSample
) -> AmplitudeSummaries:
    """Recompute a recorded posterior's amplitude summaries from its chain and seed (§8).

    ``problem`` is rebuilt from the parent fit and its observations. With the same
    software versions the result equals the recorded summaries bit for bit.
    """
    return _amplitudes(problem, posterior.run, posterior.diagnosis, posterior.determined)


__all__ = [
    "BURN_IN_TAUS",
    "FORMULATION",
    "QUANTILES",
    "STREAMS",
    "Diagnosis",
    "EnsembleRun",
    "Exchange",
    "LogPosterior",
    "ParameterSummary",
    "PosteriorSample",
    "SamplerProgress",
    "SamplerSettings",
    "diagnose",
    "regenerate_amplitudes",
    "run_ensemble",
    "sample_posterior",
    "start_positions",
]
