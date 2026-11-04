"""Amplitude summaries over a sampled posterior.

See ``docs/mcmc-formulation.md`` §5 and §6. For up to ``AMPLITUDE_SAMPLES`` kept θ
samples, evenly spaced, the conditional amplitude distribution is evaluated:

- Moments use the law of total variance: the mean of the conditional covariances
  plus the covariance of â(θ) across samples.
- Quantiles come from one conditional draw per sample.
- A peak whose amplitude is undetermined in any kept sample is withheld, with the
  fraction of samples affected. Samples are never dropped.
- Determined member amplitudes with posterior |ρ| ≥ ``CORRELATION_FLAG`` in a plane
  are flagged as not separately determined.
- Each such group, and each group of undetermined peaks whose sum is determined, is
  reported as a summed amplitude.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.diagnostics import ensemble

if TYPE_CHECKING:
    from peakfit.engine.scientific.sampling import ConditionalAmplitudes, SamplingProblem
    from peakfit.shared.typing import FloatArray

# Kept θ samples, evenly spaced, at which the amplitude distribution is evaluated.
AMPLITUDE_SAMPLES = 4000
# |ρ| at or above which two member amplitudes are not separately determined; the same
# threshold as the amplitude-correlation highlight.
CORRELATION_FLAG = 0.9
QUANTILES = (0.025, 0.16, 0.5, 0.84, 0.975)


@dataclass(frozen=True)
class AmplitudeSummary:
    """One amplitude, or a sum of amplitudes, in one plane."""

    peaks: tuple[str, ...]  # one peak, or the peaks of a sum
    plane: int
    mean: float
    sd: float
    quantiles: tuple[float, ...]  # at QUANTILES
    mean_error: float  # Monte Carlo standard errors
    quantile_errors: tuple[float, ...]


@dataclass(frozen=True)
class Correlation:
    """Two determined amplitudes too correlated in one plane to be read separately."""

    first: str
    second: str
    plane: int
    rho: float


@dataclass(frozen=True)
class AmplitudeSummaries:
    """Every reportable amplitude and sum, and what was withheld or flagged."""

    samples: int  # θ samples evaluated
    effective: float  # sample size behind the Monte Carlo errors
    amplitudes: tuple[AmplitudeSummary, ...]  # determined peaks, every plane
    undetermined: tuple[tuple[str, float], ...]  # withheld peaks, fraction of samples
    correlated: tuple[Correlation, ...]
    sums: tuple[AmplitudeSummary, ...]


def _summary(peaks, plane, mean, sd, draws, effective) -> AmplitudeSummary:
    quantiles = np.quantile(draws, QUANTILES)
    errors = ensemble.quantile_errors(draws[:, None], np.array(QUANTILES), np.array([effective]))
    return AmplitudeSummary(
        peaks=peaks,
        plane=plane,
        mean=float(mean),
        sd=float(sd),
        quantiles=tuple(float(q) for q in quantiles),
        mean_error=float(sd / np.sqrt(effective)),
        quantile_errors=tuple(float(e) for e in errors[:, 0]),
    )


def _groups(pairs: list[tuple[int, int]], n: int) -> list[tuple[int, ...]]:
    """Connected components (of two or more peaks) of the given links."""
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        parent[root(a)] = root(b)
    components: dict[int, list[int]] = {}
    for i in range(n):
        components.setdefault(root(i), []).append(i)
    return [tuple(c) for c in components.values() if len(c) > 1]


def _correlations(total: FloatArray, whole: np.ndarray, names) -> list[Correlation]:
    """Pairs of determined amplitudes with |ρ| at or above the flag, per plane."""
    found = []
    determined = np.flatnonzero(whole)
    for plane, matrix in enumerate(total):
        sd = np.sqrt(np.diag(matrix))
        for i, a in enumerate(determined):
            for b in determined[i + 1 :]:
                rho = matrix[a, b] / (sd[a] * sd[b])
                if abs(rho) >= CORRELATION_FLAG:
                    found.append(Correlation(names[a], names[b], plane, float(rho)))
    return found


def _sums(group, names, conditionals, means, draws, total, n_eff) -> list[AmplitudeSummary]:
    """A group's summed amplitude in every plane, if the sum is determined throughout."""
    weights = np.zeros(means.shape[1])
    weights[list(group)] = 1.0
    if not all(c.determines(weights)[0] for c in conditionals):
        return []
    members = tuple(names[k] for k in group)
    return [
        _summary(
            members,
            plane,
            (means[:, :, plane] @ weights).mean(),
            np.sqrt(weights @ total[plane] @ weights),
            draws[:, :, plane] @ weights,
            n_eff,
        )
        for plane in range(means.shape[2])
    ]


def summarize_amplitudes(
    problem: SamplingProblem,
    kept: FloatArray,
    determined: FloatArray,
    seed: np.random.SeedSequence,
    effective: float,
) -> AmplitudeSummaries:
    """Summaries from kept θ samples (n, parameters).

    ``determined`` is each peak's fraction of kept samples with a determined
    amplitude, over every kept sample. ``seed`` is the run's amplitude-draw stream;
    ``effective`` the sampler's smallest effective sample size.
    """
    names = tuple(p.name for p in problem.cluster.peaks)
    index = np.unique(np.linspace(0, kept.shape[0] - 1, min(kept.shape[0], AMPLITUDE_SAMPLES)))
    rng = np.random.default_rng(seed)
    conditionals: list[ConditionalAmplitudes] = []
    for theta in kept[index.astype(int)]:
        conditional = problem.conditional(theta)
        if conditional is None:  # kept samples lie inside the support
            raise ValueError("A kept MCMC sample lies outside the parameter bounds.")
        conditionals.append(conditional)
    means = np.stack([c.mean for c in conditionals])  # (n, peaks, planes)
    draws = np.concatenate([c.draw(rng, 1, withhold=False) for c in conditionals])
    covariance = np.mean([c.scale @ c.scale.T for c in conditionals], axis=0)  # per plane
    n, peaks, planes = means.shape
    n_eff = float(min(n, effective)) if np.isfinite(effective) else float(n)
    whole = determined >= 1.0
    amplitudes = []
    totals = []
    for plane in range(planes):
        spread = np.cov(means[:, :, plane].T, ddof=0) if n > 1 else np.zeros((peaks, peaks))
        totals.append(covariance + np.atleast_2d(spread))
    total = np.stack(totals)  # (planes, peaks, peaks)
    for k in np.flatnonzero(whole):
        for plane in range(planes):
            sd = np.sqrt(total[plane, k, k])
            amplitudes.append(
                _summary(
                    (names[k],),
                    plane,
                    means[:, k, plane].mean(),
                    sd,
                    draws[:, k, plane],
                    n_eff,
                )
            )
    correlated = _correlations(total, whole, names)
    index_of = {name: k for k, name in enumerate(names)}
    links = [(index_of[c.first], index_of[c.second]) for c in correlated]
    # Undetermined peaks group with those they are undetermined alongside.
    links += [
        (int(a), int(b))
        for c in conditionals
        for a in np.flatnonzero(~c.determined)
        for b in np.flatnonzero(~c.determined)
        if a < b
    ]
    sums = [
        summary
        for group in _groups(links, peaks)
        for summary in _sums(group, names, conditionals, means, draws, total, n_eff)
    ]
    return AmplitudeSummaries(
        samples=n,
        effective=n_eff,
        amplitudes=tuple(amplitudes),
        undetermined=tuple((names[k], float(1.0 - determined[k])) for k in np.flatnonzero(~whole)),
        correlated=tuple(correlated),
        sums=tuple(sums),
    )


__all__ = [
    "AMPLITUDE_SAMPLES",
    "CORRELATION_FLAG",
    "AmplitudeSummaries",
    "AmplitudeSummary",
    "Correlation",
    "summarize_amplitudes",
]
