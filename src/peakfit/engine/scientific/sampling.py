"""The posterior MCMC samples for one cluster of a completed fit.

See ``docs/mcmc-formulation.md``. The nonlinear parameters θ the final fit varied are
uniform within its bounds, and the amplitudes are integrated out under the
conditional Jeffreys prior. θ then follows the profile likelihood
exp(−½ χ²_min(θ)), where χ²_min is the fit's own VarPro residual. Given θ, each
plane's amplitudes follow N(â(θ), σ² (ΦΦᵀ)⁻¹).
"""

from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.algorithms.linear_algebra import factorize_shapes
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.scientific.completed import FinalClusterSamples
from peakfit.shared.exceptions import require_close

if TYPE_CHECKING:
    from peakfit.engine.scientific.completed import CompletedFit
    from peakfit.shared.typing import FloatArray

_EPS = float(np.finfo(float).eps)
# Directions of the shape matrix at or below this fraction of the strongest are weak:
# the fit itself withholds its amplitude covariance there (``factorize_shapes``).
_WEAK = float(np.sqrt(_EPS))
# A determined amplitude may owe at most this fraction of its conditional variance
# to weak directions, which the draws omit: its standard deviation is then
# understated by at most 1 - sqrt(1 - 1e-4), 0.005%.
_WEAK_SHARE = 1e-4


@dataclass(frozen=True)
class ConditionalAmplitudes:
    """Every plane's amplitudes given θ: N(mean, σ² Σᵢ vᵢvᵢᵀ / sᵢ²), independently.

    ``directions`` are the right singular vectors vᵢ of the shapes, as rows; the
    first ``rank`` are retained, with ``singular`` values sᵢ. A combination of
    amplitudes (one peak's, or a sum over peaks) is determined only if:

    - it has no loading above √ε on the null space: the rows below the fit's rank
      cutoff, and those a reduced SVD omits when there are fewer points than peaks;
    - weak directions (sᵢ ≤ √ε·s₁), which cannot be computed reliably, contribute
      at most ``_WEAK_SHARE`` of its variance. A contribution scales as (w·vᵢ/sᵢ)²,
      so a tiny loading can still dominate.

    Draws omit weak directions; a determined amplitude is never drawn otherwise.
    """

    log_likelihood: float
    mean: FloatArray  # (peaks, planes): â(θ), the minimum-norm solution
    noise: float
    singular: FloatArray  # (rank,)
    directions: FloatArray  # (rows, peaks), rows = min(peaks, points)

    def determines(self, weights: FloatArray) -> np.ndarray:
        """Whether each combination (a row of ``weights`` over peaks) is determined."""
        w = np.atleast_2d(weights)
        rank = self.singular.size
        loads = w @ self.directions.T  # (combinations, rows)
        squared = np.sum(w**2, axis=1)
        null = np.sum(loads[:, rank:] ** 2, axis=1)
        if self.directions.shape[0] < self.directions.shape[1]:
            null += np.clip(squared - np.sum(loads**2, axis=1), 0.0, None)
        variance = (loads[:, :rank] / self.singular) ** 2
        weak = self.singular <= _WEAK * self.singular[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            share = variance[:, weak].sum(axis=1) / variance.sum(axis=1)
        return (null <= _EPS * squared) & (share <= _WEAK_SHARE)

    @cached_property
    def determined(self) -> np.ndarray:
        """Whether each peak's own amplitude is determined, (peaks,)."""
        return self.determines(np.eye(self.mean.shape[0]))

    @cached_property
    def scale(self) -> FloatArray:
        """σ V S⁻¹ over the strong directions, (peaks, strong directions)."""
        rank = self.singular.size
        strong = self.singular > _WEAK * self.singular[0]
        return self.noise * self.directions[:rank][strong].T / self.singular[strong]

    def draw(self, rng: np.random.Generator, n: int, *, withhold: bool = True) -> FloatArray:
        """``n`` draws of shape (n, peaks, planes).

        Undetermined peaks are NaN unless ``withhold`` is False, for callers that
        combine peaks into a determined combination (``determines``) themselves.
        """
        z = rng.standard_normal((n, self.scale.shape[1], self.mean.shape[1]))
        draws = self.mean + np.einsum("kd,ndp->nkp", self.scale, z)
        if withhold:
            draws[:, ~self.determined, :] = np.nan
        return draws


def conditional_amplitudes(
    shapes: FloatArray, data: FloatArray, noise: float
) -> ConditionalAmplitudes:
    """The profile log-likelihood and the conditional amplitude distribution at θ."""
    projection = factorize_shapes(shapes)
    mean = projection.solve(data)
    residuals = data - shapes.T @ mean
    return ConditionalAmplitudes(
        log_likelihood=float(-0.5 * np.sum((residuals / noise) ** 2)),
        mean=mean,
        noise=noise,
        singular=projection.singular_values[: projection.rank],
        directions=projection.directions,
    )


@dataclass(frozen=True)
class SamplingProblem:
    """One cluster's posterior over the parameters its final fit varied.

    ``cluster`` carries the fit's geometry, observations and frozen cross-talk
    corrections; ``parameters`` its final values, fixed ones included. θ is ordered
    as ``names``; the prior's support is [``lower``, ``upper``].
    """

    cluster: Cluster
    parameters: Parameters
    names: tuple[str, ...]
    lower: FloatArray
    upper: FloatArray
    noise: float
    data: FloatArray  # the real corrected data the fit used, (points, planes)
    reduced_chi_squared: float  # the final fit's, reported alongside; never scales σ

    @property
    def start(self) -> FloatArray:
        """The final fit's θ̂."""
        return np.array([self.parameters[name].value for name in self.names])

    def conditional(self, theta: FloatArray) -> ConditionalAmplitudes | None:
        """The likelihood and amplitudes at θ, or None outside the prior's support."""
        if not np.all((self.lower <= theta) & (theta <= self.upper)):
            return None
        parameters = self.parameters.copy()
        for name, value in zip(self.names, theta, strict=True):
            parameters[name].value = float(value)
        return conditional_amplitudes(self.cluster.evaluate(parameters), self.data, self.noise)

    def log_likelihood(self, theta: FloatArray) -> float:
        """−½ χ²_min(θ) within the bounds, −∞ outside."""
        conditional = self.conditional(theta)
        return -np.inf if conditional is None else conditional.log_likelihood


def sampling_problem(completed: CompletedFit, cluster_id: int) -> SamplingProblem:
    """Build the posterior of one target cluster; only that cluster must be usable.

    The fit's χ² is recomputed at θ̂ and must match the recorded value, so the
    sampled likelihood is the parent's by construction.
    """
    outcome = completed.outcome.cluster(cluster_id)
    if not outcome.usable:
        raise ValueError(
            f"Cluster {cluster_id} is unusable ({outcome.unusable_reason}); "
            "adjust and refit it before sampling."
        )
    sample = next(s for s in completed.samples if s.cluster_id == cluster_id)
    if not isinstance(sample, FinalClusterSamples):
        raise ValueError("Sampling requires the cluster's verified observations.")
    geometry = next(c for c in completed.model_snapshot.clusters if c.cluster_id == cluster_id)
    cluster = Cluster(
        cluster_id, list(deepcopy(geometry.peaks)), list(sample.grid_indices), sample.data.copy()
    )
    cluster.corrections = sample.neighbor_corrections.copy()
    final = Parameters()
    for p in outcome.final_nonlinear_parameters:
        final.add(
            p.name,
            value=p.value,
            min_value=p.min,
            max_value=p.max,
            vary=p.vary,
            stderr=p.standard_error,
            unit=p.unit,
        )
    parameters = final.copy(include=set(cluster.parameter_names(final)))
    names = tuple(name for name, p in parameters.items() if p.vary)
    if not names:
        raise ValueError(f"Cluster {cluster_id} has no varied nonlinear parameters to sample.")
    evaluation = outcome.analytical_evaluation
    if evaluation is None:  # a usable cluster always records one
        raise ValueError(f"Cluster {cluster_id} has no analytical evaluation; refit it.")
    data = np.asarray(cluster.corrected_data.real, dtype=np.float64)
    data.flags.writeable = False
    problem = SamplingProblem(
        cluster=cluster,
        parameters=parameters,
        names=names,
        lower=np.array([parameters[name].min for name in names]),
        upper=np.array([parameters[name].max for name in names]),
        noise=completed.outcome.noise,
        data=data,
        reduced_chi_squared=evaluation.statistics.reduced_chi_squared,
    )
    require_close(
        -2.0 * problem.log_likelihood(problem.start),
        evaluation.statistics.chi_squared,
        rtol=1e-10,
        atol=0.0,
        message=f"Cluster {cluster_id}: the sampled likelihood does not reproduce the "
        "fit's recorded chi-squared; MCMC cannot sample this result. Refit the dataset.",
    )
    return problem


__all__ = [
    "ConditionalAmplitudes",
    "SamplingProblem",
    "conditional_amplitudes",
    "sampling_problem",
]
