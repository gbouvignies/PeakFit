"""The posterior MCMC samples for one cluster of a completed fit.

See ``docs/mcmc-formulation.md``. The nonlinear parameters θ the final fit varied are
uniform within its bounds, and the amplitudes are integrated out under the
conditional Jeffreys prior. θ then follows the profile likelihood
exp(−½ χ²_min(θ)), where χ²_min is the fit's own VarPro residual. Given θ, each
plane's amplitudes follow N(â(θ), σ² (ΦΦᵀ)⁻¹).
"""

from copy import deepcopy
from dataclasses import dataclass
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
    """Every plane's amplitudes given θ: N(mean, scale @ scaleᵀ), independently.

    ``scale`` spans only the strong directions of the shapes. A peak is not
    ``determined`` when its amplitude is improper (it loads on a null direction) or
    when weak directions, which cannot be computed reliably, carry a non-negligible
    share of its variance. Such an amplitude is never drawn.
    """

    log_likelihood: float
    mean: FloatArray  # (peaks, planes): â(θ), the minimum-norm solution
    scale: FloatArray  # (peaks, strong directions): σ V S⁻¹
    determined: np.ndarray  # (peaks,) bool

    def draw(self, rng: np.random.Generator, n: int) -> FloatArray:
        """``n`` draws of shape (n, peaks, planes); undetermined peaks are NaN."""
        z = rng.standard_normal((n, self.scale.shape[1], self.mean.shape[1]))
        draws = self.mean + np.einsum("kd,ndp->nkp", self.scale, z)
        draws[:, ~self.determined, :] = np.nan
        return draws


def conditional_amplitudes(
    shapes: FloatArray, data: FloatArray, noise: float
) -> ConditionalAmplitudes:
    """The profile log-likelihood and the conditional amplitude distribution at θ.

    The amplitude covariance is σ² Σᵢ vᵢvᵢᵀ / sᵢ² over the right singular vectors vᵢ
    of the shapes. A peak's amplitude is determined only if:

    - it has no loading above √ε on the null space: the directions below the fit's
      rank cutoff, and those a reduced SVD omits when there are fewer points than
      peaks;
    - weak directions (sᵢ ≤ √ε·s₁) contribute at most ``_WEAK_SHARE`` of its variance.
      A contribution scales as (vₖᵢ/sᵢ)², so a tiny loading can still dominate.
    """
    projection = factorize_shapes(shapes)
    mean = projection.solve(data)
    residuals = data - shapes.T @ mean
    rank = projection.rank
    singular = projection.singular_values[:rank]
    directions = projection.directions
    # Squared loading on the null space: the rows below the rank cutoff, plus the
    # complement of every computed row when the SVD has fewer rows than peaks.
    null = np.sum(directions[rank:] ** 2, axis=0)
    if directions.shape[0] < shapes.shape[0]:
        null += np.clip(1.0 - np.sum(directions**2, axis=0), 0.0, None)
    retained = directions[:rank]
    variance = (retained / singular[:, None]) ** 2  # (direction, peak) contributions
    weak = singular <= _WEAK * singular[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        weak_share = variance[weak].sum(axis=0) / variance.sum(axis=0)
    return ConditionalAmplitudes(
        log_likelihood=float(-0.5 * np.sum((residuals / noise) ** 2)),
        mean=mean,
        scale=noise * retained[~weak].T / singular[~weak],
        determined=(null <= _EPS) & (weak_share <= _WEAK_SHARE),
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
    )
    evaluation = outcome.analytical_evaluation
    if evaluation is not None:
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
