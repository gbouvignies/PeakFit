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

# Directions of the shape matrix weaker than this, relative to the strongest, leave
# the amplitudes they involve undetermined. It is the threshold at which the fit
# itself withholds its amplitude covariance (``factorize_shapes``).
_WEAK = float(np.sqrt(np.finfo(float).eps))


@dataclass(frozen=True)
class ConditionalAmplitudes:
    """Every plane's amplitudes given θ: N(mean, scale @ scaleᵀ), independently.

    ``scale`` spans only the well-determined directions of the shapes. A peak that
    loads on a weak direction is not ``determined``: its amplitude is improper, or
    not reliably computable, along that direction, so it is never drawn.
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
    """The profile log-likelihood and the conditional amplitude distribution at θ."""
    projection = factorize_shapes(shapes)
    mean = projection.solve(data)
    residuals = data - shapes.T @ mean
    singular = projection.singular_values
    weak = singular <= _WEAK * singular[0]
    loads = np.abs(projection.directions[weak]) > _WEAK
    strong = projection.directions[~weak]
    return ConditionalAmplitudes(
        log_likelihood=float(-0.5 * np.sum((residuals / noise) ** 2)),
        mean=mean,
        scale=noise * strong.T / singular[~weak],
        determined=~np.any(loads, axis=0),
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
