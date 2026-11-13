"""Per-cluster factors of the first-order amplitude covariance between planes.

See ``docs/profile-uncertainty.md`` §3.1. For a cluster with shapes Φ (L × n), write
M = (ΦΦᵀ)⁻¹Φ, P⊥ = I − ΦᵀM, E_m for peak m's shape derivatives with respect to the
varied lineshape parameters θ, D_s = Σ_m a_{m,s} E_m, H = Σ_s D_sᵀP⊥D_s and
T_s = H⁻¹D_sᵀP⊥. Then

    Cov(a_s, a_t) = σ²[δ_st MRMᵀ − MRT_sᵀG_tᵀ − G_sT_tRMᵀ + G_s C_θ G_tᵀ],

with G_s = M D_s and C_θ = Σ_u T_u R T_uᵀ: the amplitude block of the OLS sandwich over
all parameters. It is fixed, whatever the number of planes, by MRMᵀ, the tensors M E_m
and M R P⊥ E_m H⁻¹, and C_θ, together with the amplitudes.

- **Shared parameters** (a cluster's phase) are one θ, with each peak's own derivative.
- **Fixed parameters** (by constraints or fit-step rules) are not in θ.
- **Parameters at an active bound** (``varpro.at_active_bound``) are not in θ either: at
  a constrained optimum they are locally fixed. A parameter whose error merely crosses
  a bound stays in θ.
- **Withholding** adds no rule. Where the conditional amplitude covariance is withheld
  (``factorize_shapes``), everything is. Where the rank and conditioning rule of
  ``identifiable_covariance``, applied to the projected Jacobian [P⊥D_s] that gives
  H, withholds, the lineshape terms are: the factors keep only the conditional block,
  and the covariance between planes is unavailable.
- **Numerical failure never fails a fit.** Non-finite factors (an overflow), or a
  derivative evaluation that raises, withhold the lineshape terms; the evaluation, its
  amplitudes and its published uncertainties are unaffected.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.algorithms.varpro import identifiable_covariance
from peakfit.engine.domain.param_map import ParameterMap

if TYPE_CHECKING:
    from collections.abc import Sequence

    from peakfit.engine.domain.cluster import Cluster
    from peakfit.engine.domain.params_scalar import Parameters
    from peakfit.shared.typing import FloatArray


@dataclass(frozen=True, slots=True)
class CovarianceFactors:
    """The factors of one cluster's amplitude covariance, in absolute units.

    ``conditional`` is σ²MRMᵀ, the published conditional amplitude covariance (L × L).
    The lineshape factors are None where withheld; otherwise, with n_θ free lineshape
    parameters, ``shape_gain[m]`` is M E_m and ``cross_gain[m]`` σ²M R P⊥ E_m H⁻¹, both
    L × n_θ, and ``parameter_covariance`` is σ²C_θ (n_θ × n_θ).
    """

    conditional: FloatArray
    shape_gain: FloatArray | None
    cross_gain: FloatArray | None
    parameter_covariance: FloatArray | None

    @property
    def available(self) -> bool:
        """Whether the covariance between planes can be reconstructed."""
        return self.parameter_covariance is not None


def peak_derivatives(cluster: Cluster, params: Parameters, names: Sequence[str]) -> FloatArray:
    """Each peak's own shape derivatives, (L, n_θ, n), with respect to ``names``.

    A parameter shared by several peaks gets each peak's own contribution, not their
    sum, so that the plane's amplitudes can weight them.
    """
    param_map = ParameterMap.from_names(list(names))
    derivatives = np.zeros((len(cluster.peaks), len(names), cluster.n_points))
    for index, peak in enumerate(cluster.peaks):
        cluster.evaluate_peaks(
            [peak],
            params,
            cluster.grid_indices,
            d_matrix=derivatives[index],
            param_map=param_map,
        )
    return derivatives


def _withheld(conditional: FloatArray) -> CovarianceFactors:
    return CovarianceFactors(conditional, None, None, None)


def covariance_factors(
    shapes: FloatArray,
    derivatives: FloatArray,
    amplitudes: FloatArray,
    conditional: FloatArray,
    noise: float,
    correlation: FloatArray | None,
) -> CovarianceFactors:
    """The factors from a cluster's fitted state.

    ``shapes`` is Φ (L × n), ``derivatives`` each peak's E (L × n_θ × n), ``amplitudes``
    the fitted a (L × S), and ``conditional`` the published conditional amplitude
    covariance, kept as it is. ``correlation`` is R between the points, None meaning
    R = I. The derivatives are with respect to the free lineshape parameters only.
    """
    conditional = np.asarray(conditional, dtype=np.float64)
    if not np.all(np.isfinite(conditional)):
        return _withheld(conditional)
    with np.errstate(all="ignore"):
        factors = _lineshape_factors(
            shapes, derivatives, amplitudes, conditional, noise, correlation
        )
    finite = all(
        np.all(np.isfinite(values))
        for values in (factors.shape_gain, factors.cross_gain, factors.parameter_covariance)
        if values is not None
    )
    return factors if finite else _withheld(conditional)


def _lineshape_factors(
    shapes: FloatArray,
    derivatives: FloatArray,
    amplitudes: FloatArray,
    conditional: FloatArray,
    noise: float,
    correlation: FloatArray | None,
) -> CovarianceFactors:
    n_theta = derivatives.shape[1]
    variance = noise**2
    # Full rank here: the conditional covariance is published.
    basis, _ = np.linalg.qr(shapes.T)  # (n, L)
    pinv = np.linalg.solve(shapes @ shapes.T, shapes)  # M = (ΦΦᵀ)⁻¹Φ (L × n)
    # D_s, then P⊥D_s, for every plane, the plane first: (S, n, n_θ).
    jacobians = np.einsum("ls,lpi->sip", amplitudes, derivatives)
    projected = jacobians - basis @ (basis.T @ jacobians)
    hessian = np.einsum("sip,siq->pq", projected, projected)
    # H⁻¹, from the projected Jacobian under the existing withholding rule.
    inverse = (
        identifiable_covariance(projected.reshape(-1, n_theta), None)
        if n_theta
        else np.zeros((0, 0))
    )
    if not np.all(np.isfinite(inverse)):
        return _withheld(conditional)
    if correlation is None:
        pinv_r = pinv  # MR with R = I; then M R P⊥ = 0 up to round-off
        middle = hessian
    else:
        pinv_r = pinv @ correlation
        middle = np.einsum("sip,siq->pq", projected, correlation @ projected)
    # M R P⊥ = MR − (MR Q)Qᵀ, with Q the basis of the shapes.
    pinv_r_perp = pinv_r - (pinv_r @ basis) @ basis.T
    shape_gain = np.einsum("li,mpi->mlp", pinv, derivatives)
    cross_gain = variance * np.einsum("li,mpi->mlp", pinv_r_perp, derivatives) @ inverse
    parameter_covariance = variance * inverse @ middle @ inverse
    return CovarianceFactors(conditional, shape_gain, cross_gain, parameter_covariance)


def covariance_between_planes(factors: CovarianceFactors, amplitudes: FloatArray) -> FloatArray:
    """Cov(a_{l,s}, a_{k,t}) for every pair of peaks and planes, (L, S, L, S).

    All NaN where the factors are withheld.
    """
    n_peaks, n_planes = amplitudes.shape
    if (
        factors.shape_gain is None
        or factors.cross_gain is None
        or factors.parameter_covariance is None
    ):
        return np.full((n_peaks, n_planes, n_peaks, n_planes), np.nan)
    gain = np.einsum("ms,mlp->slp", amplitudes, factors.shape_gain)  # G_s
    cross = np.einsum("ms,mlp->slp", amplitudes, factors.cross_gain)  # σ²MRT_sᵀ
    lineshape = (
        np.einsum("slp,pq,tkq->lskt", gain, factors.parameter_covariance, gain)
        - np.einsum("slp,tkp->lskt", cross, gain)
        - np.einsum("slp,tkp->lskt", gain, cross)
    )
    conditional = np.einsum("lk,st->lskt", factors.conditional, np.eye(n_planes))
    return conditional + lineshape


__all__ = [
    "CovarianceFactors",
    "covariance_between_planes",
    "covariance_factors",
    "peak_derivatives",
]
