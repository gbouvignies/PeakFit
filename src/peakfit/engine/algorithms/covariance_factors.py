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
from enum import Enum
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
    def lineshape(self) -> tuple[FloatArray, FloatArray, FloatArray] | None:
        """The shape gain, cross gain and parameter covariance; None where withheld."""
        if self.shape_gain is None or self.cross_gain is None or self.parameter_covariance is None:
            return None
        return self.shape_gain, self.cross_gain, self.parameter_covariance

    @property
    def available(self) -> bool:
        """Whether the covariance between planes can be reconstructed."""
        return self.lineshape is not None


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
    finite = factors.lineshape is None or all(
        np.all(np.isfinite(values)) for values in factors.lineshape
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
    if factors.lineshape is None:
        return np.full((n_peaks, n_planes, n_peaks, n_planes), np.nan)
    shape_gain, cross_gain, theta = factors.lineshape
    gain = np.einsum("ms,mlp->slp", amplitudes, shape_gain)  # G_s
    cross = np.einsum("ms,mlp->slp", amplitudes, cross_gain)  # σ²MRT_sᵀ
    lineshape = (
        np.einsum("slp,pq,tkq->lskt", gain, theta, gain)
        - np.einsum("slp,tkp->lskt", cross, gain)
        - np.einsum("slp,tkp->lskt", gain, cross)
    )
    conditional = np.einsum("lk,st->lskt", factors.conditional, np.eye(n_planes))
    return conditional + lineshape


# By peak, (L, S, n_θ): G_s and σ²MRT_sᵀ, or their projections, with σ²C_θ.
type _PeakGains = tuple[FloatArray, FloatArray, FloatArray]


def _peak_gains(factors: CovarianceFactors, amplitudes: FloatArray) -> _PeakGains | None:
    """G_s and σ²MRT_sᵀ by peak, with σ²C_θ; None where the factors are withheld."""
    if factors.lineshape is None:
        return None
    shape_gain, cross_gain, theta = factors.lineshape
    gain = np.einsum("ms,mlp->lsp", amplitudes, shape_gain)
    cross = np.einsum("ms,mlp->lsp", amplitudes, cross_gain)
    return gain, cross, theta


def _peak_block(conditional: FloatArray, gains: _PeakGains, n_planes: int) -> FloatArray:
    """Each peak's (S × S) block: the conditional term plus the lineshape terms."""
    gain, cross, theta = gains
    lineshape = (
        np.einsum("lsp,pq,ltq->lst", gain, theta, gain)
        - np.einsum("lsp,ltp->lst", cross, gain)
        - np.einsum("lsp,ltp->lst", gain, cross)
    )
    return np.diag(conditional)[:, None, None] * np.eye(n_planes) + lineshape


def _standard_errors(conditional: FloatArray, gains: _PeakGains) -> FloatArray:
    """The square root of the diagonal of ``_peak_block``, without forming the rest.

    NaN where round-off makes a variance negative: near-coincident peaks can lose the
    reconstruction's positive definiteness to cancellation.
    """
    gain, cross, theta = gains
    variances = (
        np.diag(conditional)[:, None]
        + np.einsum("lsp,pq,lsq->ls", gain, theta, gain)
        - 2.0 * np.einsum("lsp,lsp->ls", cross, gain)
    )
    return np.sqrt(np.where(variances >= 0.0, variances, np.nan))


def peak_covariances_between_planes(
    factors: CovarianceFactors, amplitudes: FloatArray
) -> FloatArray:
    """Each peak's own covariance between planes, (L, S, S): Cov(a_{l,s}, a_{l,t}).

    The diagonal blocks of ``covariance_between_planes``, without forming the rest: what
    ratios to reference planes need (profile-uncertainty §4). All NaN where the factors
    are withheld.
    """
    n_peaks, n_planes = amplitudes.shape
    gains = _peak_gains(factors, amplitudes)
    if gains is None:
        return np.full((n_peaks, n_planes, n_planes), np.nan)
    return _peak_block(factors.conditional, gains, n_planes)


def marginal_standard_errors(factors: CovarianceFactors, amplitudes: FloatArray) -> FloatArray:
    """The marginal σ of every absolute amplitude, (L, S): √Cov(a_{l,s}, a_{l,s}).

    The diagonal of ``covariance_between_planes``, without forming the rest. All NaN
    where the factors are withheld; NaN where round-off makes a variance negative.
    """
    gains = _peak_gains(factors, amplitudes)
    if gains is None:
        return np.full(amplitudes.shape, np.nan)
    return _standard_errors(factors.conditional, gains)


def _projected_gains(factors: CovarianceFactors, amplitudes: FloatArray) -> _PeakGains | None:
    """PG_s and Pσ²MRT_sᵀ by peak, with σ²C_θ; None where withheld.

    P = I − uuᵀ removes each peak's own profile direction u = a/‖a‖ over the planes.
    """
    gains = _peak_gains(factors, amplitudes)
    if gains is None:
        return None
    norms = np.linalg.norm(amplitudes, axis=1, keepdims=True)
    # A zero profile has no direction to remove; its lineshape terms are zero anyway.
    units = np.divide(amplitudes, norms, out=np.zeros_like(amplitudes), where=norms > 0)

    def project(values: FloatArray) -> FloatArray:
        return values - units[:, :, None] * np.einsum("ls,lsp->lp", units, values)[:, None]

    gain, cross, theta = gains
    return project(gain), project(cross), theta


def relative_profile_covariance(factors: CovarianceFactors, amplitudes: FloatArray) -> FloatArray:
    """Σ_rel = cI + PLP of every peak's profile over the planes, (L, S, S).

    Exactly equivalent to the full covariance between planes for any downstream fit
    with a free scale (``docs/profile-uncertainty.md`` §3.3); cI for a single peak. All
    NaN where the factors are withheld.
    """
    n_peaks, n_planes = amplitudes.shape
    projected = _projected_gains(factors, amplitudes)
    if projected is None:
        return np.full((n_peaks, n_planes, n_planes), np.nan)
    return _peak_block(factors.conditional, projected, n_planes)


def profile_standard_errors(factors: CovarianceFactors, amplitudes: FloatArray) -> FloatArray:
    """The profile σ of every amplitude, (L, S): √(Σ_rel)_ss.

    The diagonal of ``relative_profile_covariance``, without forming the rest. All NaN
    where the factors are withheld; NaN where round-off makes a variance negative.
    """
    projected = _projected_gains(factors, amplitudes)
    if projected is None:
        return np.full(amplitudes.shape, np.nan)
    return _standard_errors(factors.conditional, projected)


# The correlated-profile flag (profile-uncertainty §6), set by validation T3 (§8.1):
# below both, independent profile σ misstate no downstream σ by more than 5%.
CORRELATED_MODE_THRESHOLD = 0.1  # largest correlated mode of Σ_rel, relative to c
PARTNER_CORRELATION_THRESHOLD = 0.15  # within-plane |ρ| with a cluster partner


class ProfileCorrelation(Enum):
    """Whether independent per-plane errors can describe a profile's uncertainty."""

    WITHIN_TOLERANCE = "within tolerance"
    CORRELATED = "correlated"  # the flag: a limit of the approximation, not of the data
    NOT_ASSESSED = "not assessed"  # the lineshape factors are withheld
    NO_PROFILE = "no profile"  # one plane


@dataclass(frozen=True, slots=True)
class PeakProfileCorrelation:
    """One peak's correlated-profile flag and the two diagnostics behind it.

    ``mode`` is the largest correlated mode of Σ_rel relative to c (NaN where not
    assessed). ``partner_correlation`` is the largest within-plane |ρ| with a cluster
    partner, ``partner`` that partner's index; 0 and None for a single peak.
    """

    status: ProfileCorrelation
    mode: float
    partner_correlation: float
    partner: int | None

    @property
    def flagged(self) -> bool:
        """Whether the profile's errors are correlated beyond the tolerance."""
        return self.status is ProfileCorrelation.CORRELATED


def profile_correlations(
    factors: CovarianceFactors, amplitudes: FloatArray
) -> tuple[PeakProfileCorrelation, ...]:
    """The correlated-profile flag of every peak, from the stored factors and amplitudes.

    A peak is flagged where the largest eigenvalue of PLP = Σ_rel − cI exceeds
    ``CORRELATED_MODE_THRESHOLD``·c, or its within-plane amplitude correlation with a
    partner exceeds ``PARTNER_CORRELATION_THRESHOLD`` in magnitude. Where the lineshape
    factors are withheld every peak is not assessed; with one plane there is no
    profile.
    """
    n_peaks, n_planes = amplitudes.shape
    conditional = np.asarray(factors.conditional, dtype=np.float64)
    variances = np.diag(conditional)
    with np.errstate(all="ignore"):
        correlation = np.abs(conditional / np.sqrt(np.outer(variances, variances)))
    np.fill_diagonal(correlation, 0.0)
    partners = [int(np.argmax(row)) if n_peaks > 1 else None for row in correlation]
    partner_rho = [
        0.0 if partner is None else float(correlation[peak, partner])
        for peak, partner in enumerate(partners)
    ]
    if n_planes < 2 or not factors.available:  # noqa: PLR2004
        status = ProfileCorrelation.NOT_ASSESSED if n_planes > 1 else ProfileCorrelation.NO_PROFILE
        modes = np.full(n_peaks, np.nan)
    else:
        lineshape = relative_profile_covariance(factors, amplitudes) - variances[
            :, None, None
        ] * np.eye(n_planes)
        with np.errstate(all="ignore"):
            modes = np.linalg.eigvalsh(lineshape)[:, -1] / variances
        status = None

    def judged(mode: float, rho: float) -> ProfileCorrelation:
        if status is not None:
            return status
        if mode > CORRELATED_MODE_THRESHOLD or rho > PARTNER_CORRELATION_THRESHOLD:
            return ProfileCorrelation.CORRELATED
        return ProfileCorrelation.WITHIN_TOLERANCE

    return tuple(
        PeakProfileCorrelation(judged(float(mode), rho), float(mode), rho, partner)
        for mode, rho, partner in zip(modes, partner_rho, partners, strict=True)
    )


__all__ = [
    "CORRELATED_MODE_THRESHOLD",
    "PARTNER_CORRELATION_THRESHOLD",
    "CovarianceFactors",
    "PeakProfileCorrelation",
    "ProfileCorrelation",
    "covariance_between_planes",
    "covariance_factors",
    "marginal_standard_errors",
    "peak_covariances_between_planes",
    "peak_derivatives",
    "profile_correlations",
    "profile_standard_errors",
    "relative_profile_covariance",
]
