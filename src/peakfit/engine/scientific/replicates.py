"""Replicate planes: empirical checks of a published run's uncertainties on real data.

Two methods, neither of which refits or changes any error:

- **Duplicate planes** (any series): planes whose recorded plane values are exactly
  equal. The contrasts between copies are compared with the covariance the stored
  covariance factors predict for them.
- **CEST baseline planes:** see below.

**CEST baseline planes.** Far from a peak's own offset, a CEST profile is a flat or
slowly varying baseline, so its planes are approximate replicates of one another.
Their plane-to-plane scatter is
an empirical measure of the amplitude error, compared here with the published profile
σ (``docs/profile-uncertainty.md`` §5.1, validation T4).

- **Scatter:** the robust SD of second differences across offsets, divided by √6.
  Second differences remove slow trends and common-mode errors; the robust SD limits
  the influence of minor-state dips.
- **Limits:** it measures repeatability, not accuracy. An error common to every
  baseline plane (a lineshape misfit, a bias from linear prediction) is invisible.
  Drift, residual saturation and minor-state dips can only inflate the scatter.

Values are those recorded: nothing is refitted.

**Duplicate planes.** For each peak, all within-group orthonormal contrasts C are
stacked, k = Σ_g (m_g − 1) for groups of m_g copies, and whitened jointly by their
covariance CΣCᵀ, Σ being the peak's covariance between planes reconstructed from the
covariance factors: q = (Ca)ᵀ(CΣCᵀ)⁻¹(Ca), with E[q] = k at first order. Joint
whitening removes the correlation between groups from shared parameters; q depends on
neither the contrast basis nor the plane order. The pooled RMS √(Σq / Σk) is 1 when
the uncertainty of contrasts is calibrated.

- **Pools:** peaks without and with the correlated-profile flag, kept apart.
- **Exclusions:** unusable clusters, withheld covariance factors, and a contrast
  covariance that is not positive definite or has a direction weaker than √ε of the
  largest. That is the codebase's weak-direction convention s ≤ √ε·s₁ applied to the
  singular values of the matrix inverted, here the eigenvalues of CΣCᵀ (equivalently
  cond ≥ 1/√ε, as in ``evaluation``). Nothing is regularized.
- **Limits:** repeatability, not accuracy; contrasts, not marginal σ (a shared scale
  uncertainty cancels); raw amplitudes, not normalized profiles. Peaks of one cluster
  have correlated contrasts, so the effective sample is smaller than the count.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from peakfit.engine.algorithms.covariance_factors import (
    peak_covariances_between_planes,
    profile_correlations,
    profile_standard_errors,
)
from peakfit.engine.domain.param_id import ParameterId
from peakfit.engine.domain.spectrum import SpectralParameters
from peakfit.engine.scientific.profiles import CEST_REFERENCE_HZ

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from peakfit.engine.algorithms.covariance_factors import CovarianceFactors
    from peakfit.engine.scientific.final_outcome import FinalFitOutcome
    from peakfit.shared.typing import FloatArray

BASELINE_HZ = 400.0  # planes this far from the peak's own offset are baseline
MIN_BASELINE_PLANES = 20
_MAD_TO_SD = 1.4826


@dataclass(frozen=True, slots=True)
class BaselineScatter:
    """One peak's baseline scatter against its stated profile σ, in amplitude units."""

    planes: int  # baseline planes used
    observed: float  # robust SD of second differences / √6
    stated: float  # median profile σ over the baseline planes
    level: float  # median baseline amplitude

    @property
    def ratio(self) -> float:
        """Observed / stated: 1 when the profile σ is calibrated."""
        return self.observed / self.stated


def robust_sd(values: np.ndarray) -> float:
    """SD from the median absolute deviation."""
    return float(_MAD_TO_SD * np.median(np.abs(values - np.median(values))))


def baseline_scatter(
    offsets: np.ndarray,
    amplitudes: np.ndarray,
    errors: np.ndarray,
    *,
    peak_offset: float,
    reference: np.ndarray,
    window: float = BASELINE_HZ,
) -> BaselineScatter | str:
    """Compare one peak's baseline scatter with its profile σ, or say why not.

    ``offsets`` (Hz from the carrier), ``amplitudes``, ``errors`` (profile σ) and the
    ``reference`` mask are per plane, in any order. Baseline planes are those more than
    ``window`` Hz from ``peak_offset``, references excluded, taken in offset order.
    """
    offsets = np.asarray(offsets, dtype=float)
    order = np.argsort(offsets, kind="stable")
    keep = (np.abs(offsets - peak_offset) > window) & ~np.asarray(reference, dtype=bool)
    planes = order[keep[order]]
    if planes.size < MIN_BASELINE_PLANES:
        return f"{planes.size} baseline planes, at least {MIN_BASELINE_PLANES} needed"
    stated = float(np.median(np.asarray(errors, dtype=float)[planes]))
    if not np.isfinite(stated) or stated <= 0:
        return "profile σ unavailable"
    a = np.asarray(amplitudes, dtype=float)[planes]
    level = float(np.median(a))
    if level <= 0:
        return "baseline intensity not positive"
    second = (a[:-2] - 2 * a[1:-1] + a[2:]) / np.sqrt(6.0)
    return BaselineScatter(int(planes.size), robust_sd(second), stated, level)


def run_baseline_scatter(
    outcome: FinalFitOutcome,
    context: Mapping[str, Any],
    *,
    reference: list[int] | None = None,
    window: float = BASELINE_HZ,
) -> dict[str, BaselineScatter | str]:
    """Every recorded peak's baseline scatter, from a published run.

    The plane values are the CEST offsets in Hz from the carrier of the first spectral
    axis, whose fitted chemical shift gives each peak's own offset. ``reference`` lists
    reference planes by their index in offset order; by default, the planes at least
    ``CEST_REFERENCE_HZ`` off resonance, with no fallback: a run without them has none.
    """
    offsets = np.asarray(context["plane_values"], dtype=float)
    if reference is None:
        is_reference = np.abs(offsets) >= CEST_REFERENCE_HZ
    else:
        is_reference = np.zeros(offsets.size, dtype=bool)
        is_reference[np.argsort(offsets, kind="stable")[reference]] = True
    axis = SpectralParameters.model_validate(context["spectral_parameters"][1])
    results: dict[str, BaselineScatter | str] = {}
    for cluster in outcome.clusters:
        evaluation = cluster.analytical_evaluation
        if not cluster.usable or evaluation is None:
            for name in cluster.peak_names:
                results[name] = "no usable fit"
            continue
        shifts = {p.name: p.value for p in cluster.final_nonlinear_parameters}
        errors = profile_standard_errors(evaluation.covariance_factors, evaluation.amplitudes)
        for peak, name in enumerate(cluster.peak_names):
            shift = shifts[ParameterId(peak_name=name, axis=axis.label, label="cs").name]
            peak_offset = shift * axis.obs - axis.car
            results[name] = baseline_scatter(
                offsets,
                evaluation.amplitudes[peak],
                errors[peak],
                peak_offset=peak_offset,
                reference=is_reference,
                window=window,
            )
    return results


# Below this many contrasts, the independent-case SE of the RMS, 1/√(2n), exceeds 0.1.
MIN_DUPLICATE_CONTRASTS = 50
_WEAK = float(np.sqrt(np.finfo(float).eps))

UNUSABLE_CLUSTER = "unusable cluster"
FACTORS_WITHHELD = "covariance factors withheld"
NOT_POSITIVE_DEFINITE = "contrast covariance not positive definite"
WEAK_DIRECTION = "contrast covariance has a direction weaker than √ε of the largest"


@dataclass(frozen=True, slots=True)
class RecordedCluster:
    """One cluster as recorded: the input of the duplicate check.

    ``plane_values`` (S), ``amplitudes`` (L × S), ``factors`` and ``flagged``, the
    correlated-profile flag of each of the L peaks. ``amplitudes`` and ``factors`` are
    None for an unusable cluster.
    """

    peak_names: tuple[str, ...]
    plane_values: FloatArray
    amplitudes: FloatArray | None
    factors: CovarianceFactors | None
    flagged: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class DuplicatePool:
    """The pooled standardized contrasts of one group of peaks."""

    q: float = 0.0  # Σq
    contrasts: int = 0  # Σk
    peaks: int = 0
    clusters: int = 0

    @property
    def rms(self) -> float:
        """√(Σq / Σk): 1 when the uncertainty of contrasts is calibrated; NaN if empty."""
        return float(np.sqrt(self.q / self.contrasts)) if self.contrasts else float("nan")

    @property
    def standard_error_bound(self) -> float:
        """1/√(2n), the independent-case SE: a lower bound on the RMS's sampling SE."""
        return float(1.0 / np.sqrt(2.0 * self.contrasts)) if self.contrasts else float("inf")

    @property
    def low_sample(self) -> bool:
        """Whether that lower bound exceeds 0.1: fewer than 50 contrasts."""
        return 0 < self.contrasts < MIN_DUPLICATE_CONTRASTS


@dataclass(frozen=True, slots=True)
class DuplicateCheck:
    """The duplicate check of a run: two pools and the peaks left out, with reasons."""

    unflagged: DuplicatePool = field(default_factory=DuplicatePool)
    flagged: DuplicatePool = field(default_factory=DuplicatePool)
    excluded: dict[str, str] = field(default_factory=dict)


def duplicate_groups(plane_values: FloatArray) -> tuple[tuple[int, ...], ...]:
    """The planes whose values are exactly equal, as index groups of two or more.

    Groups are in the order of their first plane, each in plane order.
    """
    groups: dict[float, list[int]] = {}
    for index, value in enumerate(np.asarray(plane_values, dtype=float).tolist()):
        groups.setdefault(value, []).append(index)
    return tuple(tuple(group) for group in groups.values() if len(group) > 1)


def _contrasts(groups: tuple[tuple[int, ...], ...], n_planes: int) -> FloatArray:
    """Orthonormal within-group (Helmert) contrasts, stacked: Σ_g (m_g − 1) × S."""
    rows = []
    for group in groups:
        for j in range(1, len(group)):
            row = np.zeros(n_planes)
            row[list(group[:j])] = 1.0
            row[group[j]] = -float(j)
            rows.append(row / np.sqrt(j * (j + 1.0)))
    return np.array(rows).reshape(-1, n_planes)


def _standardized(contrast: FloatArray, covariance: FloatArray) -> float | str:
    """The standardized norm dᵀV⁻¹d, or why the contrast covariance V cannot whiten d."""
    if not np.all(np.isfinite(covariance)):
        return FACTORS_WITHHELD
    eigenvalues, vectors = np.linalg.eigh(covariance)
    if eigenvalues[0] <= 0.0:
        return NOT_POSITIVE_DEFINITE
    if eigenvalues[0] <= _WEAK * eigenvalues[-1]:
        return WEAK_DIRECTION
    return float(np.sum((vectors.T @ contrast) ** 2 / eigenvalues))


@dataclass(slots=True)
class _Tally:
    q: float = 0.0
    contrasts: int = 0
    peaks: int = 0
    clusters: set[int] = field(default_factory=set)

    def pool(self) -> DuplicatePool:
        return DuplicatePool(self.q, self.contrasts, self.peaks, len(self.clusters))


def duplicate_plane_check(clusters: Sequence[RecordedCluster]) -> DuplicateCheck:
    """Pool the standardized contrasts between duplicate planes over every peak.

    Each cluster's duplicates are the groups of its exactly equal plane values; a
    cluster without them contributes nothing and excludes nothing. Nothing is
    regularized: a peak whose contrasts cannot be whitened is left out with its reason.
    """
    tallies = {False: _Tally(), True: _Tally()}
    excluded: dict[str, str] = {}
    for index, cluster in enumerate(clusters):
        groups = duplicate_groups(cluster.plane_values)
        if not groups:
            continue
        if cluster.amplitudes is None or cluster.factors is None:
            excluded.update(dict.fromkeys(cluster.peak_names, UNUSABLE_CLUSTER))
            continue
        if not cluster.factors.available:
            excluded.update(dict.fromkeys(cluster.peak_names, FACTORS_WITHHELD))
            continue
        contrasts = _contrasts(groups, cluster.amplitudes.shape[1])
        covariances = peak_covariances_between_planes(cluster.factors, cluster.amplitudes)
        for peak, name in enumerate(cluster.peak_names):
            q = _standardized(
                contrasts @ cluster.amplitudes[peak],
                contrasts @ covariances[peak] @ contrasts.T,
            )
            if isinstance(q, str):
                excluded[name] = q
                continue
            tally = tallies[bool(cluster.flagged[peak])]
            tally.q += q
            tally.contrasts += contrasts.shape[0]
            tally.peaks += 1
            tally.clusters.add(index)
    return DuplicateCheck(tallies[False].pool(), tallies[True].pool(), excluded)


def run_duplicate_check(outcome: FinalFitOutcome, context: Mapping[str, Any]) -> DuplicateCheck:
    """The duplicate check of a published run, with its correlated-profile flags.

    The plane values, amplitudes and covariance factors are those recorded: nothing is
    refitted.
    """
    plane_values = np.asarray(context["plane_values"], dtype=float)
    clusters = []
    for cluster in outcome.clusters:
        names = tuple(cluster.peak_names)
        evaluation = cluster.analytical_evaluation
        if not cluster.usable or evaluation is None:
            clusters.append(RecordedCluster(names, plane_values, None, None, (False,) * len(names)))
            continue
        factors, amplitudes = evaluation.covariance_factors, evaluation.amplitudes
        flags = tuple(flag.flagged for flag in profile_correlations(factors, amplitudes))
        clusters.append(RecordedCluster(names, plane_values, amplitudes, factors, flags))
    return duplicate_plane_check(clusters)


__all__ = [
    "BASELINE_HZ",
    "FACTORS_WITHHELD",
    "MIN_DUPLICATE_CONTRASTS",
    "NOT_POSITIVE_DEFINITE",
    "UNUSABLE_CLUSTER",
    "WEAK_DIRECTION",
    "BaselineScatter",
    "DuplicateCheck",
    "DuplicatePool",
    "RecordedCluster",
    "baseline_scatter",
    "duplicate_groups",
    "duplicate_plane_check",
    "robust_sd",
    "run_baseline_scatter",
    "run_duplicate_check",
]
