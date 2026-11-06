"""Whether each member's linear amplitude is individually determined by its cluster.

Per plane, a cluster's amplitudes a solve Φᵀa ≈ d, with Φ the (peak, sample) matrix of
fitted shapes. A single coefficient aᵢ is estimable (individually identifiable) exactly
when the unit vector eᵢ has no component in the null space of Φᵀ. This holds even when
the matrix as a whole is rank deficient, as long as eᵢ lies in its row space.

The classification reuses the factorization that defines the recorded amplitudes
(:func:`factorize_shapes`) and its numerical conventions. It adds no threshold:

- null space: right singular directions below the rank cutoff ε·max(m, n)·s₀, the same
  relative cutoff as ``lstsq(rcond=None)``;
- ill-conditioned directions: s ≤ √ε·s₀, where the joint covariance is withheld;
- a projection of the unit vector eᵢ counts as non-zero above ε·max(m, n), the rank
  cutoff's relative tolerance, applied to a unit vector.

A member with a null-space component is *unidentifiable*. A member that is
mathematically identifiable but whose value or uncertainty is not reliably established
is *not established*: it projects onto an ill-conditioned direction, its shape is
near-collinear with another member's (:data:`COLLINEARITY_LIMIT`, the rule that already
withholds fitted-position adoption), or its analytical σ is unavailable. Full numerical
rank never yields unidentifiable.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from peakfit.engine.algorithms.linear_algebra import factorize_shapes
from peakfit.engine.diagnostics.coincidence import collinear_pairs

_EPS = float(np.finfo(float).eps)


class AmplitudeStatus(Enum):
    """How far the data determine one member's amplitude."""

    IDENTIFIABLE = "identifiable"
    NOT_ESTABLISHED = "not established"
    UNIDENTIFIABLE = "unidentifiable"


class Cause(Enum):
    """Why a member's amplitude is not identifiable or not established."""

    NONE = "none"
    NULL_SPACE = "null space"  # unidentifiable
    ILL_CONDITIONED = "ill-conditioned"
    NEAR_COLLINEAR = "near-collinear"
    UNCERTAINTY_UNAVAILABLE = "uncertainty unavailable"  # the value itself is determined
    INVALID_SHAPE = "invalid shape"


@dataclass(frozen=True, slots=True)
class MemberAmplitude:
    """One member's classification, with the quantities it rests on."""

    status: AmplitudeStatus
    cause: Cause
    null_projection: float  # ‖P_null eᵢ‖; 0 for a full-rank shape matrix
    weak_projection: float  # ‖P eᵢ‖ onto directions with s ≤ √ε·s₀ (null space included)
    rank: int  # numerical rank of the shape matrix
    peaks: int  # members in the cluster
    condition: float  # s_min / s₀ over the retained singular values; NaN when undefined
    partners: tuple[int, ...] = ()  # near-collinear members (row indices)
    dissimilarity: float = float("nan")  # smallest 1 − |cos| to a partner

    @property
    def determined(self) -> bool:
        """The recorded value is determined by the data (only its σ may be missing)."""
        return self.status is AmplitudeStatus.IDENTIFIABLE or (
            self.cause is Cause.UNCERTAINTY_UNAVAILABLE
        )


def amplitude_identifiability(
    shapes: np.ndarray,
    amplitudes: np.ndarray | None = None,
    errors: np.ndarray | None = None,
) -> tuple[MemberAmplitude, ...]:
    """Classify every member of one cluster from its recorded shapes (peak, sample).

    ``amplitudes`` (peak, plane) only report how near-collinear pairs cancel and do
    not change any status; ``errors`` are the recorded analytical standard errors.
    """
    shapes = np.asarray(shapes, dtype=float)
    n_peaks = shapes.shape[0]
    norms = np.linalg.norm(shapes, axis=1) if shapes.size else np.zeros(n_peaks)
    invalid = ~np.all(np.isfinite(shapes), axis=1) | ~np.isfinite(norms) | (norms == 0)
    if invalid.any() or shapes.shape[1] == 0:
        return tuple(
            MemberAmplitude(
                AmplitudeStatus.NOT_ESTABLISHED,
                Cause.INVALID_SHAPE,
                np.nan,
                np.nan,
                0,
                n_peaks,
                np.nan,
            )
            for _ in range(n_peaks)
        )
    projection = factorize_shapes(shapes)
    singular = projection.singular_values
    directions = projection.directions  # (min(m, n), peaks), orthonormal rows
    rank = projection.rank
    tolerance = _EPS * max(shapes.shape)
    largest = float(singular[0])
    weak = singular <= np.sqrt(_EPS) * largest
    weak[rank:] = True
    null = np.sum(directions[rank:] ** 2, axis=0)
    weak_sum = np.sum(directions[weak] ** 2, axis=0)
    if directions.shape[0] < n_peaks:
        # Fewer samples than peaks: the directions not returned are null directions.
        missing = np.clip(1.0 - np.sum(directions**2, axis=0), 0.0, None)
        null, weak_sum = null + missing, weak_sum + missing
    null_projection, weak_projection = np.sqrt(null), np.sqrt(weak_sum)
    retained = singular[:rank]
    condition = float(retained[-1] / largest) if rank and largest > 0 else float("nan")
    partners: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n_peaks)}
    for pair in collinear_pairs(shapes, amplitudes):
        partners[pair.first].append((pair.second, pair.dissimilarity))
        partners[pair.second].append((pair.first, pair.dissimilarity))
    sigma = (
        np.full(n_peaks, True) if errors is None else np.isfinite(np.asarray(errors, dtype=float))
    )
    members = []
    for i in range(n_peaks):
        others = partners[i]
        if null_projection[i] > tolerance:
            status, cause = AmplitudeStatus.UNIDENTIFIABLE, Cause.NULL_SPACE
        elif weak_projection[i] > tolerance:
            status, cause = AmplitudeStatus.NOT_ESTABLISHED, Cause.ILL_CONDITIONED
        elif others:
            status, cause = AmplitudeStatus.NOT_ESTABLISHED, Cause.NEAR_COLLINEAR
        elif not sigma[i]:
            status, cause = AmplitudeStatus.NOT_ESTABLISHED, Cause.UNCERTAINTY_UNAVAILABLE
        else:
            status, cause = AmplitudeStatus.IDENTIFIABLE, Cause.NONE
        members.append(
            MemberAmplitude(
                status,
                cause,
                float(null_projection[i]),
                float(weak_projection[i]),
                rank,
                n_peaks,
                condition,
                tuple(j for j, _d in others),
                min((d for _j, d in others), default=float("nan")),
            )
        )
    return tuple(members)
