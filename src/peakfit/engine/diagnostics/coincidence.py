"""Near-collinear fitted peak shapes: an ill-conditioned amplitude decomposition.

Two peaks whose fitted shapes over a cluster's fitting samples are nearly
collinear make the per-plane linear amplitude problem nearly rank deficient: the
data determine their combined contribution far better than how it is split, and
the split, with the positions that produced it, can be unstable. A fit can reach
such a state even when the data contain two distinct resonances
(docs/validation/repeat-fit-collapse.md), so near-collinearity says nothing
about how many resonances are present.

The threshold is a conservative operational limit supported by the bundled
evidence, not a physical boundary: there, healthy pairs have 1 − |cos| ≥ 0.136
and near-collinear pairs ≤ 6.4×10⁻⁷. For two equal-width peaks 1 − cos ≈ r², with
r their separation in linewidths, so 10⁻⁴ means shapes equal to about 1 % of a
linewidth.
"""

from dataclasses import dataclass

import numpy as np

COLLINEARITY_LIMIT = 1e-4  # on 1 − |cos| between two fitted shapes
_MATRIX = 2  # (peak, sample) shapes; (peak, plane) amplitudes


@dataclass(frozen=True, slots=True)
class CollinearPair:
    """Two peaks (row indices of the shape matrix) with near-collinear shapes."""

    first: int
    second: int
    dissimilarity: float  # 1 − |cos| over the fitting samples
    cancellation: float  # (‖s₁a₁ᵀ‖ + ‖s₂a₂ᵀ‖) / ‖s₁a₁ᵀ + s₂a₂ᵀ‖; NaN when undefined


def collinear_pairs(
    shapes, amplitudes=None, limit: float = COLLINEARITY_LIMIT
) -> tuple[CollinearPair, ...]:
    """Pairs of fitted shapes with 1 − |cos| ≤ ``limit``.

    ``shapes`` is (peak, sample); ``amplitudes``, if given, (peak, plane) and only
    used to report how much the two contributions cancel. A zero or non-finite
    shape cannot be assessed and is never paired. 1 − |cos| is computed as half the
    squared distance between unit vectors, which stays accurate near collinearity.
    """
    s = np.asarray(shapes, dtype=float)
    if s.ndim != _MATRIX or s.shape[0] < _MATRIX:
        return ()
    norms = np.linalg.norm(s, axis=1)
    valid = np.all(np.isfinite(s), axis=1) & np.isfinite(norms) & (norms > 0)
    units = np.zeros_like(s)
    units[valid] = s[valid] / norms[valid, None]
    a = None if amplitudes is None else np.asarray(amplitudes, dtype=float)
    pairs = []
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            if not (valid[i] and valid[j]):
                continue
            sign = 1.0 if units[i] @ units[j] >= 0 else -1.0
            dissimilarity = 0.5 * float(np.sum((units[i] - sign * units[j]) ** 2))
            if dissimilarity <= limit:
                pairs.append(CollinearPair(i, j, dissimilarity, _cancellation(s[i], s[j], a, i, j)))
    return tuple(pairs)


def _cancellation(si, sj, a, i, j) -> float:
    if a is None or a.ndim != _MATRIX or not np.all(np.isfinite(a[[i, j]])):
        return float("nan")
    ai, aj = a[i], a[j]
    parts = np.linalg.norm(si) * np.linalg.norm(ai) + np.linalg.norm(sj) * np.linalg.norm(aj)
    total = (si @ si) * (ai @ ai) + (sj @ sj) * (aj @ aj) + 2 * (si @ sj) * (ai @ aj)
    return float(parts / np.sqrt(total)) if total > 0 else float("inf")
