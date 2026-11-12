"""Derived intensity profiles: reference ratios and CPMG relaxation rates.

One implementation serves the desktop profile display and the CLI's CEST and CPMG
plots, so both publish the same values and uncertainties. The inputs are recorded
amplitudes and their analytical standard errors (lineshapes held fixed), so
amplitudes of different planes are treated as uncorrelated.
"""

from dataclasses import dataclass

import numpy as np

RELIABILITY_LIMIT = 0.1  # largest σ_R/|R| for which first-order propagation is used


@dataclass(frozen=True, slots=True)
class Normalized:
    """Ratios r_k = A_k / R to a reference mean, with propagated σ."""

    values: np.ndarray
    errors: np.ndarray  # first-order σ(r_k); zero for an exact reference point
    reference: float  # R
    reference_error: float  # σ_R
    in_reference: np.ndarray  # columns belonging to S
    exact: bool  # one reference plane: its ratio is exactly 1, with no error bar


@dataclass(frozen=True, slots=True)
class Unavailable:
    """Why no ratio is given: undefined (cannot be computed) or unreliable (withheld)."""

    reason: str
    undefined: bool  # False: computable, but not trustworthy enough to give


def normalize(
    amplitudes: np.ndarray,
    errors: np.ndarray,
    reference: np.ndarray,
    limit: float = RELIABILITY_LIMIT,
) -> Normalized | Unavailable:
    """Normalize one peak's amplitudes to the mean of its reference columns S.

    Uses first-order propagation with the full gradient (GUI specification §9):
    ∂r_k/∂A_k = 1/R − [k∈S]·A_k/(nR²) and ∂r_k/∂A_j = −A_k/(nR²) for j∈S, j≠k.
    Returns why instead when the ratio is undefined, or computable but not reliable
    enough to give: beyond ``limit`` the ratio's distribution is strongly skewed and
    first-order propagation no longer describes it.
    """
    amplitudes = np.asarray(amplitudes, dtype=float)
    errors = np.asarray(errors, dtype=float)
    columns = np.asarray(reference, dtype=int)
    n = columns.size
    if n == 0:
        return Unavailable("no reference plane chosen", undefined=True)
    if not np.all(np.isfinite(errors)):
        return Unavailable("σ unavailable for this peak", undefined=True)
    mean = float(np.mean(amplitudes[columns]))
    if not np.isfinite(mean) or mean == 0:
        return Unavailable("reference amplitude is zero or not finite", undefined=True)
    sigma_r = float(np.sqrt(np.sum(errors[columns] ** 2)) / n)
    ratio = sigma_r / abs(mean)
    if ratio > limit:
        return Unavailable(
            f"reference not distinguishable from its uncertainty: σ_R/|R| = {ratio:.2f}",
            undefined=False,
        )
    values = amplitudes / mean
    in_reference = np.zeros(amplitudes.size, dtype=bool)
    in_reference[columns] = True
    coupling = amplitudes / (n * mean**2)  # |∂r_k/∂A_j| for j in S, j ≠ k
    reference_variance = float(np.sum(errors[columns] ** 2))
    own = 1 / mean - in_reference * coupling  # ∂r_k/∂A_k
    others = reference_variance - in_reference * errors**2  # Σ σ_j² over S without k
    variance = own**2 * errors**2 + coupling**2 * others
    exact = n == 1
    if exact:
        values[columns] = 1.0
        variance[columns] = 0.0
    return Normalized(values, np.sqrt(variance), mean, sigma_r, in_reference, exact)


def r2eff(ratios: np.ndarray, errors: np.ndarray, time_t2: float) -> tuple[np.ndarray, np.ndarray]:
    """CPMG effective relaxation rates R2eff = −ln(r)/T and their first-order σ.

    ``ratios`` are intensities relative to the reference without refocusing, with
    their propagated σ (see ``normalize``). Since d(−ln r)/dr = −1/r,
    σ(R2eff) = σ_r / (|r| T). Ratios must be positive.
    """
    ratios = np.asarray(ratios, dtype=float)
    errors = np.asarray(errors, dtype=float)
    return -np.log(ratios) / time_t2, errors / (np.abs(ratios) * time_t2)
