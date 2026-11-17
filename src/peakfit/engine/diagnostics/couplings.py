"""What a fitted doublet's J depends on: the linewidth it competes with, and ρ.

A doublet's components sit at cs ± J/2. When J is below the linewidth of its axis,
they overlap at half height: a smaller J with a wider line, or a larger J with a
narrower one, draw nearly the same multiplet, so J and the linewidth are determined
together. That is a qualification of the J value, not a numerical warning:

- **Below the linewidth:** the fitted J is smaller than the fitted linewidth of the
  same peak on the same axis (both in hertz).
- **Correlations:** ρ of J with that linewidth and, where fitted, with the cluster's
  phase on that axis, from the recorded lineshape-parameter covariance σ²C_θ
  (``CovarianceFactors.parameter_covariance``). They are first-order, like σ(J), and
  absent wherever those factors are withheld: whenever a varied J has no σ, as for
  an unresolved coupling (σ(J)/J above ``RELIABILITY_LIMIT``) or a J at a bound.

The linewidth is the unapodized FWHM, so for a strongly apodized axis J can be
below the apodized width while above the linewidth; the σ(J) rule still withholds
an unresolved J there.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.algorithms.covariance_factors import free_parameters
from peakfit.engine.domain.param_id import name_parts
from peakfit.engine.domain.params_scalar import COUPLING_LABEL, LINEWIDTH_LABEL, PHASE_LABEL

if TYPE_CHECKING:
    from collections.abc import Sequence

    from peakfit.engine.algorithms.covariance_factors import LineshapeParameter
    from peakfit.shared.typing import FloatArray


@dataclass(frozen=True, slots=True)
class CouplingQualification:
    """One fitted J: whether it is below its linewidth, and its first-order ρ.

    A correlation is None where the lineshape covariance is withheld, or where the
    partner is not a free parameter (fixed, or held at a bound).
    """

    coupling: str  # the J parameter, "A.F2.j"
    linewidth: str  # the same peak's linewidth on that axis, "A.F2.lw"
    below_linewidth: bool
    linewidth_correlation: float | None = None  # ρ(J, linewidth)
    phase: str | None = None  # the cluster's phase on that axis, where fitted
    phase_correlation: float | None = None  # ρ(J, phase)


def coupling_qualifications(
    parameters: Sequence[LineshapeParameter],
    order: Sequence[str],
    parameter_covariance: FloatArray | None,
) -> dict[str, CouplingQualification]:
    """Each J coupling among a cluster's nonlinear ``parameters``, by name.

    ``parameters`` are the cluster's recorded nonlinear parameters, ``order`` their
    names in ``lineshape_order`` of the cluster's recorded peaks, and
    ``parameter_covariance`` its σ²C_θ, or None where withheld; its rows are the free
    parameters in that order. A covariance that does not match them is not used.
    """
    by_name = {parameter.name: parameter for parameter in parameters}
    ordered = [name for name in order if name in by_name]
    if sorted(ordered) != sorted(by_name):
        parameter_covariance = None
    free = free_parameters([by_name[name] for name in ordered])
    covariance = None
    if parameter_covariance is not None:
        candidate = np.asarray(parameter_covariance, dtype=np.float64)
        if candidate.shape == (len(free), len(free)):
            covariance = candidate

    def correlation(name: str, partner: str | None) -> float | None:
        if covariance is None or partner is None or not {name, partner} <= set(free):
            return None
        i, k = free.index(name), free.index(partner)
        scale = np.sqrt(covariance[i, i] * covariance[k, k])
        return float(covariance[i, k] / scale) if np.isfinite(scale) and scale > 0 else None

    qualifications = {}
    for name, parameter in by_name.items():
        peak, axis, label = name_parts(name)
        if label != COUPLING_LABEL:
            continue
        linewidth = f"{peak}.{axis}.{LINEWIDTH_LABEL}"
        width = by_name[linewidth].value if linewidth in by_name else np.nan
        phase = next(
            (other for other in by_name if name_parts(other)[1:] == (axis, PHASE_LABEL)), None
        )
        qualifications[name] = CouplingQualification(
            coupling=name,
            linewidth=linewidth,
            below_linewidth=bool(parameter.value < width),
            linewidth_correlation=correlation(name, linewidth),
            phase=phase,
            phase_correlation=correlation(name, phase),
        )
    return qualifications


__all__ = ["CouplingQualification", "coupling_qualifications"]
