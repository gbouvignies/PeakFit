"""CEST baseline planes as replicates: a check of the published profile σ on real data.

Far from a peak's own offset, a CEST profile is a flat or slowly varying baseline, so
its planes are approximate replicates of one another. Their plane-to-plane scatter is
an empirical measure of the amplitude error, compared here with the published profile
σ (``docs/profile-uncertainty.md`` §5.1, validation T4).

- **Scatter:** the robust SD of second differences across offsets, divided by √6.
  Second differences remove slow trends and common-mode errors; the robust SD limits
  the influence of minor-state dips.
- **Limits:** it measures repeatability, not accuracy. An error common to every
  baseline plane (a lineshape misfit, a bias from linear prediction) is invisible.
  Drift, residual saturation and minor-state dips can only inflate the scatter.

Values are those recorded: nothing is refitted.
"""

from dataclasses import dataclass

import numpy as np

from peakfit.engine.algorithms.covariance_factors import profile_standard_errors

BASELINE_HZ = 400.0  # planes this far from the peak's own offset are baseline
MIN_BASELINE_PLANES = 20
CEST_REFERENCE_HZ = 10000.0  # planes this far off resonance are references
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
    outcome, context, *, reference: list[int] | None = None, window: float = BASELINE_HZ
) -> dict[str, BaselineScatter | str]:
    """Every recorded peak's baseline scatter, from a published run.

    The plane values are the CEST offsets in Hz from the carrier of the first spectral
    axis, whose fitted chemical shift gives each peak's own offset. ``reference`` lists
    reference planes by their index in offset order; by default, the planes at least
    10 kHz off resonance.
    """
    offsets = np.asarray(context["plane_values"], dtype=float)
    if reference is None:
        is_reference = np.abs(offsets) >= CEST_REFERENCE_HZ
    else:
        is_reference = np.zeros(offsets.size, dtype=bool)
        is_reference[np.argsort(offsets, kind="stable")[reference]] = True
    axis = context["spectral_parameters"][1]
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
            peak_offset = shifts[f"{name}.{axis['label']}.cs"] * axis["obs"] - axis["car"]
            results[name] = baseline_scatter(
                offsets,
                evaluation.amplitudes[peak],
                errors[peak],
                peak_offset=peak_offset,
                reference=is_reference,
                window=window,
            )
    return results


__all__ = [
    "BASELINE_HZ",
    "BaselineScatter",
    "baseline_scatter",
    "robust_sd",
    "run_baseline_scatter",
]
