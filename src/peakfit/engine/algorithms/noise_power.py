"""The noise-power check: processing the header does not record (correlated-noise §2.6).

Linear prediction, an earlier window or ``SP -size`` leave a header that passes every
check, so the noise model takes the recorded processing at its word. The spectrum's
own noise can contradict it. Along one axis, the periodogram of the real spectrum's
noise is the time-domain noise power, blurred by any extraction. The recorded
processing predicts it exactly from the axis's correlation ρ:

    E|X_f|² ∝ Σ_{|k|<S} (1 − |k|/S) ρ(|k|) e^{−2πifk/S},  f = 0 … S − 1.

Each unrecorded operation above leaves less noise in the late FID than the recorded
window predicts (``SP -start`` would leave more). The check measures the periodogram
on noise-only lines, then compares the power in the late FID, the ordinates holding
the last ``LATE_FRACTION`` of the predicted power, with the power before it:

    late power = (measured late / measured early) / (predicted late / predicted early).

It is 1 when the processing is as recorded. The zero-frequency ordinate, which holds
any baseline offset, is left out. The check is a warning only: it never changes the
noise model or any error.
"""

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.algorithms.noise_correlation import AxisNoisePower, AxisProcessing, NoiseModel

if TYPE_CHECKING:
    from peakfit.shared.typing import FloatArray

# A line is noise only when every point lies within this many robust σ.
NOISE_ONLY = 4.5
# The late FID: the ordinates holding the last fifth of the predicted noise power.
LATE_FRACTION = 0.2
# A mismatch is a late power outside [LIMIT, 1/LIMIT]. See docs/correlated-noise.md §2.6
# for the evidence that sets it.
LIMIT = 0.9
# Truncated noise explains a mismatch when it matches the measured periodogram to within
# this total-variation distance (a fraction of the noise power).
TRUNCATION_MATCH = 0.05
# Lines are split into this many contiguous groups to measure the sampling spread.
GROUPS = 8
MIN_LINES_PER_GROUP = 4
# A mismatch must exceed this many spreads; "consistent" needs the spread to resolve
# |ln LIMIT| by the same factor. Anything else is inconclusive.
SIGNIFICANCE = 3.0


def check_noise_power(model: NoiseModel, data: FloatArray) -> NoiseModel:
    """``model`` with the noise-power check of each axis, measured on ``data``.

    ``data`` is the spectrum with the pseudo axis first, as the noise model reads it.
    A model that is not modelled has no prediction to check and is returned as is.
    """
    if not model.modelled:
        return model
    values = np.asarray(data)
    sigma = 1.4826 * float(np.median(np.abs(values - np.median(values))))
    checks = tuple(
        _check_axis(axis, _noise_only_lines(values, index + 1, sigma))
        for index, axis in enumerate(model.axes)
    )
    return replace(model, power_check=checks)


def _noise_only_lines(data: np.ndarray, axis: int, sigma: float) -> np.ndarray:
    """Every line along ``axis`` whose points all lie within NOISE_ONLY robust σ."""
    lines = np.moveaxis(data, axis, -1).reshape(-1, data.shape[axis])
    if not sigma > 0:
        return lines[:0]
    return lines[np.abs(lines).max(axis=1) < NOISE_ONLY * sigma]


def _check_axis(axis: AxisProcessing, lines: np.ndarray) -> AxisNoisePower:
    if len(lines) < GROUPS * MIN_LINES_PER_GROUP:
        return AxisNoisePower("inconclusive", len(lines))
    stored = lines.shape[1]
    predicted = _periodogram(axis, stored)
    measured = np.stack([_measured(group) for group in np.array_split(lines, GROUPS)])
    late = _late(predicted)
    total = measured.mean(axis=0)
    ratio = _late_power(total, predicted, late)
    logs = np.log([_late_power(m, predicted, late) for m in measured])
    spread = float(np.std(logs, ddof=1) / np.sqrt(GROUPS))
    limit = -np.log(LIMIT)
    if not (np.isfinite(ratio) and ratio > 0 and np.isfinite(spread)):
        return AxisNoisePower("inconclusive", len(lines), _finite(ratio), _finite(spread))
    deviation = abs(float(np.log(ratio)))
    if deviation > limit and deviation > SIGNIFICANCE * spread:
        estimate = _acquired_estimate(axis, stored, total)
        return AxisNoisePower("mismatch", len(lines), ratio, spread, estimate)
    if deviation <= limit and spread <= limit / SIGNIFICANCE:
        return AxisNoisePower("consistent", len(lines), ratio, spread)
    return AxisNoisePower("inconclusive", len(lines), ratio, spread)


def _measured(lines: np.ndarray) -> np.ndarray:
    """The mean periodogram of ``lines``, ordinates 1 … S/2."""
    power = np.abs(np.fft.rfft(np.asarray(lines, dtype=np.float64), axis=1)) ** 2
    return power.mean(axis=0)[1:]


def _periodogram(
    axis: AxisProcessing, stored: int, noise_power: FloatArray | None = None
) -> np.ndarray:
    """The expected periodogram of S stored points, ordinates 1 … S/2, from ρ.

    Folding the Bartlett-weighted lags −k onto S − k turns the sum into one FFT.
    """
    rho = axis.correlation(noise_power)
    k = np.arange(1, stored)
    folded = np.empty(stored)
    folded[0] = rho[0]
    # Lags beyond the transform size are periodic in it.
    folded[1:] = (1 - k / stored) * rho[k % axis.transform] + (k / stored) * rho[
        (stored - k) % axis.transform
    ]
    return np.real(np.fft.rfft(folded))[1:]


def _late(predicted: np.ndarray) -> np.ndarray:
    """The ordinates that hold the last LATE_FRACTION of the predicted power."""
    share = np.cumsum(predicted) / predicted.sum()
    return share > 1 - LATE_FRACTION


def _late_power(measured: np.ndarray, predicted: np.ndarray, late: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(
            (measured[late].sum() / measured[~late].sum())
            / (predicted[late].sum() / predicted[~late].sum())
        )


def _acquired_estimate(axis: AxisProcessing, stored: int, measured: np.ndarray) -> int | None:
    """How many points carry acquired noise, if that alone explains the mismatch.

    Linear prediction extends N₀ acquired points to the recorded N, and the extension
    carries little of the noise. The estimate is the N₀ whose truncated noise power
    best matches the measured periodogram in total variation; it is kept only when
    that match is within TRUNCATION_MATCH, so that truncation explains the whole
    profile and not only its late power.
    """
    w2 = axis.weights() ** 2
    target = measured / measured.sum()
    candidates = []
    for acquired in range(2, axis.acquired):
        truncated = w2.copy()
        truncated[acquired:] = 0.0
        predicted = _periodogram(axis, stored, truncated)
        distance = 0.5 * float(np.abs(target - predicted / predicted.sum()).sum())
        candidates.append((distance, acquired))
    if not candidates:
        return None
    distance, acquired = min(candidates)
    return acquired if distance < TRUNCATION_MATCH else None


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


__all__ = ["check_noise_power"]
