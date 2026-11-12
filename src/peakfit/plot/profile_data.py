"""Data transforms for intensity, CEST, and CPMG profile plots."""

from __future__ import annotations

from typing import Any

import numpy as np

from peakfit.engine.scientific import profiles

_CEST_AUTO_REF_OFFSET_THRESHOLD = 10000.0
_CEST_AUTO_REF_FALLBACK_POINTS = 2


def prepare_intensity_data(points: list[tuple[float, float, float]]) -> Any:
    """Convert raw profile points to intensity plot rows."""
    dtype = [("xlabel", "f8"), ("intensity", "f8"), ("error", "f8")]
    return np.array(points, dtype=dtype)


def prepare_cest_data(points: list[tuple[float, float, float]], ref_points: list[int]) -> Any | str:
    """Normalize CEST intensities to their reference points, or say why not.

    Ratios and their σ come from ``engine.scientific.profiles.normalize``, the
    same propagation and reliability limit as the desktop profile display.
    """
    offset = np.array([p[0] for p in points], dtype=float)
    intensity = np.array([p[1] for p in points], dtype=float)
    error = np.array([p[2] for p in points], dtype=float)

    ref_mask = _cest_reference_mask(offset, ref_points)
    if not np.any(ref_mask) or np.all(ref_mask):
        return "no reference points apart from the profile"
    shown = profiles.normalize(intensity, error, np.flatnonzero(ref_mask))
    if isinstance(shown, profiles.Unavailable):
        return shown.reason

    data_mask = ~shown.in_reference
    dtype = [("offset", "f8"), ("intensity", "f8"), ("error", "f8")]
    return np.array(
        list(
            zip(
                offset[data_mask],
                shown.values[data_mask],
                shown.errors[data_mask],
                strict=True,
            )
        ),
        dtype=dtype,
    )


def prepare_cpmg_data(points: list[tuple[float, float, float]], time_t2: float) -> Any | str:
    """Convert CPMG intensities to R2eff with first-order σ, or say why not.

    The reference is the mean of the planes without refocusing (ncyc = 0), or the
    first plane when there is none.
    """
    ncyc = np.array([p[0] for p in points], dtype=float)
    intensity = np.array([p[1] for p in points], dtype=float)
    error = np.array([p[2] for p in points], dtype=float)

    ref_mask = ncyc == 0
    if not np.any(ref_mask):
        ref_mask[0] = True
    shown = profiles.normalize(intensity, error, np.flatnonzero(ref_mask))
    if isinstance(shown, profiles.Unavailable):
        return shown.reason

    data_mask = ~shown.in_reference & np.isfinite(shown.values) & (shown.values > 0)
    if not np.any(data_mask):
        return "no positive intensity ratio to convert"

    nu_cpmg = np.where(ncyc[data_mask] > 0, ncyc[data_mask] / time_t2, 0.5 / time_t2)
    rates, rate_errors = profiles.r2eff(shown.values[data_mask], shown.errors[data_mask], time_t2)
    dtype = [("nu_cpmg", "f8"), ("r2eff", "f8"), ("error", "f8")]
    return np.array(list(zip(nu_cpmg, rates, rate_errors, strict=True)), dtype=dtype)


def _cest_reference_mask(offset: np.ndarray, ref_points: list[int]) -> np.ndarray:
    """Return the points used as CEST references."""
    if ref_points == [-1]:
        ref_mask = np.abs(offset) >= _CEST_AUTO_REF_OFFSET_THRESHOLD
        if np.any(ref_mask):
            return ref_mask

        n_points = len(offset)
        if n_points <= 1:
            return np.zeros_like(offset, dtype=bool)

        n_fallback = min(_CEST_AUTO_REF_FALLBACK_POINTS, n_points - 1)
        distance_to_center = np.abs(offset - np.median(offset))
        fallback_indices = np.argsort(distance_to_center)[-n_fallback:]
        ref_mask = np.zeros_like(offset, dtype=bool)
        ref_mask[fallback_indices] = True
        return ref_mask

    ref_mask = np.zeros_like(offset, dtype=bool)
    for idx in ref_points:
        if 0 <= idx < len(offset):
            ref_mask[idx] = True
    return ref_mask


__all__ = [
    "prepare_cest_data",
    "prepare_cpmg_data",
    "prepare_intensity_data",
]
