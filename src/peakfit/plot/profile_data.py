"""Data transforms for intensity, CEST, and CPMG profile plots."""

from __future__ import annotations

from typing import Any

import numpy as np

_CEST_AUTO_REF_OFFSET_THRESHOLD = 10000.0
_CEST_AUTO_REF_FALLBACK_POINTS = 2


def prepare_intensity_data(points: list[tuple[float, float, float]]) -> Any:
    """Convert raw profile points to intensity plot rows."""
    dtype = [("xlabel", "f8"), ("intensity", "f8"), ("error", "f8")]
    return np.array(points, dtype=dtype)


def prepare_cest_data(
    points: list[tuple[float, float, float]], ref_points: list[int]
) -> Any | None:
    """Normalize CEST intensities against explicit or inferred references."""
    offset = np.array([p[0] for p in points], dtype=float)
    intensity = np.array([p[1] for p in points], dtype=float)
    error = np.array([p[2] for p in points], dtype=float)

    ref_mask = _cest_reference_mask(offset, ref_points)
    if not np.any(ref_mask) or np.all(ref_mask):
        return None

    intensity_ref = float(np.mean(intensity[ref_mask]))
    if not np.isfinite(intensity_ref) or intensity_ref == 0:
        return None

    ref_error = _mean_error(error[ref_mask])
    data_mask = ~ref_mask
    normalized = intensity[data_mask] / intensity_ref
    normalized_error = _ratio_error(
        numerator=intensity[data_mask],
        numerator_error=error[data_mask],
        denominator=intensity_ref,
        denominator_error=ref_error,
    )

    dtype = [("offset", "f8"), ("intensity", "f8"), ("error", "f8")]
    return np.array(
        list(zip(offset[data_mask], normalized, normalized_error, strict=True)), dtype=dtype
    )


def prepare_cpmg_data(points: list[tuple[float, float, float]], time_t2: float) -> Any | None:
    """Convert CPMG intensities to R2eff with deterministic error propagation."""
    ncyc = np.array([p[0] for p in points], dtype=float)
    intensity = np.array([p[1] for p in points], dtype=float)
    error = np.array([p[2] for p in points], dtype=float)

    ref_mask = ncyc == 0
    if not np.any(ref_mask):
        ref_mask[0] = True

    intensity_ref = float(np.mean(intensity[ref_mask]))
    if not np.isfinite(intensity_ref) or intensity_ref == 0:
        return None

    ref_error = _mean_error(error[ref_mask])
    ratio = intensity / intensity_ref
    data_mask = (~ref_mask) & np.isfinite(ratio) & (ratio > 0)
    if not np.any(data_mask):
        return None

    nu_cpmg = np.where(ncyc[data_mask] > 0, ncyc[data_mask] / time_t2, 0.5 / time_t2)
    r2eff = -np.log(ratio[data_mask]) / time_t2
    r2eff_error = (
        _ratio_error(
            numerator=intensity[data_mask],
            numerator_error=error[data_mask],
            denominator=intensity_ref,
            denominator_error=ref_error,
        )
        / time_t2
    )

    dtype = [("nu_cpmg", "f8"), ("r2eff", "f8"), ("error", "f8")]
    return np.array(list(zip(nu_cpmg, r2eff, r2eff_error, strict=True)), dtype=dtype)


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


def _mean_error(errors: np.ndarray) -> float:
    """Standard error of a mean from independent point errors."""
    if len(errors) == 0:
        return 0.0
    return float(np.sqrt(np.sum(np.square(errors))) / len(errors))


def _ratio_error(
    *,
    numerator: np.ndarray,
    numerator_error: np.ndarray,
    denominator: float,
    denominator_error: float,
) -> np.ndarray:
    """Propagate uncertainty for numerator / denominator."""
    relative_num = np.divide(
        numerator_error,
        np.abs(numerator),
        out=np.zeros_like(numerator_error, dtype=float),
        where=numerator != 0,
    )
    relative_den = abs(denominator_error / denominator) if denominator != 0 else 0.0
    ratio = numerator / denominator
    return np.abs(ratio) * np.sqrt(np.square(relative_num) + relative_den**2)


__all__ = [
    "prepare_cest_data",
    "prepare_cpmg_data",
    "prepare_intensity_data",
]
