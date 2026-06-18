"""Pure ROI and candidate-selection helpers for automatic peak picking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure, label, maximum_filter

if TYPE_CHECKING:
    from peakfit.engine.domain.spectrum import Spectra
    from peakfit.shared.typing import FloatArray, IntArray


_FLOAT_EPS = 1e-12
_MIN_PLOT_DIMS = 2


def point_to_ppm(point_indices: tuple[int, ...], spectra: Spectra) -> tuple[float, ...]:
    """Convert integer grid coordinates to ppm coordinates."""
    return tuple(
        float(spectral_param.pts2ppm(float(point_indices[i])))
        for i, spectral_param in enumerate(spectra.spectral_params)
    )


def candidate_ppm_for_plot(
    candidate: tuple[int, float] | None,
    roi_points: np.ndarray,
    spectra: Spectra,
) -> tuple[float, float] | None:
    """Convert a candidate ROI index to (y_ppm, x_ppm) for plotting."""
    if candidate is None:
        return None
    candidate_idx, _score = candidate
    point = tuple(int(v) for v in roi_points[candidate_idx])
    point_ppm = point_to_ppm(point, spectra)
    return float(point_ppm[0]), float(point_ppm[-1])


def roi_plot_limits(
    roi_indices: list[IntArray],
    spectra: Spectra,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Return X/Y ppm limits for visual zoom on the active ROI."""
    if not roi_indices or spectra.n_spectral_dims < _MIN_PLOT_DIMS:
        return None, None

    x_values = np.asarray(spectra.spectral_params[-1].pts2ppm(roi_indices[-1]), dtype=np.float64)
    y_values = np.asarray(spectra.spectral_params[0].pts2ppm(roi_indices[0]), dtype=np.float64)
    if x_values.size == 0 or y_values.size == 0:
        return None, None

    x_limits = (float(np.min(x_values)), float(np.max(x_values)))
    y_limits = (float(np.min(y_values)), float(np.max(y_values)))
    return x_limits, y_limits


def find_global_seed(
    data: FloatArray,
    blocked_mask: np.ndarray | None = None,
) -> tuple[tuple[int, ...] | None, float]:
    """Find the highest-intensity point in a pseudo-ND data cube."""
    if data.size == 0:
        return None, 0.0

    intensity = np.max(np.abs(data), axis=0)
    if intensity.size == 0:
        return None, 0.0
    if blocked_mask is not None:
        if blocked_mask.shape != intensity.shape:
            raise ValueError("blocked_mask shape must match spectral grid shape")
        if np.all(blocked_mask):
            return None, 0.0
        intensity = intensity.copy()
        intensity[blocked_mask] = -np.inf

    max_flat = int(np.argmax(intensity))
    max_value = float(intensity.flat[max_flat])
    max_point = tuple(int(x) for x in np.unravel_index(max_flat, intensity.shape))
    return max_point, max_value


def extract_roi_indices(
    data: FloatArray,
    contour_level: float,
    seed_point: tuple[int, ...],
) -> list[IntArray]:
    """Extract non-wrapping contour-connected ROI indices containing the seed point."""
    mask = np.any(np.abs(data) >= contour_level, axis=0)
    if mask.ndim == 0:
        return [np.asarray([coord], dtype=np.int_) for coord in seed_point]

    structure = generate_binary_structure(mask.ndim, mask.ndim)
    seed_mask = np.zeros_like(mask, dtype=bool)
    seed_mask[seed_point] = True
    selected = mask | binary_dilation(seed_mask, structure=structure)

    segments, _ = label(selected, structure=structure)
    segment_id = int(segments[seed_point])
    if segment_id <= 0:
        return [np.asarray([coord], dtype=np.int_) for coord in seed_point]

    coords = np.where(segments == segment_id)
    return [np.asarray(axis_coords, dtype=np.int_) for axis_coords in coords]


def select_seed_candidate(
    residual: FloatArray,
    roi_points: np.ndarray,
    seed_point: tuple[int, ...],
    threshold: float,
) -> tuple[int, float] | None:
    """Use the ROI seed point as first candidate when it is above threshold."""
    if roi_points.size == 0:
        return None

    matches = np.flatnonzero(np.all(roi_points == np.asarray(seed_point, dtype=np.int_), axis=1))
    if matches.size == 0:
        return None

    idx = int(matches[0])
    score = float(np.max(np.abs(residual[idx])))
    if not np.isfinite(score) or score < threshold:
        return None
    return idx, score


def initial_local_maxima_candidates(
    working_data: FloatArray,
    roi_indices: list[IntArray],
    roi_points: np.ndarray,
    threshold: float | None,
) -> list[tuple[int, float]]:
    """Return ROI local maxima as initial candidate list sorted by intensity."""
    if not roi_indices or roi_points.size == 0:
        return []

    intensity = np.max(np.abs(working_data), axis=0)
    if intensity.size == 0:
        return []

    roi_mask = np.zeros_like(intensity, dtype=bool)
    roi_mask[tuple(roi_indices)] = True
    if not np.any(roi_mask):
        return []

    structure = generate_binary_structure(intensity.ndim, intensity.ndim)
    local_mask = roi_mask & (
        intensity == maximum_filter(intensity, footprint=structure, mode="nearest")
    )
    if threshold is not None:
        local_mask &= intensity >= threshold

    labeled, n_labels = label(local_mask, structure=structure)
    if n_labels <= 0:
        return []

    index_by_point = {tuple(int(v) for v in point): idx for idx, point in enumerate(roi_points)}
    candidates: list[tuple[int, float]] = []
    for label_id in range(1, n_labels + 1):
        points = np.column_stack(np.where(labeled == label_id))
        if points.size == 0:
            continue
        scores = intensity[tuple(points.T)]
        best_idx = int(np.argmax(scores))
        best_point = tuple(int(v) for v in points[best_idx])
        roi_idx = index_by_point.get(best_point)
        if roi_idx is None:
            continue
        best_score = float(scores[best_idx])
        if best_score <= _FLOAT_EPS:
            continue
        candidates.append((roi_idx, best_score))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates


def extract_roi_data(data: FloatArray, roi_indices: list[IntArray]) -> FloatArray:
    """Extract ROI data as (n_points, n_series)."""
    roi_slice = (slice(None), *roi_indices)
    return np.asarray(data[roi_slice].T, dtype=np.float64)


def stack_roi_points(roi_indices: list[IntArray]) -> np.ndarray:
    """Stack ROI coordinate arrays into shape (n_points, n_dims)."""
    return np.column_stack(roi_indices)


def select_manual_candidate(
    residual: FloatArray,
    roi_points: np.ndarray,
    spectra: Spectra,
    target_ppm: tuple[float, float],
    used_points: list[tuple[int, ...]],
    min_separation_pts: int,
    threshold: float | None,
    eligible_mask: np.ndarray | None = None,
) -> tuple[int, float] | None:
    """Pick the closest ROI point to a user-selected ppm target."""
    if roi_points.size == 0:
        return None

    y_ppm, x_ppm = target_ppm
    x_param = spectra.spectral_params[-1]
    y_param = spectra.spectral_params[0]
    x_values = np.asarray(x_param.pts2ppm(roi_points[:, -1]), dtype=np.float64)
    y_values = np.asarray(y_param.pts2ppm(roi_points[:, 0]), dtype=np.float64)
    distances = (x_values - x_ppm) ** 2 + (y_values - y_ppm) ** 2

    point_scores = np.max(np.abs(residual), axis=1)
    for idx in np.argsort(distances):
        if eligible_mask is not None and not bool(eligible_mask[idx]):
            continue
        score = float(point_scores[idx])
        if threshold is not None and score < threshold:
            continue

        candidate = tuple(int(v) for v in roi_points[idx])
        if far_from_existing(candidate, used_points, min_separation_pts):
            return int(idx), score

    return None


def select_next_candidate(
    residual: FloatArray,
    roi_points: np.ndarray,
    used_points: list[tuple[int, ...]],
    min_separation_pts: int,
    threshold: float | None,
    eligible_mask: np.ndarray | None = None,
) -> tuple[int, float] | None:
    """Select the next residual maximum that is sufficiently separated."""
    point_scores = np.max(np.abs(residual), axis=1)
    order = np.argsort(point_scores)[::-1]

    for idx in order:
        if eligible_mask is not None and not bool(eligible_mask[idx]):
            continue
        score = float(point_scores[idx])
        if threshold is not None and score < threshold:
            return None
        if score <= _FLOAT_EPS:
            return None

        candidate = tuple(int(v) for v in roi_points[idx])
        if far_from_existing(candidate, used_points, min_separation_pts):
            return int(idx), score

    return None


def far_from_existing(
    point: tuple[int, ...],
    others: list[tuple[int, ...]],
    min_separation_pts: int,
) -> bool:
    """Check whether a candidate point is sufficiently separated from existing points."""
    if not others or min_separation_pts <= 0:
        return True

    candidate = np.asarray(point, dtype=np.float64)
    threshold_sq = float(min_separation_pts**2)
    for other in others:
        distance_sq = float(np.sum((candidate - np.asarray(other, dtype=np.float64)) ** 2))
        if distance_sq < threshold_sq:
            return False
    return True
