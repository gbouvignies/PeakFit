"""Automatic peak picking for workflows without an input peak list.

The implementation follows a residual-driven ROI strategy:
1. Find the highest residual intensity point.
2. Build a contour-connected ROI around that seed.
3. Iteratively add peaks in that ROI and accept additions by F-test.
4. Subtract accepted model intensity from a working residual spectrum.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure, label, maximum_filter
from scipy.stats import f as f_dist

from peakfit.engine.algorithms.common import calculate_shape_heights
from peakfit.engine.algorithms.varpro import ScipyOptimizerError, fit_cluster
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.constraints import apply_constraints
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.lineshapes import LineshapeFactory

if TYPE_CHECKING:
    from peakfit.engine.domain.config import PeakFitConfig
    from peakfit.engine.domain.spectrum import Spectra
    from peakfit.shared.typing import FloatArray, IntArray


_FLOAT_EPS = 1e-12
_HISTORY_REWIND_STEPS = 2
_MIN_PLOT_DIMS = 2


@dataclass(frozen=True)
class AutoPickDiagnostics:
    """Diagnostic counters for automatic peak picking."""

    iterations: int
    accepted_rois: int
    rejected_rois: int
    accepted_peaks: int
    stopped_by_user: bool


@dataclass(frozen=True)
class AutoPickResult:
    """Result container for automatic peak picking."""

    peaks: list[Peak]
    diagnostics: AutoPickDiagnostics


@dataclass(frozen=True)
class FTestDecision:
    """Detailed F-test acceptance decision for one trial."""

    accepted: bool
    reason: str
    old_rss: float
    new_rss: float
    df1: int
    df2: int
    f_stat: float | None
    p_value: float | None


@dataclass(frozen=True)
class AutoPickTrialReport:
    """Per-trial report inside one ROI cycle."""

    trial_index: int
    candidate_point: tuple[int, ...]
    candidate_ppm: tuple[float, ...]
    candidate_score: float
    fit_success: bool
    accepted: bool
    reason: str
    f_test: FTestDecision | None
    protocol_rounds: int
    cs_at_constraint: bool
    zero_amplitude_peak: bool


@dataclass(frozen=True)
class AutoPickCycleReport:
    """Cycle-level report used for terminal logging and step-wise control."""

    iteration: int
    seed_point: tuple[int, ...]
    seed_ppm: tuple[float, ...]
    seed_height: float
    roi_size: int
    add_threshold: float
    accepted: bool
    peaks_added: int
    total_peaks: int
    working_max_after: float
    trials: list[AutoPickTrialReport]
    contour_level: float
    experimental_projection: np.ndarray
    simulated_projection: np.ndarray
    current_peaks: list[Peak]
    roi_peaks: list[Peak]
    roi_x_limits: tuple[float, float] | None
    roi_y_limits: tuple[float, float] | None
    next_candidate_ppm: tuple[float, float] | None
    next_candidate_name: str | None
    feedback_message: str | None = None
    stage: str = "cycle_complete"


@dataclass(frozen=True)
class AutoPickCycleAction:
    """User action for interactive auto-pick stepping."""

    command: Literal[
        "continue",
        "remove_last_peak",
        "release_linewidths",
        "next_cluster",
        "previous_cluster",
        "stop",
    ] = "continue"
    candidate_ppm: tuple[float, float] | None = None
    candidate_ppm_list: list[tuple[float, float]] | None = None
    allow_suggested_fallback: bool = True


AutoPickCycleCallback = Callable[[AutoPickCycleReport], AutoPickCycleAction]


@dataclass(frozen=True)
class _TrialState:
    """Intermediate fit state used for ROI peak-growth decisions."""

    peaks: list[Peak]
    data: FloatArray  # (n_points, n_series)
    model: FloatArray  # (n_points, n_series)
    residual: FloatArray  # (n_points, n_series)
    footprint: np.ndarray  # (n_points,)
    n_params: int
    dof_scale: float
    params: Parameters


@dataclass(frozen=True)
class _RoiFitResult:
    """Final accepted state and trial trace for a ROI."""

    accepted_state: _TrialState | None
    trials: list[AutoPickTrialReport]
    add_threshold: float
    stopped_by_user: bool = False
    previous_cluster_requested: bool = False


@dataclass(frozen=True)
class _TrialFitOutcome:
    """Detailed fit outcome for one trial peak candidate."""

    state: _TrialState
    protocol_rounds: int
    cs_at_constraint: bool
    zero_amplitude_peak: bool


@dataclass(frozen=True)
class _AutoPickSnapshot:
    """Checkpoint of global auto-pick state before processing one ROI."""

    working_data: np.ndarray
    calculated_data: np.ndarray
    processed_mask: np.ndarray
    accepted_peaks: list[Peak]
    accepted_rois: int
    rejected_rois: int
    iterations: int
    next_peak_number: int


def auto_pick_peaks(
    spectra: Spectra,
    shape_names: list[str],
    noise: float,
    contour_level: float,
    config: PeakFitConfig,
    cycle_callback: AutoPickCycleCallback | None = None,
) -> AutoPickResult:
    """Automatically pick peaks from spectra using iterative ROI fitting."""
    if noise <= 0:
        raise ValueError(f"Noise must be positive, got {noise}")

    auto_cfg = config.auto_peak
    factory = LineshapeFactory(spectra, config.fitting)
    working_data = np.asarray(spectra.data, dtype=np.float64).copy()
    calculated_data = np.zeros_like(working_data, dtype=np.float64)
    experimental_projection = np.max(
        np.abs(np.asarray(spectra.data, dtype=np.float64)),
        axis=0,
    ).copy()

    next_peak_number = [1]
    accepted_peaks: list[Peak] = []
    accepted_rois = 0
    rejected_rois = 0
    iterations = 0
    stopped_by_user = False
    processed_mask = np.zeros(working_data.shape[1:], dtype=bool)
    history: list[_AutoPickSnapshot] = []
    iteration = 1

    while iteration <= auto_cfg.max_clusters:
        seed_point, seed_height = _find_global_seed(working_data, blocked_mask=processed_mask)
        if seed_point is None:
            break

        if seed_height < auto_cfg.start_threshold_sigma * noise:
            break

        if cycle_callback is not None:
            history.append(
                _AutoPickSnapshot(
                    working_data=working_data.copy(),
                    calculated_data=calculated_data.copy(),
                    processed_mask=processed_mask.copy(),
                    accepted_peaks=list(accepted_peaks),
                    accepted_rois=accepted_rois,
                    rejected_rois=rejected_rois,
                    iterations=iterations,
                    next_peak_number=next_peak_number[0],
                )
            )
        iterations += 1
        roi_indices = _extract_roi_indices(working_data, contour_level, seed_point)
        processed_mask[tuple(roi_indices)] = True
        roi_x_limits, roi_y_limits = _roi_plot_limits(roi_indices, spectra)

        def _emit_peak_added_update(
            state: _TrialState | None,
            trials: list[AutoPickTrialReport],
            add_threshold: float,
            next_candidate_ppm: tuple[float, float] | None,
            next_candidate_name: str | None,
            feedback_message: str | None,
            *,
            _iteration: int = iteration,
            _seed_point: tuple[int, ...] = seed_point,
            _seed_height: float = seed_height,
            _roi_indices: list[IntArray] = roi_indices,
            _roi_x_limits: tuple[float, float] | None = roi_x_limits,
            _roi_y_limits: tuple[float, float] | None = roi_y_limits,
            _working_data: FloatArray = working_data,
            _processed_mask: np.ndarray = processed_mask,
            _calculated_data: FloatArray = calculated_data,
            _accepted_peaks: list[Peak] = accepted_peaks,
        ) -> AutoPickCycleAction:
            if cycle_callback is None:
                return AutoPickCycleAction()

            _next_seed, working_max_after = _find_global_seed(
                _working_data,
                blocked_mask=_processed_mask,
            )
            simulated_data = _calculated_data.copy()
            if state is not None:
                roi_slice = (slice(None), *_roi_indices)
                simulated_data[roi_slice] = simulated_data[roi_slice] + state.model.T
            roi_peaks = [] if state is None else list(state.peaks)
            current_peaks = [*_accepted_peaks, *roi_peaks]

            report = AutoPickCycleReport(
                iteration=_iteration,
                seed_point=_seed_point,
                seed_ppm=_point_to_ppm(_seed_point, spectra),
                seed_height=_seed_height,
                roi_size=int(_roi_indices[0].size) if _roi_indices else 0,
                add_threshold=add_threshold,
                accepted=state is not None,
                peaks_added=len(roi_peaks),
                total_peaks=len(current_peaks),
                working_max_after=working_max_after,
                trials=list(trials),
                contour_level=contour_level,
                experimental_projection=experimental_projection,
                simulated_projection=np.max(np.abs(simulated_data), axis=0).copy(),
                current_peaks=current_peaks,
                roi_peaks=roi_peaks,
                roi_x_limits=_roi_x_limits,
                roi_y_limits=_roi_y_limits,
                next_candidate_ppm=next_candidate_ppm,
                next_candidate_name=next_candidate_name,
                feedback_message=feedback_message,
                stage="peak_added",
            )
            return cycle_callback(report)

        roi_result = _fit_roi_iteratively(
            spectra=spectra,
            working_data=working_data,
            factory=factory,
            shape_names=shape_names,
            roi_indices=roi_indices,
            noise=noise,
            next_peak_number=next_peak_number,
            seed_point=seed_point,
            config=config,
            peak_added_callback=_emit_peak_added_update if cycle_callback is not None else None,
        )

        if roi_result.previous_cluster_requested:
            if len(history) >= _HISTORY_REWIND_STEPS:
                target = history[-_HISTORY_REWIND_STEPS]
                history = history[:-_HISTORY_REWIND_STEPS]
                working_data = target.working_data.copy()
                calculated_data = target.calculated_data.copy()
                processed_mask = target.processed_mask.copy()
                accepted_peaks = list(target.accepted_peaks)
                accepted_rois = target.accepted_rois
                rejected_rois = target.rejected_rois
                iterations = target.iterations
                next_peak_number[0] = target.next_peak_number
                iteration = max(1, iteration - 1)
            else:
                first = history[-1]
                history = history[:-1]
                working_data = first.working_data.copy()
                calculated_data = first.calculated_data.copy()
                processed_mask = first.processed_mask.copy()
                accepted_peaks = list(first.accepted_peaks)
                accepted_rois = first.accepted_rois
                rejected_rois = first.rejected_rois
                iterations = first.iterations
                next_peak_number[0] = first.next_peak_number
                iteration = 1
            continue

        accepted_state = roi_result.accepted_state
        accepted = accepted_state is not None
        peaks_added = len(accepted_state.peaks) if accepted_state is not None else 0

        if accepted_state is None:
            rejected_rois += 1
            _zero_roi(working_data, roi_indices)
        else:
            accepted_peaks.extend(accepted_state.peaks)
            accepted_rois += 1
            _subtract_roi_model(working_data, roi_indices, accepted_state.model)
            _add_roi_model(calculated_data, roi_indices, accepted_state.model)

        if roi_result.stopped_by_user:
            stopped_by_user = True
            break

        if cycle_callback is not None:
            _next_seed, working_max_after = _find_global_seed(
                working_data,
                blocked_mask=processed_mask,
            )
            report = AutoPickCycleReport(
                iteration=iteration,
                seed_point=seed_point,
                seed_ppm=_point_to_ppm(seed_point, spectra),
                seed_height=seed_height,
                roi_size=int(roi_indices[0].size) if roi_indices else 0,
                add_threshold=roi_result.add_threshold,
                accepted=accepted,
                peaks_added=peaks_added,
                total_peaks=len(accepted_peaks),
                working_max_after=working_max_after,
                trials=roi_result.trials,
                contour_level=contour_level,
                experimental_projection=experimental_projection,
                simulated_projection=np.max(np.abs(calculated_data), axis=0).copy(),
                current_peaks=list(accepted_peaks),
                roi_peaks=[] if accepted_state is None else list(accepted_state.peaks),
                roi_x_limits=roi_x_limits,
                roi_y_limits=roi_y_limits,
                next_candidate_ppm=None,
                next_candidate_name=None,
                feedback_message=None,
                stage="cycle_complete",
            )
            action = cycle_callback(report)
            if action.command == "stop":
                stopped_by_user = True
                break
            if action.command == "previous_cluster":
                if len(history) >= _HISTORY_REWIND_STEPS:
                    target = history[-_HISTORY_REWIND_STEPS]
                    history = history[:-_HISTORY_REWIND_STEPS]
                    working_data = target.working_data.copy()
                    calculated_data = target.calculated_data.copy()
                    processed_mask = target.processed_mask.copy()
                    accepted_peaks = list(target.accepted_peaks)
                    accepted_rois = target.accepted_rois
                    rejected_rois = target.rejected_rois
                    iterations = target.iterations
                    next_peak_number[0] = target.next_peak_number
                    iteration = max(1, iteration - 1)
                else:
                    first = history[-1]
                    history = history[:-1]
                    working_data = first.working_data.copy()
                    calculated_data = first.calculated_data.copy()
                    processed_mask = first.processed_mask.copy()
                    accepted_peaks = list(first.accepted_peaks)
                    accepted_rois = first.accepted_rois
                    rejected_rois = first.rejected_rois
                    iterations = first.iterations
                    next_peak_number[0] = first.next_peak_number
                    iteration = 1
                continue

        iteration += 1
    diagnostics = AutoPickDiagnostics(
        iterations=iterations,
        accepted_rois=accepted_rois,
        rejected_rois=rejected_rois,
        accepted_peaks=len(accepted_peaks),
        stopped_by_user=stopped_by_user,
    )
    return AutoPickResult(peaks=accepted_peaks, diagnostics=diagnostics)


def _point_to_ppm(point_indices: tuple[int, ...], spectra: Spectra) -> tuple[float, ...]:
    """Convert integer grid coordinates to ppm coordinates."""
    return tuple(
        float(spectral_param.pts2ppm(float(point_indices[i])))
        for i, spectral_param in enumerate(spectra.spectral_params)
    )


def _candidate_ppm_for_plot(
    candidate: tuple[int, float] | None,
    roi_points: np.ndarray,
    spectra: Spectra,
) -> tuple[float, float] | None:
    """Convert a candidate ROI index to (y_ppm, x_ppm) for plotting."""
    if candidate is None:
        return None
    candidate_idx, _score = candidate
    point = tuple(int(v) for v in roi_points[candidate_idx])
    point_ppm = _point_to_ppm(point, spectra)
    return float(point_ppm[0]), float(point_ppm[-1])


def _roi_plot_limits(
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


def _find_global_seed(
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


def _extract_roi_indices(
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


def _fit_roi_iteratively(
    spectra: Spectra,
    working_data: FloatArray,
    factory: LineshapeFactory,
    shape_names: list[str],
    roi_indices: list[IntArray],
    noise: float,
    next_peak_number: list[int],
    seed_point: tuple[int, ...],
    config: PeakFitConfig,
    peak_added_callback: Callable[
        [
            _TrialState | None,
            list[AutoPickTrialReport],
            float,
            tuple[float, float] | None,
            str | None,
            str | None,
        ],
        AutoPickCycleAction,
    ]
    | None = None,
) -> _RoiFitResult:
    """Iteratively fit and grow a peak list in a single ROI."""
    auto_cfg = config.auto_peak
    add_threshold = _addition_threshold(config, noise)
    interactive_mode = peak_added_callback is not None
    candidate_threshold = None if interactive_mode else add_threshold
    min_separation_pts = 0 if interactive_mode else auto_cfg.min_peak_separation_pts
    roi_data = _extract_roi_data(working_data, roi_indices)
    roi_points = _stack_roi_points(roi_indices)
    dof_scale = _calculate_dof_scale_from_header(spectra)
    local_maxima_candidates = _initial_local_maxima_candidates(
        working_data=working_data,
        roi_indices=roi_indices,
        roi_points=roi_points,
        threshold=candidate_threshold,
    )

    accepted_state: _TrialState | None = None
    accepted_states: list[_TrialState] = []
    accepted_batch_sizes: list[int] = []
    used_points: list[tuple[int, ...]] = []
    trial_reports: list[AutoPickTrialReport] = []
    stopped_by_user = False
    advance_to_next_cluster = False
    previous_cluster_requested = False

    def _consume_local_candidate(current_state: _TrialState | None) -> tuple[int, float] | None:
        if not local_maxima_candidates:
            return None

        residual = roi_data if current_state is None else current_state.residual
        point_scores = np.max(np.abs(residual), axis=1)
        eligible_mask = (
            None if interactive_mode or current_state is None else current_state.footprint
        )

        while local_maxima_candidates:
            idx, _seed_score = local_maxima_candidates.pop(0)
            if idx < 0 or idx >= roi_points.shape[0]:
                continue
            if eligible_mask is not None and not bool(eligible_mask[idx]):
                continue

            candidate = tuple(int(v) for v in roi_points[idx])
            if not _far_from_existing(candidate, used_points, min_separation_pts):
                continue

            score = float(point_scores[idx])
            if candidate_threshold is not None and score < candidate_threshold:
                continue
            if score <= _FLOAT_EPS:
                continue
            return idx, score
        return None

    def _suggest_candidate(current_state: _TrialState | None) -> tuple[int, float] | None:
        local_candidate = _consume_local_candidate(current_state)
        if local_candidate is not None:
            return local_candidate

        residual = roi_data if current_state is None else current_state.residual
        eligible_mask = (
            None if interactive_mode or current_state is None else current_state.footprint
        )
        return _select_next_candidate(
            residual=residual,
            roi_points=roi_points,
            used_points=used_points,
            min_separation_pts=min_separation_pts,
            threshold=candidate_threshold,
            eligible_mask=eligible_mask,
        )

    def _undo_last_accepted() -> _TrialState | None:
        nonlocal accepted_state
        if not accepted_states:
            return accepted_state

        accepted_states.pop()
        rollback_count = accepted_batch_sizes.pop() if accepted_batch_sizes else 1
        for _ in range(rollback_count):
            _rollback_next_peak_number(next_peak_number)
        if used_points and rollback_count > 0:
            del used_points[-min(rollback_count, len(used_points)) :]

        # Remove the last accepted trial and any trailing rejected trials.
        while trial_reports:
            removed_trial = trial_reports.pop()
            if removed_trial.accepted:
                break

        accepted_state = accepted_states[-1] if accepted_states else None
        return accepted_state

    def _wait_for_user_candidate(
        current_state: _TrialState | None,
        *,
        initial_candidate: tuple[int, float] | None = None,
        initial_feedback: str | None = None,
    ) -> tuple[str, list[tuple[int, float]] | None]:
        nonlocal accepted_state

        if peak_added_callback is None:
            if initial_candidate is None:
                return "continue", None
            return "continue", [initial_candidate]

        suggested_candidate = initial_candidate
        feedback_message: str | None = initial_feedback
        while True:
            if suggested_candidate is None:
                suggested_candidate = _suggest_candidate(current_state)

            action = peak_added_callback(
                current_state,
                trial_reports,
                add_threshold,
                _candidate_ppm_for_plot(suggested_candidate, roi_points, spectra),
                _peek_peak_name(next_peak_number) if suggested_candidate is not None else None,
                feedback_message,
            )
            feedback_message = None

            if action.command == "stop":
                return "stop", None
            if action.command == "next_cluster":
                return "next_cluster", None
            if action.command == "previous_cluster":
                return "previous_cluster", None
            if action.command == "remove_last_peak":
                current_state = _undo_last_accepted()
                suggested_candidate = None
                continue
            if action.command == "release_linewidths":
                if current_state is None:
                    feedback_message = "Release LWs ignored: no accepted peak in this cluster yet."
                    continue
                refined_state, release_feedback = _refit_state_with_released_linewidths(
                    accepted_state=current_state,
                    roi_indices=roi_indices,
                    roi_data=roi_data,
                    noise=noise,
                    config=config,
                    dof_scale=dof_scale,
                )
                feedback_message = release_feedback
                if refined_state is None:
                    continue
                current_state = refined_state
                accepted_state = refined_state
                if accepted_states:
                    accepted_states[-1] = refined_state
                suggested_candidate = None
                continue

            residual = roi_data if current_state is None else current_state.residual
            eligible_mask = (
                None if interactive_mode or current_state is None else current_state.footprint
            )

            selected_candidates: list[tuple[int, float]] = []
            selected_points: list[tuple[int, ...]] = []
            selected_indices: set[int] = set()

            manual_targets = action.candidate_ppm_list or []
            if action.candidate_ppm is not None:
                manual_targets = [*manual_targets, action.candidate_ppm]

            for target in manual_targets:
                manual_candidate = _select_manual_candidate(
                    residual=residual,
                    roi_points=roi_points,
                    spectra=spectra,
                    target_ppm=target,
                    used_points=[*used_points, *selected_points],
                    min_separation_pts=min_separation_pts,
                    threshold=candidate_threshold,
                    eligible_mask=eligible_mask,
                )
                if manual_candidate is None:
                    continue
                candidate_idx, _ = manual_candidate
                if candidate_idx in selected_indices:
                    continue
                selected_candidates.append(manual_candidate)
                selected_indices.add(candidate_idx)
                selected_points.append(tuple(int(v) for v in roi_points[candidate_idx]))

            if (
                not selected_candidates
                and action.allow_suggested_fallback
                and suggested_candidate is not None
            ):
                selected_candidates = [suggested_candidate]

            if not selected_candidates:
                suggested_candidate = None
                continue

            accepted_state = current_state
            return "continue", selected_candidates

    initial_state = accepted_state if interactive_mode else None
    candidate = _suggest_candidate(initial_state)
    if candidate is None and not interactive_mode:
        return _RoiFitResult(
            accepted_state=accepted_state,
            trials=trial_reports,
            add_threshold=add_threshold,
        )

    command, selected = _wait_for_user_candidate(
        initial_state,
        initial_candidate=candidate,
        initial_feedback=None,
    )
    if command == "stop":
        return _RoiFitResult(
            accepted_state=accepted_state,
            trials=trial_reports,
            add_threshold=add_threshold,
            stopped_by_user=True,
        )
    if command == "next_cluster":
        advance_to_next_cluster = True
        return _RoiFitResult(
            accepted_state=accepted_state,
            trials=trial_reports,
            add_threshold=add_threshold,
        )
    if command == "previous_cluster":
        return _RoiFitResult(
            accepted_state=accepted_state,
            trials=trial_reports,
            add_threshold=add_threshold,
            previous_cluster_requested=True,
        )
    candidate_batch = selected
    if candidate_batch is None:
        return _RoiFitResult(
            accepted_state=accepted_state,
            trials=trial_reports,
            add_threshold=add_threshold,
        )

    trial_index = 0
    while True:
        trial_index += 1
        if (
            not interactive_mode
            and auto_cfg.max_peaks_per_roi is not None
            and trial_index > auto_cfg.max_peaks_per_roi
        ):
            break

        candidate_points: list[tuple[int, ...]] = []
        candidate_ppms: list[tuple[float, ...]] = []
        candidate_scores: list[float] = []
        candidate_names: list[str] = []
        candidate_peaks: list[Peak] = []
        for candidate_idx, candidate_score in candidate_batch:
            candidate_point = tuple(int(v) for v in roi_points[candidate_idx])
            candidate_points.append(candidate_point)
            candidate_ppms.append(_point_to_ppm(candidate_point, spectra))
            candidate_scores.append(candidate_score)
            candidate_name = _consume_peak_name(next_peak_number)
            candidate_names.append(candidate_name)
            candidate_peaks.append(
                _create_peak(
                    point_indices=candidate_point,
                    spectra=spectra,
                    factory=factory,
                    shape_names=shape_names,
                    peak_name=candidate_name,
                )
            )

        report_point = candidate_points[0]
        report_ppm = candidate_ppms[0]
        report_score = float(max(candidate_scores))

        previous_peaks = [] if accepted_state is None else accepted_state.peaks
        trial_peaks = [*previous_peaks, *candidate_peaks]
        trial_outcome = _fit_trial_state(
            cluster_id=1,
            peaks=trial_peaks,
            spectra=spectra,
            roi_indices=roi_indices,
            roi_data=roi_data,
            noise=noise,
            config=config,
            previous_params=accepted_state.params if accepted_state is not None else None,
            new_peak_names=candidate_names,
            dof_scale=dof_scale,
        )
        if trial_outcome is None:
            trial_reports.append(
                AutoPickTrialReport(
                    trial_index=trial_index,
                    candidate_point=report_point,
                    candidate_ppm=report_ppm,
                    candidate_score=report_score,
                    fit_success=False,
                    accepted=False,
                    reason="fit_failed",
                    f_test=None,
                    protocol_rounds=0,
                    cs_at_constraint=False,
                    zero_amplitude_peak=False,
                )
            )
            for _ in candidate_names:
                _rollback_next_peak_number(next_peak_number)
            if peak_added_callback is None:
                break
            command, selected = _wait_for_user_candidate(accepted_state)
            if command == "stop":
                stopped_by_user = True
                break
            if command == "next_cluster":
                advance_to_next_cluster = True
                break
            if command == "previous_cluster":
                previous_cluster_requested = True
                break
            if selected is None:
                break
            candidate_batch = selected
            continue

        if trial_outcome.zero_amplitude_peak:
            trial_reports.append(
                AutoPickTrialReport(
                    trial_index=trial_index,
                    candidate_point=report_point,
                    candidate_ppm=report_ppm,
                    candidate_score=report_score,
                    fit_success=True,
                    accepted=False,
                    reason="zero_amplitude_peak",
                    f_test=None,
                    protocol_rounds=trial_outcome.protocol_rounds,
                    cs_at_constraint=trial_outcome.cs_at_constraint,
                    zero_amplitude_peak=True,
                )
            )
            for _ in candidate_names:
                _rollback_next_peak_number(next_peak_number)
            if peak_added_callback is None:
                break
            command, selected = _wait_for_user_candidate(accepted_state)
            if command == "stop":
                stopped_by_user = True
                break
            if command == "next_cluster":
                advance_to_next_cluster = True
                break
            if command == "previous_cluster":
                previous_cluster_requested = True
                break
            if selected is None:
                break
            candidate_batch = selected
            continue

        decision = _accept_trial(
            previous=accepted_state,
            new=trial_outcome.state,
            noise=noise,
            config=config,
        )
        trial_reports.append(
            AutoPickTrialReport(
                trial_index=trial_index,
                candidate_point=report_point,
                candidate_ppm=report_ppm,
                candidate_score=report_score,
                fit_success=True,
                accepted=decision.accepted,
                reason=decision.reason,
                f_test=decision,
                protocol_rounds=trial_outcome.protocol_rounds,
                cs_at_constraint=trial_outcome.cs_at_constraint,
                zero_amplitude_peak=False,
            )
        )

        if not decision.accepted:
            for _ in candidate_names:
                _rollback_next_peak_number(next_peak_number)
            if peak_added_callback is None:
                break
            command, selected = _wait_for_user_candidate(accepted_state)
            if command == "stop":
                stopped_by_user = True
                break
            if command == "next_cluster":
                advance_to_next_cluster = True
                break
            if command == "previous_cluster":
                previous_cluster_requested = True
                break
            if selected is None:
                break
            candidate_batch = selected
            continue

        accepted_state = trial_outcome.state
        accepted_states.append(accepted_state)
        accepted_batch_sizes.append(len(candidate_points))
        used_points.extend(candidate_points)

        if peak_added_callback is not None:
            command, selected = _wait_for_user_candidate(accepted_state)
            if command == "stop":
                stopped_by_user = True
                break
            if command == "next_cluster":
                advance_to_next_cluster = True
                break
            if command == "previous_cluster":
                previous_cluster_requested = True
                break
            candidate_batch = selected
        else:
            next_candidate = _suggest_candidate(accepted_state)
            candidate_batch = [next_candidate] if next_candidate is not None else None
        if candidate_batch is None:
            break

        if previous_cluster_requested:
            break

    if (
        not stopped_by_user
        and not advance_to_next_cluster
        and not previous_cluster_requested
        and accepted_state is not None
        and len(accepted_state.peaks) > 1
    ):
        final_state = _fit_final_untied_state(
            accepted_state=accepted_state,
            roi_indices=roi_indices,
            roi_data=roi_data,
            spectra=spectra,
            noise=noise,
            config=config,
            dof_scale=dof_scale,
        )
        if final_state is not None:
            accepted_state = final_state

    return _RoiFitResult(
        accepted_state=accepted_state,
        trials=trial_reports,
        add_threshold=add_threshold,
        stopped_by_user=stopped_by_user,
        previous_cluster_requested=previous_cluster_requested,
    )


def _addition_threshold(config: PeakFitConfig, noise: float) -> float:
    """Threshold for adding peaks inside an ROI.

    Uses a residual-noise cutoff, but never below the contour level that
    defined the ROI.
    """
    add_threshold = float(config.auto_peak.add_threshold_sigma) * float(noise)
    contour_level = config.clustering.contour_level
    if contour_level is None:
        contour_level = config.clustering.contour_factor * noise
    return max(add_threshold, float(contour_level))


def _refit_state_with_released_linewidths(
    accepted_state: _TrialState,
    roi_indices: list[IntArray],
    roi_data: FloatArray,
    noise: float,
    config: PeakFitConfig,
    dof_scale: float,
    linewidth_fraction: float = 0.2,
) -> tuple[_TrialState | None, str]:
    """Refit ROI with per-peak linewidths released within +/- fraction bounds."""
    cluster = Cluster(
        cluster_id=1,
        peaks=accepted_state.peaks,
        grid_indices=roi_indices,
        data=roi_data,
    )
    params = accepted_state.params.copy(deep=True)
    params = apply_constraints(params, config.parameters)

    # Keep scalar couplings tied; release linewidths individually.
    shared_aliases = _build_shared_param_aliases(params, shared_labels={"j"})
    _sync_shared_params(params, shared_aliases)

    has_released_lw = False
    for name, param in params.items():
        if param.computed:
            continue
        if name in shared_aliases:
            param.vary = False
            continue

        param_id = param.param_id
        is_peak_lw = param_id is not None and bool(param_id.peak_name) and param_id.label == "lw"
        if not is_peak_lw:
            param.vary = False
            continue

        value = max(float(param.value), _FLOAT_EPS)
        delta = max(abs(value) * linewidth_fraction, _FLOAT_EPS)
        min_bound = max(_FLOAT_EPS, value - delta, float(param.min))
        max_bound = min(value + delta, float(param.max))
        if max_bound <= min_bound:
            max_bound = min_bound + _FLOAT_EPS

        param.min = min_bound
        param.max = max_bound
        param.value = float(np.clip(value, min_bound, max_bound))
        param.vary = True
        has_released_lw = True

    if not has_released_lw:
        return accepted_state, "Release LWs: no linewidth parameters found."

    fitted_params = _fit_with_varpro(
        params,
        cluster,
        noise,
        config,
        shared_aliases=shared_aliases,
    )
    if fitted_params is None:
        return None, "Release LWs failed: linewidth-only refit did not converge."
    _sync_shared_params(fitted_params, shared_aliases)
    refined_state = _build_trial_state(
        params=fitted_params,
        cluster=cluster,
        dof_scale=dof_scale,
    )
    if refined_state is None:
        return None, "Release LWs failed: could not rebuild fitted state."

    changed, total, max_delta = _linewidth_change_stats(
        before=accepted_state.params,
        after=refined_state.params,
    )
    return (
        refined_state,
        f"Release LWs applied: changed {changed}/{total} linewidths (max Δ={max_delta:.3g} Hz).",
    )


def _peek_peak_name(next_peak_number: list[int]) -> str:
    """Return the name that will be assigned to the next trial peak."""
    return f"ap{next_peak_number[0]}"


def _consume_peak_name(next_peak_number: list[int]) -> str:
    """Allocate and advance the global peak name counter."""
    name = _peek_peak_name(next_peak_number)
    next_peak_number[0] += 1
    return name


def _rollback_next_peak_number(next_peak_number: list[int]) -> None:
    """Rollback peak numbering by one (used when undoing last accepted peak)."""
    if next_peak_number[0] > 1:
        next_peak_number[0] -= 1


def _linewidth_change_stats(
    before: Parameters,
    after: Parameters,
    atol_hz: float = 1e-6,
) -> tuple[int, int, float]:
    """Return (changed_count, total_count, max_abs_delta_hz) for linewidth parameters."""
    changed = 0
    total = 0
    max_delta = 0.0

    for name, after_param in after.items():
        if not name.endswith(".lw") or name not in before:
            continue
        total += 1
        delta = abs(float(after_param.value) - float(before[name].value))
        if delta > atol_hz:
            changed += 1
        max_delta = max(max_delta, delta)

    return changed, total, max_delta


def _select_seed_candidate(
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


def _initial_local_maxima_candidates(
    working_data: FloatArray,
    roi_indices: list[IntArray],
    roi_points: np.ndarray,
    threshold: float | None,
) -> list[tuple[int, float]]:
    """Return ROI local maxima as initial candidate list (sorted by intensity)."""
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


def _extract_roi_data(data: FloatArray, roi_indices: list[IntArray]) -> FloatArray:
    """Extract ROI data as (n_points, n_series)."""
    roi_slice = (slice(None), *roi_indices)
    return np.asarray(data[roi_slice].T, dtype=np.float64)


def _stack_roi_points(roi_indices: list[IntArray]) -> np.ndarray:
    """Stack ROI coordinate arrays into shape (n_points, n_dims)."""
    return np.column_stack(roi_indices)


def _create_peak(
    point_indices: tuple[int, ...],
    spectra: Spectra,
    factory: LineshapeFactory,
    shape_names: list[str],
    peak_name: str,
) -> Peak:
    """Create a Peak object at a grid point."""
    positions = _point_to_ppm(point_indices, spectra)
    shapes = factory.create_shapes(peak_name, positions, shape_names)
    return Peak(name=peak_name, positions=np.array(positions, dtype=np.float64), shapes=shapes)


def _fit_trial_state(
    cluster_id: int,
    peaks: list[Peak],
    spectra: Spectra,
    roi_indices: list[IntArray],
    roi_data: FloatArray,
    noise: float,
    config: PeakFitConfig,
    previous_params: Parameters | None = None,
    new_peak_names: str | list[str] | None = None,
    dof_scale: float = 1.0,
) -> _TrialFitOutcome | None:
    """Fit a trial peak set using the staged SI protocol."""
    cluster = Cluster(cluster_id=cluster_id, peaks=peaks, grid_indices=roi_indices, data=roi_data)
    params = Parameters.from_peaks(peaks, fixed=True)
    params = apply_constraints(params, config.parameters)
    _initialize_existing_params_from_previous(
        params=params,
        previous_params=previous_params,
        new_peak_names=new_peak_names,
    )
    _apply_position_windows(params, window_ppm=config.auto_peak.position_window_ppm)
    _initialize_new_peak_from_median(
        params=params,
        previous_params=previous_params,
        new_peak_names=new_peak_names,
    )
    shared_aliases = _build_shared_param_aliases(params)
    _sync_shared_params(params, shared_aliases)
    allowed_vary_all = {name for name, param in params.items() if param.vary and not param.computed}
    allowed_vary_tied = allowed_vary_all - set(shared_aliases)

    # Stage 1 (SI): fit only M0 with fixed nonlinear parameters.
    try:
        calculate_shape_heights(params, cluster)
    except ValueError:
        return None

    protocol_rounds = 0
    cs_at_constraint = False
    max_rounds = 1 + config.auto_peak.max_constraint_refits

    for round_idx in range(1, max_rounds + 1):
        protocol_rounds = round_idx

        # Stage 2 (SI): release non-CS parameters and refit.
        _set_stage_vary_flags(
            params,
            allowed_vary=allowed_vary_tied,
            release_cs=False,
            force_fix_positions=config.fitting.fix_positions,
        )
        fitted_params = _fit_with_varpro(
            params,
            cluster,
            noise,
            config,
            shared_aliases=shared_aliases,
        )
        if fitted_params is None:
            return None
        params = fitted_params
        _sync_shared_params(params, shared_aliases)

        # Stage 3 (SI): release CS with +/- factor * linewidth bounds and refit.
        _apply_cs_bounds_from_lw(params, spectra, config)
        _set_stage_vary_flags(
            params,
            allowed_vary=allowed_vary_tied,
            release_cs=True,
            force_fix_positions=config.fitting.fix_positions,
        )
        fitted_params = _fit_with_varpro(
            params,
            cluster,
            noise,
            config,
            shared_aliases=shared_aliases,
        )
        if fitted_params is None:
            return None
        params = fitted_params
        _sync_shared_params(params, shared_aliases)

        cs_at_constraint = _any_cs_close_to_constraint(params, spectra, config)
        if not cs_at_constraint or config.fitting.fix_positions:
            break

    zero_amplitude_peak = _has_zero_amplitude_peak(
        params, peaks, atol=config.auto_peak.amplitude_zero_tolerance
    )

    state = _build_trial_state(
        params=params,
        cluster=cluster,
        dof_scale=dof_scale,
    )
    if state is None:
        return None

    return _TrialFitOutcome(
        state=state,
        protocol_rounds=protocol_rounds,
        cs_at_constraint=cs_at_constraint,
        zero_amplitude_peak=zero_amplitude_peak,
    )


def _apply_position_windows(params: Parameters, window_ppm: float) -> None:
    """Apply symmetric windows to chemical-shift parameters."""
    for param in params.values():
        if param.name.endswith(".cs"):
            center = float(param.value)
            param.min = center - window_ppm
            param.max = center + window_ppm


def _initialize_existing_params_from_previous(
    params: Parameters,
    previous_params: Parameters | None,
    new_peak_names: str | list[str] | None = None,
    *,
    new_peak_name: str | None = None,
) -> None:
    """Warm-start existing parameters from the last accepted trial."""
    if previous_params is None:
        return

    new_peak_name_set = _normalize_new_peak_names(new_peak_names)
    if new_peak_name:
        new_peak_name_set.add(new_peak_name)
    for name, current in params.items():
        if name not in previous_params:
            continue

        param_id = current.param_id
        if param_id is not None and param_id.peak_name in new_peak_name_set:
            continue

        previous_value = float(previous_params[name].value)
        if not np.isfinite(previous_value):
            continue
        current.value = float(np.clip(previous_value, current.min, current.max))


def _initialize_new_peak_from_median(
    params: Parameters,
    previous_params: Parameters | None,
    new_peak_names: str | list[str] | None = None,
    *,
    new_peak_name: str | None = None,
) -> None:
    """Initialize new-peak starts from median values of previously fitted peaks."""
    new_peak_name_set = _normalize_new_peak_names(new_peak_names)
    if new_peak_name:
        new_peak_name_set.add(new_peak_name)
    if previous_params is None or not new_peak_name_set:
        return

    grouped_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for previous in previous_params.values():
        param_id = previous.param_id
        if param_id is None or not param_id.peak_name:
            continue
        if param_id.label in {"cs", "I"}:
            continue
        key = (param_id.axis or "", param_id.label)
        grouped_values[key].append(float(previous.value))

    if not grouped_values:
        return

    for current in params.values():
        param_id = current.param_id
        if param_id is None or param_id.peak_name not in new_peak_name_set:
            continue
        if param_id.label in {"cs", "I"}:
            continue

        key = (param_id.axis or "", param_id.label)
        values = grouped_values.get(key)
        if not values:
            continue

        median_value = float(np.median(np.asarray(values, dtype=np.float64)))
        current.value = float(np.clip(median_value, current.min, current.max))


def _normalize_new_peak_names(new_peak_names: str | list[str] | None) -> set[str]:
    """Normalize new-peak names to a non-empty set."""
    if new_peak_names is None:
        return set()
    if isinstance(new_peak_names, str):
        return {new_peak_names} if new_peak_names else set()
    return {name for name in new_peak_names if name}


def _build_shared_param_aliases(
    params: Parameters,
    shared_labels: set[str] | None = None,
) -> dict[str, str]:
    """Build target->source alias map for in-cluster shared shape parameters."""
    anchor_for_key: dict[tuple[str, str], str] = {}
    aliases: dict[str, str] = {}
    if shared_labels is None:
        shared_labels = {"lw", "j"}

    for name, param in params.items():
        param_id = param.param_id
        if param_id is None or not param_id.peak_name:
            continue
        if param_id.label not in shared_labels:
            continue

        key = (param_id.axis or "", param_id.label)
        anchor = anchor_for_key.get(key)
        if anchor is None:
            anchor_for_key[key] = name
            continue
        aliases[name] = anchor

    return aliases


def _sync_shared_params(params: Parameters, shared_aliases: dict[str, str]) -> None:
    """Force aliased target parameters to match their source values."""
    for target_name, source_name in shared_aliases.items():
        if target_name not in params or source_name not in params:
            continue
        params[target_name].value = params[source_name].value
        params[target_name].vary = False


def _fit_with_varpro(
    params: Parameters,
    cluster: Cluster,
    noise: float,
    config: PeakFitConfig,
    shared_aliases: dict[str, str] | None = None,
) -> Parameters | None:
    """Run one VARPRO fit stage and return updated parameters on success."""
    try:
        fit_result = fit_cluster(
            params=params,
            cluster=cluster,
            noise=noise,
            shared_aliases=shared_aliases,
            max_nfev=config.auto_peak.max_nfev_per_fit,
            ftol=config.fitting.tolerance,
            xtol=config.fitting.tolerance,
        )
    except (ScipyOptimizerError, ValueError):
        return None
    return fit_result.params if fit_result.success else None


def _build_trial_state(
    params: Parameters,
    cluster: Cluster,
    dof_scale: float,
) -> _TrialState | None:
    """Build trial state arrays from fitted parameters."""
    try:
        shapes, amplitudes = calculate_shape_heights(params, cluster)
    except ValueError:
        return None

    model = shapes.T @ amplitudes
    residual = cluster.corrected_data - model
    footprint = np.ones(model.shape[0], dtype=bool)
    _update_peak_positions(cluster.peaks, params)
    return _TrialState(
        peaks=cluster.peaks,
        data=cluster.corrected_data,
        model=np.asarray(model, dtype=np.float64),
        residual=np.asarray(residual, dtype=np.float64),
        footprint=footprint,
        n_params=params.get_n_fitted_params(),
        dof_scale=float(max(dof_scale, _FLOAT_EPS)),
        params=params.copy(deep=True),
    )


def _update_peak_positions(peaks: list[Peak], params: Parameters) -> None:
    """Update peak objects from fitted CS values for reporting and downstream use."""
    for peak in peaks:
        peak.update_positions(params)


def _fit_final_untied_state(
    accepted_state: _TrialState,
    roi_indices: list[IntArray],
    roi_data: FloatArray,
    spectra: Spectra,
    noise: float,
    config: PeakFitConfig,
    dof_scale: float,
) -> _TrialState | None:
    """Final ROI refinement with untied lw/j (outside F-test growth loop)."""
    cluster = Cluster(
        cluster_id=1,
        peaks=accepted_state.peaks,
        grid_indices=roi_indices,
        data=roi_data,
    )
    params = accepted_state.params.copy(deep=True)
    params = apply_constraints(params, config.parameters)
    _apply_cs_bounds_from_lw(params, spectra, config)

    allowed_vary_all = {name for name, param in params.items() if not param.computed}
    _set_stage_vary_flags(
        params,
        allowed_vary=allowed_vary_all,
        release_cs=True,
        force_fix_positions=config.fitting.fix_positions,
    )
    final_params = _fit_with_varpro(
        params,
        cluster,
        noise,
        config,
        shared_aliases=None,
    )
    if final_params is None:
        return None

    return _build_trial_state(
        params=final_params,
        cluster=cluster,
        dof_scale=dof_scale,
    )


def _calculate_dof_scale_from_header(spectra: Spectra) -> float:
    """Estimate effective independent-point fraction from header zero-filling."""
    scale = 1.0
    for spectral_param in spectra.spectral_params:
        if not spectral_param.ft:
            continue
        if spectral_param.size <= 0:
            continue
        if spectral_param.td_size is None:
            continue

        ratio = float(spectral_param.td_size) / float(spectral_param.size)
        if np.isfinite(ratio) and ratio > 0.0:
            scale *= min(1.0, ratio)
    return float(max(scale, _FLOAT_EPS))


def _set_stage_vary_flags(
    params: Parameters,
    *,
    allowed_vary: set[str],
    release_cs: bool,
    force_fix_positions: bool,
) -> None:
    """Set vary flags for staged fitting passes."""
    for name, param in params.items():
        if param.computed:
            continue
        if name not in allowed_vary:
            param.vary = False
            continue
        if param.name.endswith(".cs"):
            param.vary = release_cs and not force_fix_positions
        else:
            param.vary = True


def _apply_cs_bounds_from_lw(params: Parameters, spectra: Spectra, config: PeakFitConfig) -> None:
    """Constrain CS to +/- factor * linewidth converted to ppm."""
    obs_by_axis = {
        spectral_param.label: float(spectral_param.obs)
        for spectral_param in spectra.spectral_params
        if spectral_param.label
    }
    factor = float(config.auto_peak.position_constraint_factor)
    fallback = float(config.auto_peak.position_window_ppm)

    for name, param in params.items():
        if not name.endswith(".cs"):
            continue

        peak_name, axis, _ = name.rsplit(".", 2)
        lw_name = f"{peak_name}.{axis}.lw"

        if lw_name in params and axis in obs_by_axis and obs_by_axis[axis] > _FLOAT_EPS:
            lw_hz = max(float(params[lw_name].value), _FLOAT_EPS)
            ppm_halfwidth = factor * lw_hz / obs_by_axis[axis]
        else:
            ppm_halfwidth = fallback

        center = float(param.value)
        param.min = center - ppm_halfwidth
        param.max = center + ppm_halfwidth


def _any_cs_close_to_constraint(
    params: Parameters,
    spectra: Spectra,
    config: PeakFitConfig,
) -> bool:
    """Check if any CS parameter is near bounds using SI ppm tolerances."""
    proton_tol = float(config.auto_peak.proton_constraint_margin_ppm)
    hetero_tol = float(config.auto_peak.heteronuclear_constraint_margin_ppm)
    tol_by_axis = {}
    for spectral_param in spectra.spectral_params:
        nucleus = (spectral_param.nucleus or "").upper()
        tol_by_axis[spectral_param.label] = proton_tol if "1H" in nucleus else hetero_tol

    for name, param in params.items():
        if not name.endswith(".cs"):
            continue
        if not np.isfinite(param.min) or not np.isfinite(param.max):
            continue
        param_id = param.param_id
        axis = param_id.axis if param_id is not None else name.rsplit(".", 2)[1]
        margin = tol_by_axis.get(axis, hetero_tol)
        if (param.value - param.min) <= margin or (param.max - param.value) <= margin:
            return True
    return False


def _has_zero_amplitude_peak(params: Parameters, peaks: list[Peak], atol: float) -> bool:
    """Return True if any peak has near-zero amplitudes in all spectra."""
    amplitudes_by_peak: dict[str, list[float]] = defaultdict(list)

    for param in params.values():
        param_id = param.param_id
        if param_id is None or param_id.label != "I":
            continue
        amplitudes_by_peak[param_id.peak_name].append(abs(float(param.value)))

    for peak in peaks:
        amplitudes = amplitudes_by_peak.get(peak.name)
        if amplitudes and all(value <= atol for value in amplitudes):
            return True
    return False


def _accept_trial(
    previous: _TrialState | None,
    new: _TrialState,
    noise: float,
    config: PeakFitConfig,
) -> FTestDecision:
    """Decide whether to accept a new trial using an F-test."""
    if previous is None:
        union = new.footprint
        old_rss = _rss(new.data, union, noise)
        old_params = 0
    else:
        union = previous.footprint | new.footprint
        old_rss = _rss(previous.residual, union, noise)
        old_params = previous.n_params

    new_rss = _rss(new.residual, union, noise)
    if new_rss >= old_rss - _FLOAT_EPS:
        return FTestDecision(
            accepted=False,
            reason="rss_not_improved",
            old_rss=old_rss,
            new_rss=new_rss,
            df1=0,
            df2=0,
            f_stat=None,
            p_value=None,
        )

    df1 = new.n_params - old_params
    n_points_raw = int(np.sum(union)) * new.residual.shape[1]
    n_points = round(n_points_raw * new.dof_scale)
    n_points = max(n_points, 1)
    df2 = n_points - new.n_params
    if df1 <= 0 or df2 <= 0:
        return FTestDecision(
            accepted=False,
            reason="invalid_dof",
            old_rss=old_rss,
            new_rss=new_rss,
            df1=df1,
            df2=df2,
            f_stat=None,
            p_value=None,
        )

    f_stat = ((old_rss - new_rss) / df1) / (new_rss / df2)
    if not np.isfinite(f_stat) or f_stat <= 0:
        return FTestDecision(
            accepted=False,
            reason="invalid_f_stat",
            old_rss=old_rss,
            new_rss=new_rss,
            df1=df1,
            df2=df2,
            f_stat=None,
            p_value=None,
        )

    p_value = float(f_dist.sf(f_stat, df1, df2))
    accepted = p_value < config.auto_peak.f_test_pvalue
    return FTestDecision(
        accepted=accepted,
        reason="accepted" if accepted else "p_value_above_threshold",
        old_rss=old_rss,
        new_rss=new_rss,
        df1=df1,
        df2=df2,
        f_stat=float(f_stat),
        p_value=p_value,
    )


def _rss(residual: FloatArray, mask: np.ndarray, noise: float) -> float:
    """Compute residual sum of squares on masked points."""
    if not np.any(mask):
        return float("inf")
    masked = residual[mask, :]
    return float(np.sum((masked / noise) ** 2))


def _select_manual_candidate(
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
        if _far_from_existing(candidate, used_points, min_separation_pts):
            return int(idx), score

    return None


def _select_next_candidate(
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
        if _far_from_existing(candidate, used_points, min_separation_pts):
            return int(idx), score

    return None


def _far_from_existing(
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


def _subtract_roi_model(
    data: FloatArray,
    roi_indices: list[IntArray],
    model: FloatArray,
) -> None:
    """Subtract a fitted ROI model from the working residual spectrum."""
    roi_slice = (slice(None), *roi_indices)
    data[roi_slice] = data[roi_slice] - model.T


def _add_roi_model(
    data: FloatArray,
    roi_indices: list[IntArray],
    model: FloatArray,
) -> None:
    """Add a fitted ROI model to an accumulated simulated spectrum."""
    roi_slice = (slice(None), *roi_indices)
    data[roi_slice] = data[roi_slice] + model.T


def _zero_roi(data: FloatArray, roi_indices: list[IntArray]) -> None:
    """Zero-out an ROI in the working residual spectrum."""
    roi_slice = (slice(None), *roi_indices)
    data[roi_slice] = 0.0
