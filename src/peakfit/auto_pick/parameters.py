"""Parameter setup helpers for automatic peak picking."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from peakfit.engine.domain.config import PeakFitConfig
    from peakfit.engine.domain.params_scalar import Parameters
    from peakfit.engine.domain.peaks import Peak
    from peakfit.engine.domain.spectrum import Spectra


_FLOAT_EPS = 1e-12


def apply_position_windows(params: Parameters, window_ppm: float) -> None:
    """Apply symmetric windows to chemical-shift parameters."""
    for param in params.values():
        if param.name.endswith(".cs"):
            center = float(param.value)
            param.min = center - window_ppm
            param.max = center + window_ppm


def initialize_existing_params_from_previous(
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


def initialize_new_peak_from_median(
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


def build_shared_param_aliases(
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


def sync_shared_params(params: Parameters, shared_aliases: dict[str, str]) -> None:
    """Force aliased target parameters to match their source values."""
    for target_name, source_name in shared_aliases.items():
        if target_name not in params or source_name not in params:
            continue
        params[target_name].value = params[source_name].value
        params[target_name].vary = False


def set_stage_vary_flags(
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


def apply_cs_bounds_from_lw(params: Parameters, spectra: Spectra, config: PeakFitConfig) -> None:
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


def any_cs_close_to_constraint(
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


def has_zero_amplitude_peak(params: Parameters, peaks: list[Peak], atol: float) -> bool:
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


def _normalize_new_peak_names(new_peak_names: str | list[str] | None) -> set[str]:
    """Normalize new-peak names to a non-empty set."""
    if new_peak_names is None:
        return set()
    if isinstance(new_peak_names, str):
        return {new_peak_names} if new_peak_names else set()
    return {name for name in new_peak_names if name}
