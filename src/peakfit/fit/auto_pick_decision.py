"""Statistical decisions for automatic peak picking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import f as f_dist

from peakfit.fit.auto_pick_types import FTestDecision

if TYPE_CHECKING:
    from peakfit.engine.domain.config import PeakFitConfig
    from peakfit.engine.domain.spectrum import Spectra
    from peakfit.fit.auto_pick_state import TrialState
    from peakfit.shared.typing import FloatArray

_FLOAT_EPS = 1e-12


def calculate_dof_scale_from_header(spectra: Spectra) -> float:
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


def addition_threshold(config: PeakFitConfig, noise: float) -> float:
    """Threshold for adding peaks inside an ROI."""
    add_threshold = float(config.auto_peak.add_threshold_sigma) * float(noise)
    contour_level = config.clustering.contour_level
    if contour_level is None:
        contour_level = config.clustering.contour_factor * noise
    return max(add_threshold, float(contour_level))


def accept_trial(
    previous: TrialState | None,
    new: TrialState,
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


__all__ = [
    "accept_trial",
    "addition_threshold",
    "calculate_dof_scale_from_header",
]
