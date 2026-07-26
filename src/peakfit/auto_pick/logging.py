"""Reporter output for automatic peak picking."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from peakfit.auto_pick.types import AutoPickCycleReport
    from peakfit.shared.reporter import Reporter


def log_auto_pick_cycle(reporter: Reporter, cycle: AutoPickCycleReport) -> None:
    """Emit detailed auto-pick progress for one ROI cycle."""
    if cycle.stage == "peak_added":
        _log_peak_added(reporter, cycle)
        return

    reporter.info(
        f"[auto-pick] cycle={cycle.iteration} seed_pts={cycle.seed_point} "
        f"seed_ppm={_format_tuple(cycle.seed_ppm)} "
        f"seed={cycle.seed_height:.3e} roi_size={cycle.roi_size} "
        f"threshold={cycle.add_threshold:.3e}"
    )
    if cycle.feedback_message:
        reporter.info(f"[auto-pick] cycle={cycle.iteration} note={cycle.feedback_message}")

    if not cycle.trials:
        reporter.info("[auto-pick]   no candidate above addition threshold")
    for trial in cycle.trials:
        stage_info = (
            f"fit_steps={trial.fit_step_rounds} "
            f"cs_at_constraint={'yes' if trial.cs_at_constraint else 'no'} "
            f"zero_amplitude_peak={'yes' if trial.zero_amplitude_peak else 'no'}"
        )
        prefix = (
            f"[auto-pick]   trial={trial.trial_index} score={trial.candidate_score:.3e} "
            f"pts={trial.candidate_point} ppm={_format_tuple(trial.candidate_ppm)} "
            f"{stage_info}"
        )
        if not trial.fit_success:
            reporter.info(f"{prefix} fit=failed decision=reject reason={trial.reason}")
            continue

        if trial.f_test is None:
            reporter.info(f"{prefix} fit=ok decision=reject reason={trial.reason}")
            continue

        ftest = trial.f_test
        if ftest.f_stat is None or ftest.p_value is None:
            reporter.info(
                f"{prefix} fit=ok rss_old={ftest.old_rss:.3e} rss_new={ftest.new_rss:.3e} "
                f"df=({ftest.df1},{ftest.df2}) decision="
                f"{'accept' if trial.accepted else 'reject'} reason={trial.reason} "
                "f_test=skipped"
            )
            continue

        reporter.info(
            f"{prefix} fit=ok rss_old={ftest.old_rss:.3e} rss_new={ftest.new_rss:.3e} "
            f"df=({ftest.df1},{ftest.df2}) f={ftest.f_stat:.3e} p={ftest.p_value:.3e} "
            f"decision={'accept' if trial.accepted else 'reject'} reason={trial.reason}"
        )

    reporter.info(
        f"[auto-pick] cycle={cycle.iteration} result="
        f"{'accepted' if cycle.accepted else 'rejected'} "
        f"peaks_added={cycle.peaks_added} total_peaks={cycle.total_peaks} "
        f"residual_max={cycle.working_max_after:.3e}"
    )


def _log_peak_added(reporter: Reporter, cycle: AutoPickCycleReport) -> None:
    if cycle.feedback_message:
        reporter.info(f"[auto-pick] cycle={cycle.iteration} note={cycle.feedback_message}")

    trial = cycle.trials[-1] if cycle.trials else None
    if trial is None:
        reporter.info(
            f"[auto-pick] cycle={cycle.iteration} stage=peak_added "
            f"peaks_in_roi={cycle.peaks_added} total_peaks={cycle.total_peaks}"
        )
        return

    prefix = (
        f"[auto-pick] cycle={cycle.iteration} stage=peak_added "
        f"trial={trial.trial_index} peaks_in_roi={cycle.peaks_added} "
        f"total_peaks={cycle.total_peaks}"
    )
    if trial.f_test is None:
        reporter.info(
            f"{prefix} decision={'accept' if trial.accepted else 'reject'} reason={trial.reason}"
        )
        return

    ftest = trial.f_test
    if ftest.f_stat is None or ftest.p_value is None:
        reporter.info(
            f"{prefix} decision={'accept' if trial.accepted else 'reject'} "
            f"reason={trial.reason} f_test=skipped"
        )
        return

    reporter.info(
        f"{prefix} decision={'accept' if trial.accepted else 'reject'} "
        f"f={ftest.f_stat:.3e} p={ftest.p_value:.3e} reason={trial.reason}"
    )


def _format_tuple(values: tuple[float | int, ...], precision: int = 3) -> str:
    """Format tuple values for compact progress logs."""
    return "(" + ", ".join(f"{float(value):.{precision}f}" for value in values) + ")"


__all__ = ["log_auto_pick_cycle"]
