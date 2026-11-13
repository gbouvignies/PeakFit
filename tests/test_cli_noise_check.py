"""The CLI states the noise-power check in the run summary."""

from peakfit.ui.console import console
from peakfit.ui.views import display_post_fit_summary

WARNING = "Noise-power check: the noise does not match its recorded processing."


def _summary(noise_warning: str | None) -> str:
    with console.capture() as captured:
        display_post_fit_summary(
            total_time=1.0,
            total_clusters=1,
            successful_fits=1,
            usable_non_converged=0,
            unusable=0,
            chi_sq_stats={"Mean": 1.0},
            clusters_to_review=[],
            output_dir="results",
            noise_warning=noise_warning,
        )
    return " ".join(captured.get().replace("┃", " ").split())  # unwrap the panel


def test_the_run_summary_reports_a_noise_power_mismatch():
    summary = _summary(WARNING)
    assert "Noise check" in summary
    assert "does not match its recorded processing" in summary


def test_the_run_summary_is_silent_when_the_noise_matches():
    assert "Noise check" not in _summary(None)
