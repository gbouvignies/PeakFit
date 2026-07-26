"""UI Views for PeakFit CLI.

Implements the "Stream + Sticky Footer" design philosophy:
- Stream (History): Scrolling log of completed events
- Sticky Footer: Fixed line at bottom with global progress and ETA

Design Principles:
- Success is boring - failures and outliers must visually jump out
- Color is information, never decoration
- Preserve scrollback at all times
- Professional scientific software aesthetic
"""

import datetime
import time
from typing import TYPE_CHECKING, Any

from rich import box
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from peakfit.ui.console import console, display_path, icon

if TYPE_CHECKING:
    from collections.abc import Callable


_DEFAULT_MAX_PEAKS_DISPLAY = 3
_MAX_WARNINGS_SHOWN = 2
_HIGH_REDUCED_CHI2_WARNING_THRESHOLD = 5.0
_SECONDS_PER_MINUTE = 60.0
_SECONDS_PER_HOUR = 3600.0
_SECONDS_PER_DAY = 86400.0

# =============================================================================
# Streaming Log Display (Live Execution Phase)
# =============================================================================


def _format_peaks_compact(
    peak_names: list[str],
    max_display: int = _DEFAULT_MAX_PEAKS_DISPLAY,
) -> str:
    """Format peak names in a compact inline format.

    Examples:
    --------
    - "38N-H"
    - "12N-H, 15N-H"
    - "12N-H, 15N-H, …" (if more than max_display)
    """
    if not peak_names:
        return ""

    if len(peak_names) <= max_display:
        return ", ".join(peak_names)

    shown = ", ".join(peak_names[:max_display])
    return f"{shown}, …"


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable form with consistent decimal format.

    Examples: " 9.07s", "56.58s", " 1.34m", "12.43h", " 1.34d"
    Right-aligned to 6 chars for proper column alignment.
    """
    if seconds < _SECONDS_PER_MINUTE:
        return f"{seconds:6.2f}s"
    if seconds < _SECONDS_PER_HOUR:
        return f"{seconds / _SECONDS_PER_MINUTE:6.2f}m"
    if seconds < _SECONDS_PER_DAY:
        return f"{seconds / _SECONDS_PER_HOUR:6.2f}h"
    return f"{seconds / _SECONDS_PER_DAY:6.2f}d"


def _format_chi2_delta(old: float | None, new: float, precision: int = 2) -> Text:
    """Format χ² with refinement delta and proper coloring.

    Rules (from spec):
    - If displayed digits are identical: neutral color
    - If new < old and digits differ: only new value in bold green
    - If new > old and digits differ: only new value in bold red
    - Never color the arrow or old value
    """
    new_str = f"{new:.{precision}f}"

    if old is None:
        # First time seeing this cluster
        return Text(new_str, style="bold")

    old_str = f"{old:.{precision}f}"

    # Build the delta text
    result = Text()
    result.append(old_str)
    result.append(" → ", style="dim")

    if old_str == new_str:
        # Displayed digits identical - neutral
        result.append(new_str)
    elif new < old:
        # Improvement - bold green for new value only
        result.append(new_str, style="metric.good")
    else:
        # Degradation - bold red for new value only
        result.append(new_str, style="metric.bad")

    return result


def _build_cluster_log_line(
    timestamp: str,
    cluster_id: str,
    peak_names: list[str],
    success: bool,
    chisqr: float,
    prev_chisqr: float | None,
    fit_time: float,
    message: str | None = None,
    warnings: list[str] | None = None,
) -> Text:
    """Build a single cluster completion log line.

    Format (success): [HH:MM:SS]  Cluster X    χ²: old → new  Time: 0.5s  (peaks)
    Format (warning): [HH:MM:SS]  Cluster X    χ²: old → new  Time: 0.5s  (peaks)
                     ⚠ Warning (reason)
    Format (failure): [HH:MM:SS]  Cluster X    ✗ FAILED  (peaks)  (reason)
    """
    line = Text()

    # Timestamp - dim
    line.append(f"[{timestamp}]", style="dim")
    line.append("  ")

    # Cluster ID (fixed width for alignment)
    line.append(f"Cluster {cluster_id:<3}", style="key")
    line.append("  ")

    if not success:
        # Failure case: show failure prominently, then peaks, then reason
        line.append(f"{icon('error')} FAILED", style="bold red")
        line.append("  ")

        # Peaks
        peaks_str = _format_peaks_compact(peak_names)
        if peaks_str:
            line.append(f"({peaks_str})", style="dim")

        # Failure reason
        if message:
            line.append(f"  ({message})", style="dim red")
    else:
        # Success/Warning case: χ², time, peaks, then warning if any
        # χ² display
        line.append("χ²: ")
        chi2_text = _format_chi2_delta(prev_chisqr, chisqr)
        line.append_text(chi2_text)
        line.append("  ")

        # Time (always dim per spec)
        duration_str = _format_duration(fit_time)
        line.append(f"Time: {duration_str}", style="dim")
        line.append("  ")

        # Peaks - compact, dim
        peaks_str = _format_peaks_compact(peak_names)
        if peaks_str:
            line.append(f"({peaks_str})", style="dim")

        # Warning indicator (if any)
        if warnings:
            line.append("  ")
            line.append(f"{icon('warn')} ", style="metric.warn")
            warn_str = ", ".join(warnings[:_MAX_WARNINGS_SHOWN])  # Limit warnings shown
            line.append(f"({warn_str})", style="metric.warn")

    return line


class FitProgressTracker:
    """Tracks and displays live fitting progress using 'Stream + Sticky Footer'.

    Encapsulates the context manager logic previously in live_fit_display.
    """

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.stats: dict[str, Any] = {
            "success": 0,
            "warn": 0,
            "fail": 0,
            "chi2_values": [],
            "start_time": time.time(),
            "completed": 0,
        }
        self.cluster_history: dict[str, float] = {}
        self.progress = self._create_progress()
        self.task_id = self.progress.add_task(
            "Fitting...", total=total_steps, stats_text="Initializing..."
        )

    def _create_progress(self) -> Progress:
        return Progress(
            SpinnerColumn(spinner_name="dots"),
            BarColumn(bar_width=30, style="dim blue", complete_style="green"),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("[dim]|[/dim]"),
            TimeRemainingColumn(),
            TextColumn("[dim]|[/dim]"),
            TextColumn("{task.fields[stats_text]}"),
            console=console,
            transient=True,
        )

    def __enter__(self) -> Callable[..., None]:
        """Start the progress display."""
        self.progress.start()
        return self.update

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Stop the progress display."""
        self.progress.stop()

    def update(
        self,
        cluster_id: str | None = None,
        success: bool = False,
        chisqr: float = 0.0,
        fit_time: float = 0.0,
        message: str | None = None,
        warnings: list[str] | None = None,
        advance: int = 1,
        status_message: str | None = None,
        peak_names: list[str] | None = None,
    ) -> None:
        """Update the display with a cluster completion or status message."""
        # Handle status messages (iteration headers, phase transitions)
        if status_message:
            self.progress.console.print()
            self.progress.console.print(status_message, markup=True, highlight=False)
            return

        if cluster_id is None:
            return

        # Update stats
        prev_chisqr = self._update_stats(cluster_id, success, chisqr, warnings)

        # Build and print the log line
        self._print_log_line(
            cluster_id, peak_names, success, chisqr, prev_chisqr, fit_time, message, warnings
        )

        # Update footer
        self.progress.update(self.task_id, advance=advance, stats_text=self._format_footer_stats())

    def _update_stats(
        self, cluster_id: str, success: bool, chisqr: float, warnings: list[str] | None
    ) -> float | None:
        """Update internal statistics and return previous chisqr."""
        prev_chisqr = self.cluster_history.get(cluster_id)

        if success:
            self.cluster_history[cluster_id] = chisqr

        is_first_time = prev_chisqr is None

        if not success:
            if is_first_time:
                self.stats["fail"] += 1
        else:
            has_warnings = bool(warnings) or chisqr > _HIGH_REDUCED_CHI2_WARNING_THRESHOLD
            if has_warnings:
                if is_first_time:
                    self.stats["warn"] += 1
            elif is_first_time:
                self.stats["success"] += 1
            self.stats["chi2_values"].append(chisqr)

        self.stats["completed"] += 1
        return prev_chisqr

    def _print_log_line(
        self,
        cluster_id: str,
        peak_names: list[str] | None,
        success: bool,
        chisqr: float,
        prev_chisqr: float | None,
        fit_time: float,
        message: str | None,
        warnings: list[str] | None,
    ) -> None:
        timestamp = datetime.datetime.now(datetime.UTC).astimezone().strftime("%H:%M:%S")
        all_warnings = list(warnings or [])

        if (
            success
            and chisqr > _HIGH_REDUCED_CHI2_WARNING_THRESHOLD
            and "High Red. χ²" not in str(all_warnings)
        ):
            all_warnings.insert(0, f"High Red. χ²={chisqr:.1f}")

        log_line = _build_cluster_log_line(
            timestamp=timestamp,
            cluster_id=cluster_id,
            peak_names=peak_names or [],
            success=success,
            chisqr=chisqr,
            prev_chisqr=prev_chisqr,
            fit_time=fit_time,
            message=message,
            warnings=all_warnings if success else None,
        )
        self.progress.console.print(log_line, highlight=False)

    def _format_footer_stats(self) -> str:
        """Format statistics for the sticky footer."""
        parts = []
        if self.stats["success"] > 0:
            parts.append(f"[metric.good]Success: {self.stats['success']}[/metric.good]")
        if self.stats["warn"] > 0:
            parts.append(f"[metric.warn]Warn: {self.stats['warn']}[/metric.warn]")
        if self.stats["fail"] > 0:
            parts.append(f"[metric.bad]Fail: {self.stats['fail']}[/metric.bad]")

        if self.stats["chi2_values"]:
            avg_chi2 = sum(self.stats["chi2_values"]) / len(self.stats["chi2_values"])
            parts.append(f"Avg Red. χ²: {avg_chi2:.2f}")

        elapsed = time.time() - self.stats["start_time"]
        if elapsed > 0 and self.stats["completed"] > 0:
            throughput = self.stats["completed"] / elapsed
            parts.append(f"{throughput:.1f} clusters/s")

        return " | ".join(parts) if parts else "Processing..."


def live_fit_display(total_steps: int) -> FitProgressTracker:
    """Context manager for live fitting progress with streaming log.

    Implements the Stream + Sticky Footer pattern:
    - Cluster completions are printed to scrolling log (permanent)
    - Progress bar with stats updates at the bottom (sticky footer)

    Returns:
    -------
        FitProgressTracker instance (acting as context manager)
    """
    return FitProgressTracker(total_steps)


# =============================================================================
# Post-Fit Report (Actionable Summary)
# =============================================================================


def display_post_fit_summary(  # noqa: PLR0915
    total_time: float,
    total_clusters: int,
    successful_fits: int,
    usable_non_converged: int,
    unusable: int,
    chi_sq_stats: dict[str, float | None],
    clusters_to_review: list[dict[str, Any]],
    output_dir: str,
) -> None:
    """Display the final executive summary report.

    Goal: Actionability - What needs manual attention?

    Shows:
    - Summary section with total time, success rate, mean χ²
    - Action Required table for problematic clusters
    - Output location
    """
    console.print()

    # --- Build Summary Content ---
    summary_table = Table.grid(padding=(0, 2))
    summary_table.add_column(width=2)
    summary_table.add_column(width=16, style="key")
    summary_table.add_column()

    # Format time nicely
    if total_time < _SECONDS_PER_MINUTE:
        time_str = f"{total_time:.1f}s"
    else:
        minutes = int(total_time // _SECONDS_PER_MINUTE)
        seconds = int(total_time % _SECONDS_PER_MINUTE)
        time_str = f"{minutes}m {seconds}s"

    success_rate = (successful_fits / total_clusters * 100) if total_clusters > 0 else 0
    mean_chi2 = chi_sq_stats.get("Mean")

    summary_table.add_row(icon("bullet"), "Total Time:", time_str)
    summary_table.add_row(
        icon("bullet"),
        "Converged:",
        f"{success_rate:.1f}% ({successful_fits}/{total_clusters})",
    )
    summary_table.add_row(
        icon("bullet"),
        "Usable, not converged:",
        str(usable_non_converged),
    )
    summary_table.add_row(icon("bullet"), "Unusable:", str(unusable))
    summary_table.add_row(
        icon("bullet"),
        "Mean Red. χ²:",
        f"{mean_chi2:.2f}" if mean_chi2 is not None else "N/A (no usable outcomes)",
    )

    # --- Build Action Required Section ---
    action_section = None
    if clusters_to_review:
        action_table = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="subheader",
            padding=(0, 1),
            collapse_padding=True,
        )
        action_table.add_column("ID / Peaks", style="key", no_wrap=False, max_width=40)
        action_table.add_column("Issue", no_wrap=True)
        action_table.add_column("Metric", style="dim")

        for item in clusters_to_review:
            # Format ID / Peaks column
            label = item.get("label", item["id"])
            id_peaks = f"{item['id']} ({label})" if label != item["id"] else str(item["id"])

            # Issue column with color
            issue_text = Text(item["status"], style=item.get("status_color", "yellow"))

            # Metric column
            if item.get("chi_sq") is None or "Bounds" in item["status"]:
                metric = item.get("details", "-")
            else:
                metric = f"Red. χ²={item['chi_sq']:.2f}"

            action_table.add_row(id_peaks, issue_text, metric)

        action_section = Table.grid()
        action_section.add_column()
        action_section.add_row(
            Text(f"Action Required ({len(clusters_to_review)})", style="subheader")
        )
        action_section.add_row(
            Text("(These clusters failed or look suspicious)", style="dim italic")
        )
        action_section.add_row(Text(""))
        action_section.add_row(action_table)

    # --- Assemble Panel Content ---
    content = Table.grid(padding=(0, 0))
    content.add_column()

    content.add_row(Text("Summary", style="bold underline"))
    content.add_row(Text(""))
    content.add_row(summary_table)

    if action_section:
        content.add_row(Text(""))
        content.add_row(action_section)

    # Create the boxed panel
    panel = Panel(
        content,
        title="[bold green]Run Complete[/bold green]",
        title_align="left",
        border_style="green",
        box=box.HEAVY,
        padding=(1, 2),
    )

    console.print(panel)
    console.print(f"   Results saved to: [path]{display_path(output_dir)}[/path]")
    console.print()
