"""Live cluster status display for fitting progress.

This module provides a live-updating display that shows fitting progress
similar to uv sync or homebrew - with a clean progress bar and status updates.

Design principles:
- Clean, minimal UI with clear progress indication
- Progress bar with percentage and counts
- Elapsed time display
- Summary of failures (if any) shown at the end
- No scrolling list during progress (cleaner, less noisy)
"""

from __future__ import annotations

import time as time_module
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
)

from peakfit.ui.console import console, icon

if TYPE_CHECKING:
    from collections.abc import Sequence


class ClusterState(Enum):
    """State of a cluster in the fitting process."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class ClusterStatus:
    """Status information for a single cluster."""

    cluster_id: int
    peak_names: list[str]
    state: ClusterState = ClusterState.PENDING
    cost: float | None = None
    n_evaluations: int | None = None
    time_sec: float | None = None
    message: str | None = None


@dataclass
class LiveClusterDisplay:
    """Live display showing cluster fitting progress.

    Shows a clean progress bar with spinner, percentage, and elapsed time.
    Failed clusters are summarized at the end.

    Usage:
        with LiveClusterDisplay.from_clusters(clusters) as display:
            display.mark_running(cluster_id)
            # ... fit cluster ...
            display.mark_completed(cluster_id, cost=1.23e-4, n_evaluations=42, time_sec=1.5)
    """

    cluster_names: dict[int, list[str]] = field(default_factory=dict)
    _statuses: dict[int, ClusterStatus] = field(default_factory=dict, init=False)
    _progress: Progress | None = field(default=None, init=False)
    _task_id: TaskID | None = field(default=None, init=False)
    _step_name: str = field(default="Fitting", init=False)
    _n_workers: int = field(default=1, init=False)
    _start_time: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        """Initialize cluster statuses."""
        for cluster_id, peak_names in self.cluster_names.items():
            self._statuses[cluster_id] = ClusterStatus(
                cluster_id=cluster_id,
                peak_names=peak_names,
            )

    @classmethod
    def from_clusters(cls, clusters: Sequence) -> LiveClusterDisplay:
        """Create display from a list of cluster objects.

        Args:
            clusters: List of Cluster objects with peaks attribute

        Returns
        -------
            Configured LiveClusterDisplay instance
        """
        cluster_names = {}
        for idx, cluster in enumerate(clusters, 1):
            peak_names = [peak.name for peak in cluster.peaks]
            cluster_names[idx] = peak_names
        return cls(cluster_names=cluster_names)

    def set_step_name(self, name: str) -> None:
        """Set the current step name for display."""
        self._step_name = name

    def set_workers(self, n_workers: int) -> None:
        """Set number of parallel workers for display."""
        self._n_workers = n_workers

    def __enter__(self) -> LiveClusterDisplay:
        """Start the live display."""
        self._start_time = time_module.time()

        if not console.is_terminal:
            return self

        # Create a clean progress bar (similar to uv/homebrew style)
        self._progress = Progress(
            SpinnerColumn(finished_text=f"[success]{icon('check')}[/success]", spinner_name="dots"),
            TextColumn("[progress.description]{task.description}[/progress.description]", justify="left"),
            BarColumn(
                bar_width=40,
                style="dim blue",
                complete_style="green",
                finished_style="success",
            ),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("[dim]•[/dim]"),
            TimeElapsedColumn(),
            console=console,
            transient=True,  # Progress bar disappears, replaced by summary
        )

        total = len(self._statuses)
        description = self._get_description()
        self._task_id = self._progress.add_task(description, total=total)

        self._progress.start()
        return self

    def __exit__(self, *args) -> None:
        """Stop the live display and print summary."""
        if self._progress is not None:
            self._progress.stop()
            self._progress = None

        # Print final summary (this replaces the progress bar line)
        self._print_summary()

    def _get_description(self) -> str:
        """Get the progress bar description."""
        if self._n_workers > 1:
            return f"{self._step_name} ({self._n_workers} workers)"
        return self._step_name

    def mark_running(self, cluster_id: int) -> None:
        """Mark a cluster as currently being fitted."""
        if cluster_id in self._statuses:
            self._statuses[cluster_id].state = ClusterState.RUNNING

    def mark_completed(
        self,
        cluster_id: int,
        *,
        cost: float,
        n_evaluations: int,
        time_sec: float,
        success: bool = True,
        message: str | None = None,
    ) -> None:
        """Mark a cluster as completed.

        Args:
            cluster_id: The cluster ID (1-based index)
            cost: Final cost (chi-squared)
            n_evaluations: Number of function evaluations
            time_sec: Time taken in seconds
            success: Whether the fit converged successfully
            message: Optional message (used for failures)
        """
        if cluster_id not in self._statuses:
            return

        status = self._statuses[cluster_id]
        status.state = ClusterState.COMPLETED if success else ClusterState.FAILED
        status.cost = cost
        status.n_evaluations = n_evaluations
        status.time_sec = time_sec
        status.message = message

        # Update progress bar
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, advance=1)

    def _print_summary(self) -> None:
        """Print a summary line after all fitting is complete."""
        total = len(self._statuses)
        successful = sum(1 for s in self._statuses.values() if s.state == ClusterState.COMPLETED)
        failed = sum(1 for s in self._statuses.values() if s.state == ClusterState.FAILED)
        elapsed = time_module.time() - self._start_time

        # Main summary line
        if failed == 0:
            console.print(
                f"[success]{icon('check')}[/success] [bold]Fitted {total} clusters[/bold] "
                f"[progress.elapsed]in {elapsed:.1f}s[/progress.elapsed]"
            )
        else:
            console.print(
                f"[warning]{icon('warn')}[/warning] [bold]Fitted {total} clusters[/bold] "
                f"({successful} converged, {failed} failed) "
                f"[progress.elapsed]in {elapsed:.1f}s[/progress.elapsed]"
            )

            # Show failed clusters
            failed_statuses = [s for s in self._statuses.values() if s.state == ClusterState.FAILED]
            for status in failed_statuses[:5]:  # Limit to first 5
                peaks_str = ", ".join(status.peak_names[:3])
                if len(status.peak_names) > 3:
                    peaks_str += f" (+{len(status.peak_names) - 3})"
                msg = status.message or "Did not converge"
                console.print(f"  [dim]•[/dim] {peaks_str}: [warning]{msg}[/warning]")

            if len(failed_statuses) > 5:
                console.print(f"  [dim]... and {len(failed_statuses) - 5} more[/dim]")


__all__ = ["ClusterState", "ClusterStatus", "LiveClusterDisplay"]
