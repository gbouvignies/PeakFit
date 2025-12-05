"""Live cluster status display for fitting progress.

This module provides a live-updating display that shows fitting progress
with spinners for in-progress clusters and updates as they complete,
similar to uv sync or homebrew.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from rich.console import Group
from rich.live import Live
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn
from rich.text import Text

from peakfit.ui.console import console

if TYPE_CHECKING:
    from collections.abc import Sequence

# Maximum number of recently completed clusters to show
MAX_RECENT_COMPLETED = 8

# Threshold for using compact mode (progress bar only)
COMPACT_MODE_THRESHOLD = 20


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

    For small numbers of clusters, shows each cluster with status.
    For large numbers, shows a compact progress bar with recent completions.

    Usage:
        with LiveClusterDisplay(clusters) as display:
            display.mark_running(cluster_id)
            # ... fit cluster ...
            display.mark_completed(cluster_id, cost=1.23e-4, n_evaluations=42, time_sec=1.5)
    """

    cluster_names: dict[int, list[str]] = field(default_factory=dict)
    _statuses: dict[int, ClusterStatus] = field(default_factory=dict, init=False)
    _live: Live | None = field(default=None, init=False)
    _progress: Progress | None = field(default=None, init=False)
    _task_id: TaskID | None = field(default=None, init=False)
    _step_name: str = field(default="Fitting", init=False)
    _n_workers: int = field(default=1, init=False)
    _recent_completed: deque[Any] = field(
        default_factory=lambda: deque(maxlen=MAX_RECENT_COMPLETED)
    )
    _use_compact_mode: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Initialize cluster statuses."""
        for cluster_id, peak_names in self.cluster_names.items():
            self._statuses[cluster_id] = ClusterStatus(
                cluster_id=cluster_id,
                peak_names=peak_names,
            )
        # Use compact mode for many clusters
        self._use_compact_mode = len(self._statuses) > COMPACT_MODE_THRESHOLD

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
        if not console.is_terminal:
            return self

        # Create progress bar for compact mode
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="cyan", finished_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[dim]{task.completed}/{task.total}[/dim]"),
            console=console,
            transient=True,
        )

        total = len(self._statuses)
        description = self._get_description()
        self._task_id = self._progress.add_task(description, total=total)

        self._live = Live(
            self._render(),
            console=console,
            refresh_per_second=10,
            transient=True,  # Display disappears when done
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args) -> None:
        """Stop the live display and print summary."""
        if self._live is not None:
            self._live.__exit__(*args)
            self._live = None
            self._progress = None

        # Print final summary
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
            self._refresh()

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

        # Track recently completed for display
        self._recent_completed.append(status)

        # Update progress bar
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, advance=1)

        self._refresh()

    def _refresh(self) -> None:
        """Refresh the live display."""
        if self._live is not None:
            self._live.update(self._render())

    def _print_summary(self) -> None:
        """Print a summary line after all fitting is complete."""
        total = len(self._statuses)
        successful = sum(1 for s in self._statuses.values() if s.state == ClusterState.COMPLETED)
        failed = sum(1 for s in self._statuses.values() if s.state == ClusterState.FAILED)
        total_time = sum(s.time_sec or 0.0 for s in self._statuses.values())

        if failed == 0:
            console.print(
                f"[green]✓[/green] All {total} clusters converged "
                f"[dim](total time: {total_time:.1f}s)[/dim]"
            )
        else:
            console.print(
                f"[yellow]⚠[/yellow] {successful}/{total} clusters converged, "
                f"{failed} did not converge [dim](total time: {total_time:.1f}s)[/dim]"
            )

    def _render(self) -> Group:
        """Render the current status display."""
        elements = []

        # Progress bar
        if self._progress is not None:
            elements.append(self._progress)

        # Recently completed clusters (scrolling list)
        if self._recent_completed:
            recent_lines = []
            for status in self._recent_completed:
                line = self._format_completed_line(status)
                recent_lines.append(line)
            if recent_lines:
                elements.append(Text("\n".join(recent_lines)))

        return Group(*elements) if elements else Group(Text(""))

    def _format_completed_line(self, status: ClusterStatus) -> str:
        """Format a single completed cluster line."""
        # Build peak names string
        peaks_str = ", ".join(status.peak_names[:3])
        if len(status.peak_names) > 3:
            peaks_str += f" (+{len(status.peak_names) - 3})"

        # Pad to align results
        peaks_str = peaks_str.ljust(30)

        if status.state == ClusterState.COMPLETED:
            return (
                f"[green]✓[/green] {peaks_str} "
                f"[dim]χ²=[/dim][cyan]{status.cost:.2e}[/cyan] "
                f"[dim]iter=[/dim]{status.n_evaluations} "
                f"[dim]time=[/dim]{status.time_sec:.1f}s"
            )
        else:
            msg = status.message or "Did not converge"
            return (
                f"[yellow]✗[/yellow] {peaks_str} "
                f"[yellow]{msg}[/yellow] "
                f"[dim]χ²={status.cost:.2e}[/dim]"
            )


__all__ = ["ClusterState", "ClusterStatus", "LiveClusterDisplay"]
