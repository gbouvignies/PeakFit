"""Event handlers for UI updates.

This module provides handlers that listen to domain events and update
the UI accordingly, such as updating progress bars or printing logs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from peakfit.ui.progress import create_progress

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID

from peakfit.core.shared.events import Event, EventType, FitProgressEvent


class RichProgressHandler:
    """Updates a Rich progress bar based on fitting events."""

    def __init__(self) -> None:
        self.progress: Progress | None = None
        self.task_id: TaskID | None = None
        self.current_iteration = 0
        self.total_iterations = 0

    def handle(self, event: Event) -> None:
        """Handle an event."""
        if event.event_type == EventType.FIT_STARTED:
            self._start_progress()
        elif event.event_type in (EventType.CLUSTER_STARTED, EventType.CLUSTER_COMPLETED):
            self._update_progress(event.data)
        elif event.event_type == EventType.FIT_COMPLETED:
            self._stop_progress()
        elif event.event_type == EventType.FIT_PROGRESS and isinstance(event, FitProgressEvent):
            self._update_from_progress_event(event)

    def _start_progress(self) -> None:
        """Initialize and start the progress bar."""
        if self.progress is None:
            self.progress = create_progress(transient=True)
            self.progress.start()
            self.task_id = self.progress.add_task("Fitting clusters...", total=100)

    def _stop_progress(self) -> None:
        """Stop the progress bar."""
        if self.progress is not None:
            self.progress.stop()
            self.progress = None
            self.task_id = None

    def _update_progress(self, data: dict[str, Any]) -> None:
        """Update progress bar from event data."""
        if self.progress is None or self.task_id is None:
            return

        cluster_index = data.get("cluster_index", 0)
        total_clusters = data.get("total_clusters", 1)
        iteration = data.get("iteration", 1)
        total_iterations = data.get("total_iterations", 1)

        # Calculate overall progress
        # Each iteration processes all clusters
        clusters_per_iteration = total_clusters
        total_steps = total_clusters * total_iterations

        current_step = (iteration - 1) * clusters_per_iteration + cluster_index

        self.progress.update(
            self.task_id,
            completed=current_step,
            total=total_steps,
            description=f"Fitting clusters (Iter {iteration}/{total_iterations})...",
        )

    def _update_from_progress_event(self, event: FitProgressEvent) -> None:
        """Update progress from a specific FitProgressEvent."""
        if self.progress is None or self.task_id is None:
            return

        total_steps = event.total_clusters * event.total_iterations
        current_step = (event.current_iteration - 1) * event.total_clusters + event.current_cluster

        self.progress.update(
            self.task_id,
            completed=current_step,
            total=total_steps,
            description=f"Fitting clusters (Iter {event.current_iteration}/{event.total_iterations})...",
        )
