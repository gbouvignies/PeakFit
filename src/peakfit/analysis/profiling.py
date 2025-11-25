"""Performance profiling utilities for PeakFit.

This module provides tools for measuring and analyzing performance
of different fitting strategies.
"""

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class TimingResult:
    """Result of a timed operation."""

    name: str
    elapsed: float
    count: int = 1
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def per_call(self) -> float:
        """Average time per call."""
        return self.elapsed / max(1, self.count)


@dataclass
class ProfileReport:
    """Performance profile report."""

    timings: list[TimingResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float | None = None

    @property
    def total_time(self) -> float:
        """Total elapsed time."""
        if self.end_time is None:
            return time.perf_counter() - self.start_time
        return self.end_time - self.start_time

    def add_timing(self, result: TimingResult) -> None:
        """Add a timing result."""
        self.timings.append(result)

    def finalize(self) -> None:
        """Mark the profile as complete."""
        self.end_time = time.perf_counter()

    def summary(self) -> str:
        """Generate a summary report."""
        lines = ["Performance Profile Summary", "=" * 40]

        if not self.timings:
            lines.append("No timings recorded.")
        else:
            # Sort by elapsed time descending
            sorted_timings = sorted(self.timings, key=lambda t: t.elapsed, reverse=True)

            for timing in sorted_timings:
                pct = (timing.elapsed / self.total_time * 100) if self.total_time > 0 else 0
                lines.append(f"{timing.name}:")
                lines.append(f"  Time: {timing.elapsed:.3f}s ({pct:.1f}%)")
                if timing.count > 1:
                    lines.append(f"  Count: {timing.count}")
                    lines.append(f"  Per call: {timing.per_call * 1000:.2f}ms")
                for key, value in timing.metadata.items():
                    lines.append(f"  {key}: {value}")
                lines.append("")

        lines.append(f"Total time: {self.total_time:.3f}s")
        return "\n".join(lines)


class Profiler:
    """Performance profiler for PeakFit fitting operations."""

    def __init__(self) -> None:
        """Initialize the profiler."""
        self.report = ProfileReport()
        self._current_timer: float | None = None
        self._current_name: str = ""

    @contextmanager
    def timer(self, name: str, count: int = 1, **metadata: object) -> Generator[None]:
        """Context manager for timing a block of code.

        Args:
            name: Name of the operation
            count: Number of operations (for averaging)
            **metadata: Additional metadata to record
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.report.add_timing(
                TimingResult(name=name, elapsed=elapsed, count=count, metadata=metadata)
            )

    def start(self, name: str) -> None:
        """Start timing an operation.

        Args:
            name: Name of the operation
        """
        self._current_name = name
        self._current_timer = time.perf_counter()

    def stop(self, count: int = 1, **metadata: object) -> float:
        """Stop timing and record the result.

        Args:
            count: Number of operations
            **metadata: Additional metadata

        Returns:
            Elapsed time in seconds
        """
        if self._current_timer is None:
            msg = "No timer started"
            raise RuntimeError(msg)

        elapsed = time.perf_counter() - self._current_timer
        self.report.add_timing(
            TimingResult(
                name=self._current_name,
                elapsed=elapsed,
                count=count,
                metadata=metadata,
            )
        )
        self._current_timer = None
        return elapsed

    def finalize(self) -> ProfileReport:
        """Finalize the profile and return the report.

        Returns:
            ProfileReport with all timings
        """
        self.report.finalize()
        return self.report


def compare_fitting_methods(
    clusters: list,
    noise: float,
    refine_iterations: int = 1,
    *,
    fixed: bool = False,
) -> dict[str, ProfileReport]:
    """Compare performance of different fitting methods.

    Args:
        clusters: List of clusters to fit
        noise: Noise level
        refine_iterations: Number of refinement iterations
        fixed: Whether to fix positions

    Returns:
        Dictionary mapping method name to ProfileReport
    """
    from peakfit.core.fitting.optimizer import fit_clusters

    # No parallel testing available — this function only tests sequential fitting

    results = {}

    # Test sequential fitting
    profiler = Profiler()
    with profiler.timer("fit_clusters", count=len(clusters)):
        fit_clusters(
            clusters=clusters,
            noise=noise,
            refine_iterations=refine_iterations,
            fixed=fixed,
            verbose=False,
        )
    profiler.finalize()
    results["fast_sequential"] = profiler.report

    # Parallel fitting removed; no parallel testing available

    return results


def estimate_optimal_workers(
    clusters: list,
    noise: float,
    max_workers: int | None = None,
) -> tuple[int, dict[int, float]]:
    """Estimate optimal worker count for the (removed) parallel fitting strategy.

    Since parallel processing has been removed, this function returns 1 and a
    timing for the sequential run for convenience.

    Args:
        clusters: List of clusters to fit
        noise: Noise level
        max_workers: Ignored; retained for compatibility

    Returns:
        Tuple of (1, {1: elapsed_time})
    """
    # Parallel fitting removed - default to single worker.
    if max_workers is None:
        max_workers = 1

    # Only sequential timing available
    from peakfit.core.fitting.optimizer import fit_clusters

    start = time.perf_counter()
    fit_clusters(clusters=clusters, noise=noise, refine_iterations=0, fixed=False, verbose=False)
    elapsed = time.perf_counter() - start
    timings = {1: elapsed}
    return 1, timings
