"""Performance profiling utilities for PeakFit.

This module provides tools for measuring and analyzing performance
of different fitting strategies.
"""

import multiprocessing as mp
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TimingResult:
    """Result of a timed operation."""

    name: str
    elapsed: float
    count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

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
    n_workers: int | None = None,
) -> dict[str, ProfileReport]:
    """Compare performance of different fitting methods.

    Args:
        clusters: List of clusters to fit
        noise: Noise level
        refine_iterations: Number of refinement iterations
        fixed: Whether to fix positions
        n_workers: Number of workers for parallel fitting

    Returns:
        Dictionary mapping method name to ProfileReport
    """
    from peakfit.fitting.optimizer import fit_clusters
    from peakfit.fitting.parallel import fit_clusters_parallel_refined

    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(clusters))

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

    # Test parallel fitting
    if len(clusters) > 1:
        profiler = Profiler()
        with profiler.timer(
            "fit_clusters_parallel",
            count=len(clusters),
            n_workers=n_workers,
        ):
            fit_clusters_parallel_refined(
                clusters=clusters,
                noise=noise,
                refine_iterations=refine_iterations,
                fixed=fixed,
                n_workers=n_workers,
                verbose=False,
            )
        profiler.finalize()
        results["parallel"] = profiler.report

    return results


def estimate_optimal_workers(
    clusters: list,
    noise: float,
    max_workers: int | None = None,
) -> tuple[int, dict[int, float]]:
    """Estimate optimal number of workers for parallel fitting.

    Args:
        clusters: List of clusters to fit
        noise: Noise level
        max_workers: Maximum workers to test (default: CPU count)

    Returns:
        Tuple of (optimal_workers, timings_dict)
    """
    from peakfit.fitting.parallel import fit_clusters_parallel_refined

    if max_workers is None:
        max_workers = mp.cpu_count()

    # Test a subset of worker counts
    test_workers = [1, 2, 4]
    if max_workers >= 8:
        test_workers.append(8)
    if max_workers >= 16:
        test_workers.append(16)
    test_workers = [w for w in test_workers if w <= max_workers]

    timings = {}

    for n_workers in test_workers:
        start = time.perf_counter()
        fit_clusters_parallel_refined(
            clusters=clusters,
            noise=noise,
            refine_iterations=0,  # Quick test
            fixed=False,
            n_workers=n_workers,
            verbose=False,
        )
        elapsed = time.perf_counter() - start
        timings[n_workers] = elapsed

    # Find optimal
    optimal = min(timings, key=timings.get)
    return optimal, timings
