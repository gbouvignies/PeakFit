"""Diagnostic utilities for performance analysis."""

import time
from typing import Any

import numpy as np


class BackendTimer:
    """Timer for measuring backend performance."""

    def __init__(self, name: str):
        """Initialize timer.

        Args:
            name: Name for this timer
        """
        self.name = name
        self.timings: list[tuple[str, float]] = []
        self.start_time: float | None = None

    def start(self) -> None:
        """Start timing."""
        self.start_time = time.perf_counter()

    def lap(self, label: str) -> float:
        """Record a lap time.

        Args:
            label: Label for this lap

        Returns:
            Time since start in seconds
        """
        if self.start_time is None:
            self.start()

        elapsed = time.perf_counter() - self.start_time
        self.timings.append((label, elapsed))
        return elapsed

    def report(self) -> str:
        """Generate timing report.

        Returns:
            Formatted timing report
        """
        lines = [f"Timing Report: {self.name}", "=" * 60]

        if not self.timings:
            lines.append("No timings recorded")
            return "\n".join(lines)

        total = self.timings[-1][1] if self.timings else 0
        lines.append(f"Total time: {total:.3f}s\n")
        lines.append("Breakdown:")

        prev_time = 0.0
        for label, elapsed in self.timings:
            delta = elapsed - prev_time
            pct = (delta / total * 100) if total > 0 else 0
            lines.append(f"  {label:30s}: {delta:7.3f}s ({pct:5.1f}%)")
            prev_time = elapsed

        return "\n".join(lines)


def profile_backend_call(func: Any, *args: Any, n_warmup: int = 5, n_iter: int = 100, **kwargs: Any) -> dict[str, Any]:
    """Profile a backend function call.

    Args:
        func: Function to profile
        *args: Arguments to function
        n_warmup: Number of warmup iterations
        n_iter: Number of timing iterations
        **kwargs: Keyword arguments to function

    Returns:
        Dictionary with profiling results
    """
    # Warmup
    for _ in range(n_warmup):
        result = func(*args, **kwargs)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()

    # Timing
    times = []
    for _ in range(n_iter):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times_array = np.array(times)

    return {
        "mean": float(np.mean(times_array)),
        "std": float(np.std(times_array)),
        "min": float(np.min(times_array)),
        "max": float(np.max(times_array)),
        "median": float(np.median(times_array)),
        "n_iter": n_iter,
    }


def compare_backends_on_real_data(array_size: int = 100) -> dict[str, dict[str, Any]]:
    """Compare backend performance on realistic array sizes.

    Args:
        array_size: Size of arrays to test (realistic for fitting)

    Returns:
        Dictionary mapping backend names to profiling results
    """
    from peakfit.core import backend

    # Realistic parameters
    dx = np.linspace(-50, 50, array_size)
    r2, aq, end, off, phase = 5.0, 0.1, 1.0, 0.0, 0.0

    results = {}
    available = backend.get_available_backends()

    for backend_name in available:
        backend.set_backend(backend_name)

        # Test complex function (SP2 - most expensive)
        func = backend.get_sp2_func()

        profile = profile_backend_call(func, dx, r2, aq, end, off, phase, n_warmup=10, n_iter=100)

        results[backend_name] = profile

    return results


def diagnose_platform() -> dict[str, Any]:
    """Diagnose platform and JAX configuration.

    Returns:
        Dictionary with platform information
    """
    import platform
    import sys

    info = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
    }

    # Check JAX availability and configuration
    try:
        import jax

        info["jax_available"] = True
        info["jax_version"] = jax.__version__

        # Get JAX platform
        try:
            devices = jax.devices()
            info["jax_devices"] = [str(d) for d in devices]
            info["jax_default_backend"] = devices[0].platform if devices else "unknown"
        except Exception as e:
            info["jax_devices_error"] = str(e)

        # Check for M1/ARM
        if platform.machine() in ("arm64", "aarch64"):
            info["is_arm"] = True
            info["note"] = "ARM platform detected - JAX may not be optimized"
        else:
            info["is_arm"] = False

    except ImportError:
        info["jax_available"] = False

    return info
