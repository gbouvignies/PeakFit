# Parallel Fitting Status

## Current Implementation

PeakFit now uses `ThreadPoolExecutor` to fit multiple clusters concurrently. Each worker thread:
- Processes one cluster independently
- Limits BLAS threads to 1 using `threadpool_limits(limits=1)` to prevent oversubscription
- Returns fitted parameters, optimization result, and timing

## Limitations: Python's Global Interpreter Lock (GIL)

**Important**: Due to Python's GIL, `ThreadPoolExecutor` does not achieve true multi-core parallelism for CPU-bound tasks like numerical optimization.

### Why ThreadPoolExecutor Shows ~100% CPU

The GIL serializes Python bytecode execution, meaning only one thread can execute Python code at a time. Even though:
- NumPy/SciPy operations release the GIL
- Multiple threads are created
- BLAS is limited to 1 thread per worker

The optimization loop itself involves Python-level function calls, parameter updates, and convergence checks that are GIL-limited.

### Evidence

```bash
$ time uv run peakfit fit data/pseudo3d.ft2 data/pseudo3d.list --z-values data/b1_offsets.txt --output Fits
10.95s user 1.40s system 105% cpu 11.670 total
```

The `105% cpu` indicates ~1 core used (100% = 1 core on macOS). A truly parallel implementation on 10 cores would show ~800-1000% CPU.

## Why Not Use Process-Based Parallelism?

Process-based parallelism (`multiprocessing.Pool` or `joblib.Parallel`) would bypass the GIL and achieve true multi-core utilization. However:

1. **Pickling Issues**: The current fitting functions use closures and nested references (e.g., `_residual_wrapper`) that cannot be pickled for inter-process communication

2. **Code Refactoring Required**: To enable process-based parallelism, we would need to:
   - Make all worker functions truly top-level (no closures)
   - Pass all dependencies as explicit parameters
   - Ensure all data structures are picklable
   - This is a significant refactor that changes the code architecture

## Benefits of Current Approach

Despite the GIL limitation, the current implementation:

1. **Prevents BLAS Oversubscription**: By using `threadpool_limits(limits=1)` inside each worker, we prevent OpenBLAS/MKL from spawning excessive threads that would cause severe performance degradation on high-core-count CPUs like Threadripper PRO

2. **Clean Code Structure**: The fitting logic remains modular and easy to understand

3. **Preparation for Future**: The thread-based structure can be migrated to process-based parallelism if we refactor the closure dependencies

## Performance on Threadripper PRO 7965WX (24 cores)

On high-core-count CPUs, **preventing BLAS oversubscription is more important than achieving thread-based parallelism**:

- ❌ **Without threadpoolctl**: Each core spawns multiple BLAS threads → 24+ cores × many threads = severe thrashing, performance collapse
- ✅ **With threadpoolctl(limits=1)**: Only 1 BLAS thread per core → stable, predictable performance

The current implementation ensures efficient single-core operation, which is better than thrashing 24 cores.

## Future: True Multi-Core Parallelism

To achieve true multi-core speedup, we would need to:

1. Add `joblib` to dependencies
2. Refactor `_fit_single_cluster` to be a top-level function with no closures
3. Make `_residual_wrapper` picklable (possibly by passing it as a class method)
4. Use `joblib.Parallel(n_jobs=-1, backend="loky")` for process-based parallelism
5. Handle potential overhead from IPC serialization

This is a substantial refactor and should be done only if profiling shows it's necessary for typical workloads.

## Recommendations

1. **Current state is acceptable** for:
   - Small to medium datasets
   - Fits completing in <1 minute
   - Development/testing workflows

2. **Consider process-based parallelism** if:
   - Fitting takes >5 minutes
   - Dataset has many independent clusters
   - User has high-core-count CPU (>16 cores)
   - Profiling shows fitting is the bottleneck

3. **Alternative: Numba-parallelized fitting**:
   - Another approach is to parallelize the residual calculations within `least_squares` using Numba's `@njit(parallel=True)`
   - This would provide fine-grained parallelism without GIL limitations
   - Requires refactoring the lineshape evaluation functions
   - Potentially higher speedup for large clusters
