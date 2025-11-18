# PeakFit Optimization Guide

This guide explains how to use PeakFit's performance optimization features to speed up your NMR lineshape fitting workflows.

## Quick Start

```bash
# Check system optimization status
peakfit info

# Benchmark your dataset to find optimal settings
peakfit benchmark spectrum.ft2 peaks.list

# Run with fast scipy optimization (bypasses lmfit overhead)
peakfit fit spectrum.ft2 peaks.list --fast

# Run with parallel processing (multiple clusters simultaneously)
peakfit fit spectrum.ft2 peaks.list --parallel

# Control number of parallel workers
peakfit fit spectrum.ft2 peaks.list --parallel --workers 8
```

## Optimization Methods

### 1. Fast Scipy Optimization (`--fast`)

Bypasses lmfit's Python wrapper overhead by directly interfacing with `scipy.optimize.least_squares`.

**When to use:**
- Single-core environments
- Datasets with few clusters
- When you want consistent, predictable performance

**Expected speedup:** 10-50x faster than standard lmfit fitting

```bash
peakfit fit spectrum.ft2 peaks.list --fast
```

### 2. Parallel Processing (`--parallel`)

Fits multiple clusters concurrently using Python's multiprocessing.

**When to use:**
- Multi-core systems (4+ CPU cores)
- Datasets with many clusters (10+)
- When wall-clock time matters more than CPU usage

**Expected speedup:** Scales with number of clusters and cores

```bash
# Use all available CPU cores
peakfit fit spectrum.ft2 peaks.list --parallel

# Specify number of workers
peakfit fit spectrum.ft2 peaks.list --parallel --workers 4
```

### 3. JIT Compilation (Automatic)

When Numba is installed, lineshape functions are JIT-compiled for better performance.

**Installation:**
```bash
pip install numba
# or
pip install peakfit[performance]
```

**Check status:**
```bash
peakfit info
```

## Performance Benchmarking

Use the benchmark command to find optimal settings for your data:

```bash
peakfit benchmark spectrum.ft2 peaks.list --iterations 3
```

This will:
- Compare fast sequential vs parallel fitting
- Report average times and speedup
- Recommend the best approach for your data

## Advanced Usage

### Profiling Your Fits

For detailed performance analysis:

```python
from peakfit.core.profiling import Profiler

profiler = Profiler()

with profiler.timer("data_loading"):
    # Load your data
    pass

with profiler.timer("cluster_fitting", count=len(clusters)):
    # Fit clusters
    pass

report = profiler.finalize()
print(report.summary())
```

### Caching for Repeated Operations

Use caching for repeated computations:

```python
from peakfit.core.caching import memoize_array_function, get_cache_stats

@memoize_array_function(maxsize=128)
def expensive_computation(data):
    # Your computation here
    pass

# Check cache performance
stats = get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.1f}%")
```

### Custom Worker Configuration

For fine-grained control over parallel processing:

```python
from peakfit.core.parallel import fit_clusters_parallel_refined

params = fit_clusters_parallel_refined(
    clusters=clusters,
    noise=noise,
    refine_iterations=2,
    fixed=False,
    n_workers=8,
    verbose=True,
)
```

## Performance Tips

1. **Profile First**: Use `peakfit benchmark` to understand your data's characteristics

2. **Match Method to Data**:
   - Few clusters (< 5): Use `--fast`
   - Many clusters (> 10): Try `--parallel`
   - Medium clusters: Benchmark both

3. **Memory Considerations**:
   - Parallel fitting uses more memory (one process per worker)
   - For memory-constrained systems, use `--fast`

4. **Cluster Size Matters**:
   - Large clusters benefit more from JIT
   - Small clusters benefit more from parallelization

5. **Refinement Iterations**:
   - More iterations = more benefit from optimization
   - Consider `--refine 2` or `--refine 3` for complex data

## Troubleshooting

### Parallel fitting slower than expected

1. Check the number of clusters:
   ```bash
   peakfit benchmark spectrum.ft2 peaks.list
   ```

2. Multiprocessing has overhead; for few clusters, use `--fast`

3. Reduce workers if memory is limited:
   ```bash
   peakfit fit spectrum.ft2 peaks.list --parallel --workers 4
   ```

### JIT not providing speedup

1. Verify Numba is installed:
   ```bash
   peakfit info
   ```

2. JIT compilation has startup cost; benefit appears on repeated calls

3. Lineshapes are typically only ~0.3% of total runtime; focus on other optimizations

### Out of memory errors

1. Reduce parallel workers
2. Use sequential fast fitting (`--fast`)
3. Process data in smaller batches

## API Reference

### Command Line

```bash
# System info and optimization status
peakfit info [--benchmark]

# Performance benchmark
peakfit benchmark SPECTRUM PEAKLIST [--iterations N]

# Fit with optimizations
peakfit fit SPECTRUM PEAKLIST [--fast] [--parallel] [--workers N]
```

### Python API

```python
# Fast fitting
from peakfit.core.fast_fit import fit_clusters_fast
params = fit_clusters_fast(clusters, noise, refine_iterations=1)

# Parallel fitting
from peakfit.core.parallel import fit_clusters_parallel_refined
params = fit_clusters_parallel_refined(clusters, noise, n_workers=8)

# Profiling
from peakfit.core.profiling import Profiler, ProfileReport

# Caching
from peakfit.core.caching import LRUCache, memoize_array_function
```

## Version History

- **v0.2.0**: Added fast scipy optimization, parallel fitting, profiling, caching
- **v0.1.0**: Initial JIT optimization support
