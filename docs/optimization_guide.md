# PeakFit Optimization Guide

This guide explains how to optimize PeakFit's performance for your NMR lineshape fitting workflows.

## Overview

PeakFit uses pure NumPy implementations for all lineshape calculations, providing excellent compatibility and maintainability. Performance can be optimized through:

1. **Parallel processing** - Fit multiple clusters simultaneously
2. **Fast scipy optimization** - Direct interface with scipy.optimize
3. **Profiling** - Identify bottlenecks

## Quick Start

```bash
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
from peakfit.analysis.profiling import Profiler

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

###

### Custom Worker Configuration

For fine-grained control over parallel processing:

```python
from peakfit.fitting.parallel import fit_clusters_parallel_refined

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

4. **Refinement Iterations**:
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
from peakfit.fitting.fast_fit import fit_clusters_fast
params = fit_clusters_fast(clusters, noise, refine_iterations=1)

# Parallel fitting
from peakfit.fitting.parallel import fit_clusters_parallel_refined
params = fit_clusters_parallel_refined(clusters, noise, n_workers=8)

# Profiling
from peakfit.analysis.profiling import Profiler, ProfileReport

##
```

## Version History

- **v2025.12.0**: Removed Numba dependency, NumPy-only implementation
- **v0.2.0**: Added fast scipy optimization, parallel fitting, profiling
