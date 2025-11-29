# PeakFit Optimization Guide

This guide explains how to optimize PeakFit's performance for your NMR lineshape fitting workflows.

## Overview

PeakFit uses pure NumPy implementations for all lineshape calculations, providing excellent compatibility and maintainability. Performance can be optimized through:

1. **Sequential processing** - Fit clusters sequentially, optimized for predictable and reproducible results
2. **Fast scipy optimization** - Direct interface with scipy.optimize
3. **Profiling** - Identify bottlenecks

# Quick Start

```bash
# Benchmark your dataset to find optimal settings
peakfit benchmark spectrum.ft2 peaks.list

# Run with the default scipy least-squares optimizer (leastsq)
peakfit fit spectrum.ft2 peaks.list --optimizer leastsq

# PeakFit now runs sequentially; parallel cluster fitting support has been removed.
peakfit fit spectrum.ft2 peaks.list
```

## Optimization Methods

### 1. Scipy Least-Squares Optimization (`--optimizer leastsq`)

Use the `leastsq` optimizer (default) which interfaces with `scipy.optimize.least_squares` for fast
sequential fitting. This avoids lmfit wrapper overhead and is a good choice for single-process runs.

**When to use:**

- Single-core environments
- Datasets with few clusters
- When you want consistent, predictable performance

**Expected speedup:** 10-50x faster than standard lmfit fitting

```bash
peakfit fit spectrum.ft2 peaks.list --optimizer leastsq
```

### 2. Sequential processing

PeakFit now performs cluster fitting sequentially using scipy.optimize least squares. This conservative mode reduces the memory footprint and avoids multiprocessing overhead for reproducibility.

**When to use:**

- All systems; default mode: stable and predictable
- For datasets with few clusters where parallelization overhead would be higher than the benefit

**Expected behaviour:** Consistent, single-process wall-clock times; no user-configurable worker count

## Performance Benchmarking

Use the benchmark command to find optimal settings for your data:

```bash
peakfit benchmark spectrum.ft2 peaks.list --iterations 3
```

This will:

- Benchmark the current (sequential) fitting method and report average timings

## Advanced Usage

### Profiling Your Fits

For detailed performance and profiling analysis, use the CLI benchmark and analyze commands:

```bash
# Run a benchmark across fitting methods
peakfit benchmark spectrum.ft2 peaks.list --iterations 3

# Profile fit results (e.g., profile likelihood or MCMC)
peakfit analyze profile Fits/ --param x0 --points 20 --plot
peakfit analyze mcmc Fits/ --chains 64 --samples 2000
```

###

### Custom Worker Configuration (optional)

For fine-grained control over worker count using the CLI:

```bash
peakfit fit spectrum.ft2 peaks.list --workers 8 --refine 2
```

## Performance Tips

1. **Profile First**: Use `peakfit benchmark` to understand your data's characteristics

2. **Match Method to Data**:

   - Few clusters (< 5): Use the default `leastsq` optimizer (`--optimizer leastsq`)
   - Many clusters (> 10): Automatic parallelism may provide benefits; consider `--workers N` to bound resource usage
   - Medium clusters: Benchmark both

3. **Memory Considerations**:

   - Parallel fitting uses more memory (one process per worker)
   - For memory-constrained systems, reduce `--workers` and use the default `leastsq` optimizer

4. **Refinement Iterations**:
   - More iterations = more benefit from optimization
   - Consider `--refine 2` or `--refine 3` for complex data

## Troubleshooting

### Parallel fitting slower than expected

1. Check the number of clusters:

   ```bash
   peakfit benchmark spectrum.ft2 peaks.list
   ```

2. Multiprocessing has overhead; for few clusters, use the default `leastsq` optimizer (`--optimizer leastsq`)

3. Reduce workers if memory is limited:
   ```bash
   peakfit fit spectrum.ft2 peaks.list --workers 4
   ```

### Out of memory errors

1. Reduce parallel workers
2. Use sequential least-squares optimizer (default `--optimizer leastsq`)
3. Process data in smaller batches

## API Reference

### Command Line

```bash
# System info and optimization status
peakfit info [--benchmark]

# Performance benchmark
peakfit benchmark SPECTRUM PEAKLIST [--iterations N]

# Fit with optimizations
peakfit fit SPECTRUM PEAKLIST [--optimizer <leastsq|basin-hopping|differential-evolution>] [--workers N]
```

### Note: CLI-first usage

PeakFit is intended to be used as a command-line application via the
`peakfit` command. Internal Python functions (e.g., modules under
`peakfit.*`) are not a public API and may change without notice. For
automation or scripting, prefer invoking the CLI from your script or
workflow manager (for example, using `subprocess.run`).

```

## Version History

- **v2025.12.0**: NumPy-only implementation
- **v0.2.0**: Added fast scipy optimization, parallel fitting, profiling
```
