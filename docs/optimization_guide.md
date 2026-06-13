# PeakFit Optimization Guide

This guide covers practical performance tips and the current optimizer options.

## Overview

PeakFit uses NumPy-based lineshape evaluation and a VarPro (variable projection) solver
backed by `scipy.optimize.least_squares`. Basin-hopping is available for difficult
initialization cases but is slower.

## Quick Start

```bash
# Run the fit (default optimizer)
peakfit fit spectrum.ft2 peaks.list
```

## Optimization Options

Use `--optimizer` to select an optimizer:

- `varpro` (default): Fast and robust for most datasets
- `basin_hopping`: Global warm‑start followed by local refinement

Example:

```bash
peakfit fit spectrum.ft2 peaks.list --optimizer basin_hopping
```

## Refinement Iterations

Refinement iterations improve cross‑talk correction but add time per cluster:

- `--refine 1` (default): Good balance
- `--refine 2` or `--refine 3`: Better accuracy for dense clusters

## Parallel Workers

Cluster fitting is parallelizable:

```bash
peakfit fit spectrum.ft2 peaks.list --workers -1
```

Use `-1` to use all available CPUs.
