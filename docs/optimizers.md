# Optimizers And Runtime Tuning

PeakFit exposes the optimizer choice via the CLI `--optimizer` flag. The current
pipeline supports these optimizers:

## Available Optimizers

- `varpro` (default)
  - Variable projection with analytical Jacobian and `scipy.optimize.least_squares`.
  - Fast and robust for most datasets.

- `basin_hopping`
  - Global search warm‑start followed by local refinement.
  - Useful when initial guesses are poor or peaks overlap heavily.

## CLI Examples

```bash
peakfit fit spectrum.ft2 peaks.list --optimizer varpro
peakfit fit spectrum.ft2 peaks.list --optimizer basin_hopping
```

## Choosing An Optimizer

- Start with `varpro` for speed and stability.
- Switch to `basin_hopping` when fits diverge or local minima are common.

## Runtime Controls

Refinement iterations improve cross-talk correction but add time per cluster:

- `--refine 1` is the fastest useful pass for many datasets.
- `--refine 2` or `--refine 3` can improve dense clusters at higher runtime cost.

Cluster fitting is parallelizable:

```bash
peakfit fit spectrum.ft2 peaks.list --workers -1
```

Use `--workers -1` to use all available CPUs.

## Notes

- MCMC is **not** a fitting optimizer in the CLI workflow. Uncertainty analysis is
  performed via `peakfit mcmc` (see [mcmc.md](mcmc.md)).
