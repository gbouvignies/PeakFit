# Optimization Strategies

PeakFit exposes the optimizer choice via the CLI `--optimizer` flag. The current
pipeline supports the following strategies:

## Available Strategies

- `varpro` (default)
  - Variable projection with analytical Jacobian and `scipy.optimize.least_squares`.
  - Fast and robust for most datasets.

- `basin_hopping`
  - Global search warm‑start followed by local refinement.
  - Useful when initial guesses are poor or peaks overlap heavily.

- `differential_evolution`
  - Global optimizer for difficult landscapes.
  - Significantly slower; use when other strategies fail.

## CLI Examples

```bash
peakfit fit spectrum.ft2 peaks.list --optimizer varpro
peakfit fit spectrum.ft2 peaks.list --optimizer basin_hopping
peakfit fit spectrum.ft2 peaks.list --optimizer differential_evolution
```

## Choosing a Strategy

- Start with `varpro` for speed and stability.
- Switch to `basin_hopping` when fits diverge or local minima are common.
- Use `differential_evolution` only as a last resort for extremely difficult cases.

## Notes

- MCMC is **not** a fitting optimizer in the CLI workflow. Uncertainty analysis is
  performed via `peakfit mcmc` (see [MCMC_DIAGNOSTICS.md](MCMC_DIAGNOSTICS.md)).
