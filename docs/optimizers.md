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

`--refine N` runs exactly `N` optimizer passes. Cross-talk corrections update
only between passes, so it performs `N - 1` updates and never changes the
terminal correction state after the final pass. This is a behavior change from
the former `N + 1` pass schedule; increment an existing value by one to retain
its prior pass count.

More passes can improve cross-talk correction but add time per cluster:

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

## Overlap and identifiability

VarPro uses one rank-aware factorization for amplitudes, the model column space
and the Jacobian pseudoinverse. Ordinary groups retain triangular QR solves;
near singularity an SVD of the small R factor supplies a minimum-norm solution.
Residuals always equal `data - shapes.T @ amplitudes`, including at exact rank
loss. MCMC evaluates the same actual-model residual.

The analytical Jacobian differentiates this residual at constant rank. When
rank changes, the reduced objective need not be differentiable; an analytical
Jacobian there does not certify an identifiable optimum. Shared nonlinear
parameters sum each peak's derivative weighted by that peak's own signed
amplitude.

Close peaks can approach each other with large opposite signed amplitudes.
This can reduce the **actual** residual without yielding trustworthy individual
positions. Neither the optimizer success flag nor a smaller χ² proves the peak
decomposition is meaningful. PeakFit withholds unreliable covariance estimates
and reports an uncertainty warning; it imposes no minimum separation or
repulsion. See [the measured experimental comparison](validation/overlapping-peaks.md).

The current bare CLI defaults to **two passes** (`--refine 2`), while
`PeakFitConfig()` and the bundled example TOML specify one. State pass counts
explicitly when comparing historical runs. v0.10.5's `--refine 1` meant two
passes. Historical SP2 linewidth starts were `r2 = 20 s^-1`, equivalent to
`lw = 20/pi ≈ 6.366 Hz`; current SP2 defaults start at `lw = 20 Hz`.
