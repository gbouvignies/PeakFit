# Example 4: Post-Fit MCMC Uncertainty Analysis

This example shows how to run `peakfit mcmc` as a first-class CLI step after fitting.

## Workflow

1. Run `peakfit fit` to create a results directory.
2. Run `peakfit mcmc <results_dir>` to estimate uncertainties from the saved fit state.
3. Use `peakfit plot spectrum` for visual fit inspection.

## Quick Start

```bash
bash run.sh
```

The script performs a fit and then runs MCMC on a small peak subset to keep runtime manageable.

## Manual Commands

Fit first:

```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --output Fits
```

Resolve latest run:

```bash
LATEST_RUN="$(find Fits -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
```

Run MCMC:

```bash
peakfit mcmc "$LATEST_RUN" --peaks 2N-H --walkers 32 --steps 1000
```

Review the fit visually:

```bash
peakfit plot spectrum --spectrum data/pseudo3d.ft2 --results "$LATEST_RUN"
```

Generate diagnostics PDF from saved chains:

```bash
peakfit plot mcmc "$LATEST_RUN" --output "$LATEST_RUN/mcmc_diagnostics.pdf"
```

## Notes

- `peakfit mcmc` reads `metadata/fitting_state.pkl` from the fit output.
- Target selection is cluster-based: a requested peak triggers analysis of its full overlap cluster.
- MCMC diagnostics are printed in the terminal, and chains are saved under `chains/`.
