# MCMC Workflow

This document describes the current MCMC workflow. It is a contributor note, not a
permanent architecture constraint.

## Current Responsibilities

- Select clusters and parameters for uncertainty analysis.
- Run MCMC sampling from previous fit results.
- Save chain data and structured diagnostics.
- Support `peakfit mcmc` and MCMC plotting workflows.

## Current Flow

1. The CLI resolves the results directory and MCMC options.
2. `run_mcmc_analysis()` loads the fit state and selects target clusters.
3. Sampling runs with progress callbacks for the terminal UI.
4. Summaries, chain files, and plots are written or rendered by the caller.

## Next Architecture Pass

Keep MCMC as direct functions plus result formatters unless there is a concrete
need for shared infrastructure with fitting.
