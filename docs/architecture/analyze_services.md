# MCMC Workflow Notes

This document describes the current MCMC workflow. It is a contributor note, not a
permanent architecture constraint. Revisit it after the next architecture pass.

## Current Responsibilities

- Select clusters and parameters for uncertainty analysis.
- Run MCMC sampling from previous fit results.
- Save chain data and structured diagnostics.
- Support `peakfit mcmc` and MCMC plotting workflows.

## Current Flow

1. The CLI resolves the results directory and MCMC options.
2. The workflow loads fit state or chain data.
3. Sampling or diagnostic calculations run.
4. Summaries, chain files, and plots are written or rendered.

## Next Architecture Pass

Check whether MCMC needs a separate slice or can share simpler result-loading,
diagnostic, and output utilities with fitting without reintroducing hidden coupling.
