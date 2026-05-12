# Fit Pipeline Notes

This document describes the current fit workflow. It is context for contributors, not a
binding architecture rule. Revisit it after the next architecture pass.

## Current Responsibilities

- Validate spectra, peak lists, and configuration before expensive fitting work starts.
- Load spectra, peaks, noise estimates, and clusters.
- Manage refinement iterations and multi-step protocols.
- Dispatch per-cluster optimization through numerical code.
- Synchronize parameters across refinement steps.
- Assemble fit results and coordinate output writing.

## Current Flow

1. The CLI builds configuration and calls the fit workflow.
2. Inputs are validated and loaded.
3. Clusters and per-cluster parameters are prepared.
4. Optimizers run for each cluster.
5. Results are aggregated into models used by writers and downstream workflows.
6. Outputs are serialized.

## Next Architecture Pass

Check whether the fit workflow can be made more direct by merging thin wrappers, reducing
configuration translation, and moving output planning closer to the result data it writes.
