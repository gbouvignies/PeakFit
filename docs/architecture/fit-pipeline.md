# Fit Pipeline Architecture

The fit pipeline orchestrates the optimization workflow and lives in the `fit` slice. It is
UI‑agnostic and depends on the pure `engine` for computation.

## Responsibilities

1. **Validation gating:** Ensure inputs are valid before any fitting begins.
2. **Iteration control:** Manage refinement loops and protocol steps.
3. **Optimization dispatch:** Execute per‑cluster fits via engine algorithms.
4. **Parameter synchronization:** Update global parameters from per‑cluster results.
5. **Result aggregation:** Build result objects for downstream writing.

## Flow

1. **Initialize**: The CLI builds config and calls `peakfit.fit.run`.
2. **Validate**: `fit` validates spectrum + peak list and fails fast on errors.
3. **Load**: `fit` loads spectra, peaks, noise, and clusters.
4. **Optimize**:
   - Build per‑cluster parameters
   - Apply constraints and step rules
   - Call engine optimizers for each cluster
5. **Aggregate**: Assemble `FitResult` objects and update global state.
6. **Write**: Serialize outputs using `peakfit.io` writers.

## Decoupling Rules

- **No UI imports in engine**: Rich/Qt stay in CLI/plot.
- **No file I/O in engine**: pipeline returns data; writing is in `fit`.
- **No cross‑slice imports**: `fit` does not import `plot` or `mcmc`.
