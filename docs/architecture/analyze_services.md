# MCMC Architecture

This document describes the MCMC workflow in the vertical‑slice architecture.

## Responsibilities

```
CLI (Typer + Rich)
   → peakfit.mcmc (workflow + diagnostics)
      → peakfit.engine (computation)
      → peakfit.io (load/save chains + summaries)
```

### `peakfit.mcmc`

- Filters clusters and parameters for analysis.
- Runs MCMC sampling via engine algorithms.
- Produces diagnostics and structured summaries.
- Exposes a first‑class CLI command (`peakfit mcmc`).

### Decoupling Rules

- MCMC does not import `fit` or `plot`.
- Engine stays pure (no I/O, no Rich/Qt).
- CLI handles all formatting and user messaging.

## Adapter Interaction Pattern

1. CLI resolves input paths and options.
2. MCMC slice loads state/chains using `peakfit.io` helpers.
3. Engine computes diagnostics and parameter updates.
4. CLI renders tables or invokes plot helpers.
