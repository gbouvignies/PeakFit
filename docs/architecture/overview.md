# PeakFit Architecture Overview

## Vertical Slice Model

PeakFit is organized by feature slices on top of a minimal computation engine.

```mermaid
graph TD
    CLI[CLI (Typer + Rich)] --> Fit
    CLI --> MCMC
    CLI --> Plot
    Fit --> Engine
    Fit --> IO
    Fit --> Shared
    MCMC --> Engine
    MCMC --> IO
    MCMC --> Shared
    Plot --> Engine
    Plot --> IO
    Plot --> Shared
    Engine --> Shared
    IO --> Shared
```

### Key Decisions

1. **Minimal engine**
   - `peakfit.engine` contains pure computation only.
   - No UI dependencies, no file I/O, no side effects.

2. **Feature slices own orchestration**
   - `fit`, `mcmc`, and `plot` orchestrate workflows and adapt data to user-facing outputs.
   - Slices do not import each other directly.

3. **Single install path**
   - Dependencies for Rich and Qt/Matplotlib are mandatory.
   - No optional extras are required for core features.

4. **Validation is mandatory**
   - Input validation runs at the beginning of every fit.
   - There is no standalone `check` CLI command.

## Critical Path (Fit Workflow)

1. **CLI** builds config and routes to `peakfit.fit`.
2. **Fit slice** validates inputs, loads spectra and peaks, and constructs clusters.
3. **Engine** executes optimization and produces fit results.
4. **Fit slice** writes outputs via `peakfit.io` writers.
