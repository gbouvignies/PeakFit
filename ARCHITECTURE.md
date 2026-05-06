# Architecture Boundaries (PeakFit)

This file defines responsibility boundaries. If a change violates these, it is wrong.

## Vertical Slice Model
- **CLI**: User interaction, argument parsing, Rich output.
- **Fit**: Orchestration of validation, data loading, pipeline, and output writing.
- **MCMC**: Uncertainty estimation workflows and diagnostics.
- **Plot**: Matplotlib plotting and the Qt-based spectrum viewer.
- **Engine**: Pure computation (domain models, algorithms, lineshapes, fitting math).
- **IO**: Parsing and serialization only (no fitting logic).
- **Shared**: Small cross-cutting utilities (exceptions, typing helpers).

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

## Ownership Map
- **Engine** (`src/peakfit/engine/`)
  - Domain models, algorithms, lineshapes, fitting math.
- **Fit** (`src/peakfit/fit/`)
  - Config, validation, data loading, pipeline orchestration, results building, output writing.
- **MCMC** (`src/peakfit/mcmc/`)
  - MCMC sampling, diagnostics, CLI-facing workflow APIs.
- **Plot** (`src/peakfit/plot/`)
  - Matplotlib plots and Qt spectrum viewer.
- **CLI** (`src/peakfit/cli/`)
  - Typer app, Rich output, pre-fit manifest.
- **IO** (`src/peakfit/io/`)
  - Readers/writers, file format translation only.
- **Shared** (`src/peakfit/shared/`)
  - Exceptions, typing utilities, small helpers.

## Design Constraints
- Engine is pure computation: **no I/O, Rich, Qt, or filesystem side effects**.
- IO parses/serializes only; it must not contain fitting logic.
- Fit always performs validation before any fitting.
- Plot spectrum uses Qt + Matplotlib (no optional install paths).
- Feature slices (**fit**, **mcmc**, **plot**) do not import each other directly.
- CLI contains presentation logic only; it does not own business rules.
