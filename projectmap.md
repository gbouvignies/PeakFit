# PeakFit Project Map

## Purpose
PeakFit performs robust lineshape fitting for pseudo‑ND NMR spectra, usable as a CLI tool
and a minimal computation library.

## Core Domains
- **Spectra & Peaks**: Data models and validation of spectral inputs.
- **Clustering**: Grouping overlapping peaks for joint fitting.
- **Lineshapes & Optimization**: Gaussian/Lorentzian/PVoigt/NoApod/SP1/SP2 models.
- **Fitting Pipeline**: Iterative refinement workflow and parameter synchronization.
- **Output System**: Result serialization (JSON/CSV/Markdown/PDF) and plots.

## High‑Level Flow
1. **CLI** parses arguments and routes to slices.
2. **Fit slice** validates, loads data, runs the pipeline, and writes outputs.
3. **Engine** performs computation and produces result models.
4. **Plot slice** renders plots and interactive spectrum viewer.
5. **MCMC slice** performs uncertainty analysis and diagnostics.

## Where Changes Should Go
- **Algorithms or domain rules** → `engine`.
- **Workflow orchestration** → `fit` or `mcmc`.
- **File formats / parsing / serialization** → `io`.
- **User interaction / CLI / Rich output** → `cli`.
- **Plots and spectrum viewer** → `plot`.

## How to Run Gates
- Tests: `uv run pytest`
- Lint: `uv run ruff check .`
- Types: `uv run ty check`

## References
- [docs/architecture/overview.md](docs/architecture/overview.md)
- [docs/architecture/fit-pipeline.md](docs/architecture/fit-pipeline.md)
- [docs/architecture/output_architecture.md](docs/architecture/output_architecture.md)
