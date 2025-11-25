# Analyze Services Architecture

This document explains how the `peakfit.services.analyze` package orchestrates
analysis workflows (state loading, MCMC, and profile likelihood) and how these
services interact with the rest of the layered architecture.

## Layered Responsibilities

```
┌────────────────────┐  CLI / UI adapters (`peakfit.cli.*`)
│ peakfit.ui / cli   │  * Parses CLI args
│                    │  * Formats console/table output
└────────┬───────────┘
         │
┌────────▼───────────┐  Services (`peakfit.services.analyze.*`)
│ Analyze services   │  * Enforce orchestration + validation
│                    │  * Never touch console or files directly
└────────┬───────────┘
         │
┌────────▼───────────┐  Core domain + algorithms (`peakfit.core.*`)
│ Domain / core      │  * Pure computation
│                    │  * No infrastructure assumptions
└────────────────────┘
```

The CLI modules now depend only on the service layer. The services adapt domain
objects into UI-ready structures while keeping core modules free of adapter
concerns.

## Services Overview

### `FittingStateService`
- Location: `peakfit.services.analyze.state_service`
- Responsibilities:
  - Locate `.peakfit_state.pkl` inside a results directory.
  - Delegate serialization/deserialization to the infrastructure layer
    (`peakfit.infra.state.StateRepository`).
  - Provide a typed `LoadedFittingState` wrapper with both the materialized
    `FittingState` and the path used.
- Errors:
  - `StateFileMissingError`: captures missing directory/path metadata.
  - `StateLoadError`: wraps deserialization errors for better CLI messaging.

### `MCMCAnalysisService`
- Location: `peakfit.services.analyze.mcmc_service`
- Responsibilities:
  - Filter relevant clusters given optional peak selections.
  - Build per-cluster parameter sets via `peakfit.core.domain.peaks.create_params`.
  - Invoke the core `estimate_uncertainties_mcmc` function.
  - Update the global `Parameters` object with new standard errors so downstream
    tasks (output file rewrites, reports) remain consistent.
  - Return a structured `MCMCAnalysisResult`, which the CLI uses to render tables
    and persist diagnostics.
- Errors:
  - `PeaksNotFoundError`: raised when user-specified peak names cannot be matched
    to any clusters, preventing ambiguous CLI output.

### `ProfileLikelihoodService`
- Location: `peakfit.services.analyze.profile_service`
- Responsibilities:
  - Inspect the current `FittingState` and determine which varying parameters
    should be profiled (supporting wildcards / partial matches).
  - Locate the owning cluster for each parameter and hydrate the local parameter
    set (values + covariance stderr) before calling
    `compute_profile_likelihood` from the core layer.
  - Aggregate results into `ProfileParameterResult` and
    `ProfileLikelihoodAnalysisResult`, which include confidence thresholds and
    missing-parameter bookkeeping.
- Errors:
  - `NoVaryingParametersError`: surfaced when the loaded state has no free
    parameters to profile.
  - `ParameterMatchError`: raised when a user-supplied pattern does not match any
    parameter names; contains the full list of available parameters so the CLI
    can suggest alternatives.

## Adapter Interaction Pattern

Each CLI command follows the same high-level flow:

1. Call `load_fitting_state()` (wrapper around `FittingStateService.load`).
2. Invoke the appropriate analysis service (`MCMCAnalysisService.run` or
   `ProfileLikelihoodService.run`) with CLI-derived arguments.
3. Render the returned result objects using `rich` tables, save artifacts, or
   trigger plotting helpers. All file I/O and console operations stay inside the
   CLI layer, keeping services pure coordinators.

This separation enables future adapters (e.g., REST API, GUI) to reuse the same
services without duplicating domain logic.

## Extending the Analyze Layer

When adding a new analysis workflow (e.g., correlation diagnostics):

1. Implement a service in `peakfit.services.analyze.<new_service>` that depends on
   `peakfit.core` and infrastructure modules, but **never** on CLI/UI helpers.
2. Return dataclasses that encapsulate the data the adapter must present.
3. Surface meaningful exceptions specific to the workflow so adapters can
   translate them into user-facing guidance.
4. Update the CLI (or other adapters) to depend solely on the new service.
5. Add unit tests for the service to keep orchestration logic well-covered.

Following this pattern keeps the project aligned with the layered architecture
and simplifies future refactors.
