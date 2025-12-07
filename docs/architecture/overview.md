# PeakFit Architecture Overview

This document summarizes the layered architecture and the critical fitting path.

## Layering and allowed imports
- CLI → Services (only)
- Services → Core, IO, UI, Plotting
- Core → Domain, Algorithms, Lineshapes, Fitting; must NOT import UI or Services
- IO → Config/State/Writers; must NOT import UI
- Plotting/UI → Presentation only (never imported by Core)

ASCII map:

CLI → Services → Core → (Domain, Algorithms, Lineshapes, Fitting)
             ├────────→ IO (config/state/writers)
             └────────→ UI/Plotting (rendering only)

See also:
- Output system details: `docs/architecture/output_architecture.md`

## Critical path (fit workflow)
1. CLI: `peakfit/cli/commands/fit.py`
2. Services: `services/fit/pipeline.FitPipeline.run/_run_fit`
3. Core setup: domain IO (`core/domain/{spectrum,peaks_io}.py`), clustering/noise (`core/algorithms`), parameter creation
4. Optimization orchestration: `services/fit/fitting.fit_all_clusters` → `core/fitting/strategies.get_strategy`
5. Numerics: `core/fitting/computation.py`, `core/fitting/optimizer.py` (VarPro), `core/domain/peaks.py` (evaluate + derivatives)
6. Results & output: `core/results/*`, `io/writers/*`, UI events via `ui/*`

## Strategy options (available)
- Default: `varpro`
- Alternatives: `leastsq`, `basin_hopping`, `differential_evolution`, `mcmc`
- Protocol (optional): global warm‑start (BH/DE/MCMC) → local refine (VarPro)

## Parallelism & determinism
- Parallel cluster fitting via services; NumPy threads managed by `threadpoolctl`
- For global/MCMC strategies: set seeds and iteration/time budgets for reproducibility

## Exception & logging guidance (to be introduced)
- Small exception taxonomy in `core/shared/exceptions.py`
- Core uses `logging`; Services/UI handle rich rendering
