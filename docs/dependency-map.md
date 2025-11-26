# PeakFit Dependency Snapshot

_Updated: 2025-01-XX (post-streamlining refactoring)_

## Architecture Overview

The PeakFit codebase follows a layered architecture:

```
┌────────────────────────────────────────────────────┐
│                     CLI Layer                      │
│   cli/commands/{fit,analyze,plot,config,validate}  │
│   cli/_analyze_formatters.py                       │
├────────────────────────────────────────────────────┤
│                   Service Layer                    │
│   services/fit/{pipeline,fitting,loader,writer}    │
│   services/analyze/{formatters,state_service,...}  │
├────────────────────────────────────────────────────┤
│                     UI Layer                       │
│   ui/{console,logging,branding,messages,...}       │
│   ui/{tables,panels,progress,fit_display}          │
├────────────────────────────────────────────────────┤
│                   Core Layer                       │
│   core/domain, core/algorithms, core/fitting       │
│   core/shared/events, core/diagnostics             │
├────────────────────────────────────────────────────┤
│                Plotting & I/O Layers               │
│   plotting/{diagnostics,profiles,spectra}          │
│   io/{config,output,state,spectra,nmrpipe,...}     │
└────────────────────────────────────────────────────┘
```

## Package Structure

```
peakfit/
├── cli/           - Command-line interface
├── core/          - Domain logic and algorithms
├── services/      - Application services
├── ui/            - Terminal output formatting
├── plotting/      - Visualization
│   ├── diagnostics.py  - MCMC diagnostic plots
│   ├── profiles.py     - Profile visualization
│   └── spectra.py      - Interactive Qt viewer
└── io/            - File I/O
    ├── config.py       - TOML configuration
    ├── output.py       - Result file output
    └── state.py        - State persistence
```

Removed packages:
- `infra/` - merged into `io/`
- `analysis/` - moved to `tools/analysis/` (development utilities)
- `plotting/plots/` - orphan code deleted
- `plotting/common.py` - orphan CLI helpers deleted

## How this snapshot was generated

Run `uv run python tools/dependency_snapshot.py` to rebuild the intra-package import summary. The helper performs a lightweight AST scan (no runtime imports) and aggregates dependencies at both module and package levels.

## Recent Refactoring (Phase 1-5)

### Phase 1: Legacy File Cleanup
- Removed unused `data/` package
- Moved `core/diagnostics/plots.py` → `plotting/diagnostics.py`
- Created `plotting/profiles.py` from `cli/plot_command.py`

### Phase 2: CLI Delegation
- Created `services/analyze/formatters.py` with dataclasses for MCMC summaries
- Created `cli/_analyze_formatters.py` with Rich table formatting helpers
- Reduced `cli/analyze_command.py` from 816 to ~700 LOC

### Phase 3: UI Module Decomposition
Split `ui/style.py` into focused submodules:
- `ui/console.py` - Console instance and theme
- `ui/logging.py` - Logging setup functions
- `ui/branding.py` - Banner, version, run info display
- `ui/messages.py` - Status messages (success, error, warning, info, etc.)
- `ui/tables.py` - Table utilities
- `ui/panels.py` - Panel utilities
- `ui/progress.py` - Progress bar utilities
- `ui/fit_display.py` - Fit-specific display components

### Phase 4: Service Pipeline Simplification
Split `services/fit/pipeline.py` into:
- `services/fit/loader.py` - LoadedData dataclass, load_data(), load_spectrum(), estimate_noise()
- `services/fit/writer.py` - write_all_outputs(), write_simulated_spectra(), save_fitting_state()
- `services/fit/fitting.py` - fit_all_clusters() and fitting helpers
- Reduced `pipeline.py` from 655 to ~455 LOC (30% reduction)

## Import Rules (enforced by validate_imports.py)

The following import patterns are **forbidden**:

| Source Package | Cannot Import From |
|---------------|-------------------|
| `io`          | `ui`              |
| `core`        | `ui`, `cli`, `services` |

Run `python tools/validate_imports.py` to check for violations.

## Package-level import fan-out

| Source package | Target package | Notes |
| -------------- | -------------- | ----- |
| cli            | services       | Uses FitPipeline, FitService |
| cli            | ui             | Uses PeakFitUI, console |
| cli            | core           | Config, domain types |
| services       | core           | Domain logic, algorithms |
| services       | io             | Reading/writing spectra |
| services       | ui             | Progress/status display |
| plotting       | core           | Domain types |
| plotting       | ui             | Console output |
| io             | core           | Spectrum types |

## Layering guidance

1. **CLI Layer** imports services and ui, not core algorithms directly
2. **Service Layer** orchestrates core logic and provides clean interfaces
3. **UI Layer** provides presentation helpers, no business logic
4. **Core Layer** has no dependencies on ui, cli, or services
5. **Plotting Layer** depends on core domain types and ui for display

This document serves as the baseline for measuring dependency improvements as the refactor progresses.
