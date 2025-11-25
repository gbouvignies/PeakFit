# PeakFit Dependency Snapshot

_Updated: 2025-11-25 (post-core restructure)_

## How this snapshot was generated

Run `uv run python tools/dependency_snapshot.py` to rebuild the intra-package import summary. The helper performs a lightweight AST scan (no runtime imports) and aggregates dependencies at both module and package levels.

## Package-level import fan-out

| Source package | Target package | Import edges |
| -------------- | -------------- | ------------ |
| analysis       | core           | 9 |
| cli            | core           | 17 |
| cli            | data           | 6 |
| cli            | io             | 2 |
| cli            | plotting       | 1 |
| cli            | ui             | 6 |
| data           | core           | 12 |
| data           | ui             | 1 |
| io             | core           | 6 |
| io             | ui             | 1 |
| plotting       | core           | 1 |
| plotting       | data           | 1 |
| plotting       | ui             | 2 |

_Observations_
- `cli` still touches nearly everything, but the fan-out now terminates at the consolidated `core` namespace. Once fit/analyze pipelines live in `services`, these 17 edges should shrink dramatically.
- `data` and `io` continue to import `ui`, so they cannot move under `core` yet. Removing console dependencies from clustering/noise/output remains a priority.
- `analysis` talks only to `core`, matching expectations for advanced algorithms.

## Modules with the widest dependency surface

| Module | Number of peakfit modules imported |
| ------ | ---------------------------------- |
| `peakfit.cli.app` | 14 |
| `peakfit.cli.fit_command` | 13 |
| `peakfit.analysis.benchmarks` | 8 |
| `peakfit.core.fitting.advanced` | 7 |
| `peakfit.io.output` | 6 |
| `peakfit.cli.analyze_command` | 6 |

These files align with the refactor targets identified earlier (monolithic CLIs, advanced fitting orchestrators, and legacy benchmarking utilities). Each will be decomposed or retired as part of the service/extraction work.

## Layering guidance derived from the snapshot

1. **Core math & domain** now physically live under `peakfit.core`. Next step: strip `ui` calls from `data`/`io` so they can depend on a lightweight reporter and stop bypassing the core boundary.
2. **Service layer (to be created)** remains the next milestone: migrate `run_fit`/analysis orchestration into `peakfit.services.*` so adapters depend on a single entry point instead of the entire `core` surface.
3. **Adapters** (CLI, plotting, GUI) still import `core` directly; once services exist, they should reference only service classes plus UI helpers.

This document serves as the baseline for measuring dependency improvements as the refactor progresses.
