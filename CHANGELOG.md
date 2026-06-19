# Changelog

Notable user-facing and architecture changes are recorded here.

## Unreleased

### Changed

- Simplified the public Python package surface; import concrete modules instead of package
  facades.
- Simplified fit output generation around the canonical files documented in
  `docs/output_system.md`.
- Renamed fit orchestration concepts from service-oriented names to fit-run terminology.
- Made the CLI summaries more concise and consistent across commands.
- Simplified MCMC progress output to show observed progress and acceptance only.
- Simplified optimizer configuration to the two supported fit paths: `varpro`
  and `basin_hopping`.
- Renamed internal optimizer configuration and documentation from "strategy"
  terminology to direct optimizer terminology.
- Renamed the basin-hopping implementation module from generic global
  optimization terminology to direct basin-hopping terminology.
- Moved fit-output result models out of the numerical engine and into the fit
  output boundary.
- Renamed internal multi-step fitting modules and docs from protocol terminology
  to fit-step terminology.
- Replaced the `FitPipeline` class wrapper with direct fit pipeline functions.
- Merged the fit-pipeline architecture note into the main architecture guide.
- Simplified lineshape registration to the active shape-class registry and moved
  lineshape context helpers into the concrete utilities module.
- Simplified fit-step orchestration, optimizer execution, and peak-list reader
  dispatch to direct functions instead of wrapper objects or local registries.
- Simplified plotting around profile plots generated from the canonical
  `tables/intensities.csv` output.
- Restored CEST and CPMG profile plotting with deterministic error propagation
  instead of random bootstrap-derived error bars.
- Renamed plot generation code from service terminology to direct output
  terminology.
- Merged duplicate output architecture notes into the main output-system
  documentation.
- Consolidated user, developer, MCMC, and GitHub template documentation around
  the current CLI, canonical outputs, `prek`, and simplified architecture.

### Removed

- Removed legacy and duplicate output concepts, including model-comparison output,
  legacy parameter-name fallbacks, placeholder/manifest output paths, and duplicate
  dataclass serialization helpers.
- Removed unused import hubs and compatibility modules.
- Removed unused plot backend interface abstractions.
- Removed the stale `differential_evolution` optimizer path and redundant
  `[fitting].strategy_name` config key.
- Removed unused result helpers/constants and the obsolete module-based
  lineshape protocol.
- Removed unused fit-step result wrappers, optimizer wrapper classes, and
  the legacy serialized-state `peaks` fallback.
- Removed stale standalone MCMC workflow and duplicate terminal/typing
  development notes after merging current guidance into `docs/development.md`.

### Compatibility

- This development cycle intentionally breaks several internal Python imports as part of
  the architecture cleanup. CLI workflows and scientific behavior remain covered by tests.
- `peakfit fit --optimizer differential_evolution` and configs containing
  `[fitting].strategy_name` are no longer accepted; use the default `varpro` or
  `--optimizer basin_hopping`.
