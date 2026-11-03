# Changelog

Notable user-facing and architecture changes are recorded here.

## Unreleased

### Changed

- Simplified the public Python package surface; import concrete modules instead of package
  facades.
- Simplified fit output generation around the canonical files documented in
  `docs/outputs.md`.
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
- Moved architecture and decisions documentation into `docs/` so project
  documentation has one canonical folder.
- Damaged, inconsistent or other-version scientific archives and projects now raise
  `ScientificIntegrityError` naming the file and the next step, instead of raw
  `AssertionError`, `KeyError` or `TypeError`. Integrity tolerances are unchanged.

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
- Retired experimental automatic peak picking: the `auto_pick` package, the
  `--auto-pick-step` window and the `[auto_peak]` configuration. `peakfit fit`
  requires a peak list.
- Removed `peakfit fit --fixed/--no-fixed` and `--jx/--no-jx` and the
  `[fitting].fix_positions` and `fit_j_coupling` keys, which only auto-pick read;
  in routine fits they did nothing. Fix positions with a fit step
  (`fix = ["*.*.cs"]`) and fit J with a doublet lineshape.
- Removed the legacy `peakfit plot spectrum` viewer and its fit-result overlay
  (`--results`); `peakfit gui` is the one graphical interface. The PDF plot
  commands (`intensity`, `cest`, `cpmg`, `mcmc`) are unchanged.
- Removed reading of results without `scientific.npz`: pickled fitting state
  (`metadata/fitting_state.pkl`), JSON-only result folders and the legacy MCMC
  path that guessed input-file locations and defaulted the noise to 1.0. Also
  removed the unused run README writer and the `git_commit` run metadata, which
  recorded the repository of the current directory rather than PeakFit's source.
  Runs now record no source revision; `software_version` is the package version,
  which does not change between development commits.
- Removed `[parameters].from_file`. It had read no values since fit JSON renamed
  its parameter keys, so fits silently started from the usual defaults; a config
  that still sets it is now rejected.
- Removed unused internal functions (unused `Parameters`, spectrum and peak helpers,
  J-coupling utilities, MCMC formatters and plot helpers), the legacy
  `engine.fitting.simulation` module and the `tomli-w` dependency. Python code
  importing them breaks; the CLI, GUI and outputs are unchanged.

### Compatibility

- This development cycle intentionally breaks several internal Python imports as part of
  the architecture cleanup. CLI workflows and scientific behavior remain covered by tests.
- `peakfit fit --optimizer differential_evolution` and configs containing
  `[fitting].strategy_name` are no longer accepted; use the default `varpro` or
  `--optimizer basin_hopping`.
- Configs containing `[auto_peak]`, `fix_positions` or `fit_j_coupling` are rejected;
  delete those keys.
- Scientific archives move to 5.0.0, fit JSON to 5.0.0 and projects to 7.0.0;
  earlier versions are rejected. Refit results and recreate projects written
  before this change.
- A non-positive or non-finite `noise_level` is rejected; it was silently replaced
  by an estimate when set through `--noise` with a config file. A spectrum whose
  estimated noise is not positive and finite asks for `noise_level` explicitly.
