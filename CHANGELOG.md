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
- Renamed internal optimizer configuration and documentation from "strategy"
  terminology to direct optimizer terminology.
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
- GUI job failures log their full traceback, including those raised in worker
  processes, and dialogs show the failure's own message rather than `Type: msg`.
  Canceling Open or Relink no longer reports "Save canceled", and closing during a
  save or open explains why it waits.
- Reopening a project checks each run's adoption by its post-fit positions instead
  of comparing saved warning text, so rewording a warning no longer blocks reopening.
- Lineshape models no longer substitute defaults for missing parameters (`lw`, `j`,
  `eta`, phase); a missing model parameter raises. A phase that is not fitted is
  absent from the model (no phase correction).
- CLI and GUI "At bound" warnings use one rule: a varied parameter within 10⁻⁶ of
  its bound interval from a bound. The CLI used 10⁻⁶ × (1 + |value|) and also flagged
  fixed parameters. VarPro's uncertainty withholding at active bounds is unchanged.
- A fit run now works on its own copy of the clusters, starting with no neighbor
  corrections. Before, a second run on the same prepared clusters (for example
  from Python) started from the corrections the first run left on them, while
  recording its first pass as uncorrected, and gave different results. Parameter
  sets also no longer share parameter objects after `Parameters.update`.
- Engine boundary: `CorrectionSnapshot`, `PipelineCompletion` and `RunSummary` moved
  into `engine.scientific`, so `engine` and `io` no longer reference `fit` even for
  type checking, and the import contracts now count type-only imports. Removed
  `FitParameters`, the unused `peaks` argument of `run_pipeline[_iter]`, the
  never-matched amplitude-parameter paths (`ParameterMap.from_peaks`, VarPro's
  amplitude-parameter update, `calculate_amplitude_covariance`) and the duplicate
  nonlinear-parameter filters (`Cluster.parameter_names` replaces them). Pipeline
  progress is reported as typed `PassStart`/`CorrectionUpdate` events instead of
  Rich-markup tuples, and `run_fit` no longer changes the `peakfit` logger level.
- `CompletedFit` is the one carrier of a completed fit. `PipelineResult` keeps only
  it and the continuation state; `FitRun` keeps it, the spectra and the output
  directory; `find_review_clusters` takes the `FinalFitOutcome`;
  `write_fit_run_outputs` takes the spectra from the run.
  `FinalFitOutcome.by_cluster_id` is derived from its clusters, and
  `FinalModelSnapshot` no longer copies the outcome's noise, revision and values.
- The CLI and GUI prepare a fit through one function, `prepare_clusters` (noise,
  clustering threshold, clusters). A clustering threshold that is not positive and
  finite, for example `--contour -5`, is now rejected before fitting by the CLI too.
- Consolidated the lineshape models: the twelve shape classes live in
  `engine.lineshapes.models` with one parameter-spec helper; each lineshape package
  keeps only its kernel. Removed the test-only module functions (`function`,
  `function_doublet`, `bounds`, `PARAM_NAMES`) and the unused `LineshapeContext`,
  `require_grid` and `get_apodization_state`. Numerically bit-identical.
- GUI fits and analysis operations share one child-process runner
  (`gui.child_process.run_child`) with one handshake, serializer, error channel and
  reaping; a child that exits without a result is reported with its exit code.
  Historical runs load on the view's background worker instead of a separate
  process.
- The editor renders fitting state when it changes instead of refreshing every
  200 ms: `DatasetFitting.state_changed` fires only on job and result transitions,
  and a clock ticks only while a fit is busy, to show elapsed time. One busy state
  locks and releases every scientific edit. Fixed the Edit menu's Add peak staying
  disabled after the first fit, and the pin button staying disabled after the lens
  closed.
- "Modified" has one meaning in the toolbar, the window title and the close prompt:
  the analysis differs from the last saved or opened one, or a draft is pending. The
  toolbar tooltip names what differs. **Behaviour change:** exporting peaks (CSV) or
  fit results no longer clears the toolbar's Modified, since neither saves the
  analysis (the title and close prompt already said so). Saving and opening go
  through `EditorWorkspace.capture()`/`restore()` instead of writing workspace
  internals. Saved runs no longer record the process-local `spectrum_id` in
  `review.json`; readers already ignored it, so projects open both ways unchanged.
- `peakfit mcmc` now samples a cluster's fitted phase. It previously dropped the
  cluster-shared phase parameter and silently evaluated the likelihood at zero phase,
  so runs fitted with `--phx`/`--phy` sampled a different model than the fit.
- Fixed a crash when a view closed while its background lane was delivering a
  result: closing a lane now waits for a delivery already emitting.

### Removed

- Removed the basin-hopping optimizer, `peakfit fit --optimizer`, the
  `[fitting].optimizer_seed` key and Example 03. VarPro is the only optimizer; its
  uncertainty withholding and ½Σr² cost are the only ones PeakFit reports.
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
- `peakfit fit --optimizer` and configs containing `[fitting].strategy_name` or
  `[fitting].optimizer_seed` are no longer accepted; every fit uses VarPro.
- Configs containing `[auto_peak]`, `fix_positions` or `fit_j_coupling` are rejected;
  delete those keys.
- Scientific archives move to 6.0.0, fit JSON to 5.0.0 and projects to 9.0.0;
  earlier versions are rejected. Refit results and recreate projects written
  before this change.
- A non-positive or non-finite `noise_level` is rejected; it was silently replaced
  by an estimate when set through `--noise` with a config file. A spectrum whose
  estimated noise is not positive and finite asks for `noise_level` explicitly.
