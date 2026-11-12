# Decisions Log

Short records of non‑obvious design choices and trade‑offs.

## 2026‑01‑19: Introduced AI workflow guardrails
- Added AI contract, architecture boundaries, and project map to prevent misplacement and local hacks.
- Future behavior changes should update the smallest relevant documentation and tests.

## 2026-06-19: Keep documentation consolidated
- Prefer one concise current guide over several historical notes.
- Keep AI instructions in `AGENTS.md`, package boundaries in `docs/architecture.md`,
  and contributor workflow in `docs/development.md`.
- Delete or merge stale documents when refactors make them misleading.

## 2026-06-19: Make agentic development converge
- AI-assisted work should advance through small verified commits rather than
  long-lived planning documents.
- Durable guidance belongs in `AGENTS.md`, `docs/development.md`, executable
  checks, or focused user/developer guides.
- Experimental workflows should stay isolated and clearly labeled so they do not
  complicate the main fitting path.

## 2026-10-31: One set of agent instructions, no project-local agent tooling
- `AGENTS.md` is the only project instruction file; Claude Code reads it directly,
  so there is no `CLAUDE.md`.
- Agent skills (Matt Pocock's) are installed globally, not vendored in `.agents/`.
- Graphify was removed; the code, `GLOSSARY.md` and `docs/` are the navigation
  sources.
- The current focus is the scientifically rigorous desktop GUI, built on the
  completed scientific storage architecture.

## 2026-10-31: Track work in GitHub Issues
- GitHub Issues is the only issue tracker; agent skills read
  `docs/agents/issue-tracker.md`.
- Specifications and decisions stay in version-controlled Markdown under `docs/`.
- `.scratch/` was removed. Its two completed efforts (final fit outcome,
  point/series axis contract) remain in Git history:
  `git log --diff-filter=D --name-only -- .scratch`.
- `CONTEXT.md` was renamed `GLOSSARY.md`, the name the skills expect.

## 2026-10-31: `peakfit gui` is the only desktop entry point
- `peakfit gui` launches the v2 application (`gui.editor.window.launch`). The
  `peakfit-editor` script, `python -m peakfit.gui.editor` and the original viewer
  (`gui.app`, `gui.viewer`) were removed, with no alias.
- `peakfit gui PATH` opens an NMRPipe spectrum or a saved `.peakfit` analysis.
  `--peaks` now requires `--peak-axis-order {indirect-direct,direct-indirect}`; no
  axis order is guessed. Plane values keep `--z-values` / `-z`, matching
  `peakfit fit`. Companion options are rejected with a `.peakfit` analysis.
  Migration: add `--peak-axis-order indirect-direct` to reproduce the old
  `peakfit gui spectrum --peaks list` mapping.
- Local fit and Apply fitted positions were retired, as the v2 design specified:
  whole-dataset fitting with automatic adoption is the one refinement path. Earlier
  local-fit exports remain readable.

## 2026-10-31: Review workspace direction (GUI specification revision 2.1)
- Keep PySide6/Qt Widgets/PyQtGraph. The weaknesses found in a design review were
  information design and visual craft; invest in an owned presentation layer
  (tokens, proxy style, icons, plot chrome) rather than a framework change.
- Three columns: cluster navigator, spectrum, inspector. No lower analysis panel.
  (The Cluster lens chosen here was replaced by cluster focus on 2026-11-06.)
- Review separates numerical warnings, review measures and labeled heuristics; no
  composite quality score.
- Fit settings are an integrated setup with an exact clustering preview, always
  visible as a toolbar summary; no first-fit modal dialog.
- Reference-normalized profiles are display-only, with full-gradient uncertainty
  propagation and explicit unavailable cases.
- No storage change is authorized; four storage items are deferred (specification §14).

## 2026-10-31: Backward compatibility is not a requirement
- APIs, project formats, data structures and GUI architecture may change whenever
  that yields a cleaner design.
- No compatibility layers, migration mechanisms or legacy adapters without a
  compelling reason; obsolete code is removed rather than preserved.
- Breaking changes are stated explicitly. Numerical behavior still changes only for
  an explained scientific reason protected by tests.

## 2026-11-01: Automatic lineshape fallback is a caution, not a block
- When `auto` finds no recognized NMRPipe apodization on an axis, the engine uses
  its pseudo-Voigt default, as the CLI does. The GUI fit setup labels that axis
  and shows a visible caution, but Fit stays available.
- Blocking would make such datasets unfittable until a model is chosen explicitly;
  the scientist judges suitability.

## 2026-11-01: No normalized profile for clusters with unreliable uncertainties
- On the bundled dataset the degenerate pairs (|ρ| = 1.00) record amplitudes near
  ±6×10¹² with σ/|A| ≈ 0.014, like well-determined peaks. The σ_R/|R| limit
  therefore passes, and the ratio cancels the arbitrary scale, giving a plausible
  profile with small error bars for undetermined amplitudes.
- Reference normalization is withheld as unreliable, with the reason, for clusters
  carrying the engine's uncertainty warning; the raw profile, which shows the
  inflated amplitudes, remains. This is distinct from an undefined ratio (zero or
  non-finite reference, unavailable σ).
- Approved in review of #82 as conservative behavior; revisit when the engine
  provides more specific identifiability diagnostics.

## 2026-11-01: Position movement is not a review diagnostic
- A peak may start far from its true position; moving toward it is what the
  optimizer is for. The "position shift" highlight (|fitted − initial| over the
  fitted linewidth) is removed with its threshold. Initial and fitted positions
  remain visible on each peak page.
- Run-to-run changes no longer report positions that moved by more than σ: the
  recorded position σ (about 10⁻⁵ ppm on the bundled data) flagged 113 of 166 peaks
  after a routine threshold change. No replacement threshold is introduced; the
  largest position differences are listed as neutral information, in ppm and in
  fitted linewidths.
- Run-to-run comparison keeps numerical warnings, highlights and cluster membership,
  matched by stable peak ID; a warning that disappears because a cluster was
  regrouped is not described as resolved.

## 2026-11-02: Unexplained-signal regions are signed, diagonally connected and measured in samples
- Positive and negative excursions are labelled separately, so a dispersive residual
  next to a misplaced peak reads as two regions rather than one cancelling blob;
  diagonal neighbours are connected so a tilted ridge stays one region.
- Distance to the nearest working peak is Euclidean in native samples: the two axes
  have no common ppm scale, and samples are also the unit of the region size. The
  list shows the ppm location of each region's extreme sample.
- An Outdated run's list is its historical residual, labelled with the run; see
  "An Outdated run stays a labelled historical comparison" below.

## 2026-11-02: Focus keeps its cluster for peaks the run has not fitted
- A peak just added belongs to no recorded cluster. Following only the selected
  peak's cluster would leave focus at the moment of the correction, so a selected
  unfitted peak keeps focus on its cluster while it lies in that cluster's frame
  (the frame, not the viewport, since selecting scrolls the one view to the peak);
  elsewhere, focus moves to the cluster with its nearest member in native samples.
- After a refit the regrouping line compares the followed peak's previous and new
  clusters by stable ID. A newly added peak has no previous cluster, so the line
  says which peaks it is now fitted with instead of reporting the context cluster's
  members as having left.
- Undo back to a run's exact inputs still leaves that run Outdated, as before:
  returning values through Undo does not recertify a run (docs/architecture.md).

## 2026-11-02: An Outdated run stays a labelled historical comparison
- Scientists make several corrections before refitting. Withdrawing Overlay,
  Residual and the unexplained list at the first commit removed
  the residual that guides the next correction, so the Outdated run's immutable
  model stays on show, labelled “Run 5d7b4768 · historical” wherever it appears.
  Current/Outdated is unchanged and Undo never recertifies a run.
- It is shown only on the data the run recorded: spectral axes, source plane
  indices, plane values and the spectrum's identity, shape and source files must
  match the run's input snapshot. Otherwise it is withdrawn with the reason.
- Working peaks are compared with the run's adopted working peaks by stable ID: ◇
  marks a peak the model does not contain; × the model's position of a peak moved or
  deleted since, so a residual centred on the old position is read correctly.
- The unexplained list keeps the run's regions but measures distances to the
  current working peaks, recomputed after every edit and said in its note: the
  question while correcting is whether a working peak already covers a region.

## 2026-11-02: The largest recorded residual is a retained scalar of the published result
- The review measure "largest recorded residual" was computed from dense
  normalized residuals, which projects and default exports do not retain, so it
  vanished after Save and reopen. It is now `largest_recorded_residual` on
  `FinalAnalyticalResult`, computed once at finalization from the live recorded
  optimizer residuals and written by the single scientific writer at every
  evidence level (scientific archive 2.1.0), so CLI exports carry it too.
- Its sample count is the existing `statistics.n_observations`; no second count is
  stored. Dense residuals stay out of projects. The whole-spectrum residual is
  never used for it.
- Archives written before 2.1.0 read it as `None`; the GUI says "not retained by
  this project" instead of showing a dash or recomputing. No migration.

## 2026-11-02: A refit continues unchanged peaks from the active run
- Adopting fitted positions while restarting every linewidth at 20 Hz gave starts
  that matched no solution; on the bundled data the second fit collapsed the
  resolved pair 52N-H/102N-H (χ²ᵣ 1.53 → 2.32) and later fits kept it
  ([repeat-fit collapse](validation/repeat-fit-collapse.md)).
- Unchanged peaks now start from the run's complete fitted nonlinear state, by
  stable ID; added, moved and incompatible peaks start from defaults. An unedited
  refit reproduces every cluster (largest relative χ²ᵣ change 2×10⁻⁶).
- The exact starts stay in the existing `initial_parameters`; only their provenance
  (`continuation`) is new. A rename keeps a peak continuing, since it changes no
  model input. Coincident pairs from earlier runs continue coincident and warned;
  no separation, deletion, basin choice or classification change was introduced.

## 2026-11-02: Near-collinear fitted shapes are not adopted or continued
- On the bundled data three pairs (53N-H/147N-H, 36N-H/93N-H, 5N-H/76N-H) end
  with recorded shapes at 1 − |cos| ≤ 6.4×10⁻⁷, contributions cancelling ×320
  to ×4 000; every other pair has 1 − |cos| ≥ 0.33. A real opposite-sign peak
  0.5 linewidth away stays resolved (0.136, ×2.95); at 0.25 linewidth the fit
  collapses although two resonances are present.
- The criterion is near-collinearity alone, 1 − |cos| ≤ 10⁻⁴ (equal shapes to about
  1 % of a linewidth): a conservative operational threshold, not a physical
  boundary. It does not use the uncertainty warning, cancellation or optimality,
  and cannot tell an optimizer failure from an identifiability problem, so it
  treats both alike and never selects another solution.
- Both members keep their working positions and start the next fit from defaults;
  the fitted result is untouched. On the bundled data the first refit then
  reaches the better resolved fit of 5N-H/76N-H (χ² 18 378 → 17 947), while the
  other two pairs re-collapse in place; later refits reproduce.
- Reopening validates the recorded adoption, so runs recorded before this rule,
  which adopted such peaks, still open and export as recorded.

## 2026-11-03: The golden fit reference records every value
- The previous baseline held four numbers and compared chi-squared within 50 %,
  described as stochastic optimization. The fit is deterministic: repeated runs,
  single-threaded BLAS and serial workers reproduce `scientific.npz` bit for bit
  (apart from run identity and timings). The baseline had also drifted: its
  chi-squared, 1 829 681, differs from the current fit, 1 828 682.53, by 5×10⁻⁴.
- The reference now records the example's classifications, per-cluster
  chi-squared, all nonlinear parameters and all amplitudes with their standard
  errors, written from main at 8e0c197 with `tools/write_golden_reference.py`.
- Tolerances admit only rounding between platforms. Starting positions shifted by
  1e-9 ppm moved identifiable values by at most 1.5×10⁻⁶ standard errors
  (parameters) and 2.4×10⁻⁸ (amplitudes); the tolerance is 10⁻³ standard errors.
  Rescaling the data by 1 + 10⁻⁴ fails the comparison.
- The three clusters with the uncertainty warning (269, 360, 424) are compared by
  classification, warning and chi-squared only: the same 1e-9 ppm change moved
  their amplitudes by 4×10⁻⁴ (relative).

## 2026-11-03: Automatic peak picking is retired; earlier storage versions are rejected
- The experimental `auto_pick` workflow, its `--auto-pick-step` window and the
  `[auto_peak]` configuration are removed; `peakfit fit` requires a peak list.
  `--fixed`/`--jx` and `[fitting].fix_positions`/`fit_j_coupling` go too: only
  auto-pick read them, so in routine fits they did nothing. VarPro's shared-alias
  support, used only by auto-pick, is removed. Routine fits are bit-identical; the
  example's `scientific.npz` differs only by the three configuration keys it no
  longer records.
- Projects and archives record their configuration, so earlier ones would fail
  validation against the new model. Rather than ignore retired keys, the scientific
  archive moves to 3.0.0 and the project to 6.0.0, and only those versions are read.
  Earlier files are rejected with their version message; refit them from their
  spectrum and peak list. The schema-1 context branch and the "largest recorded
  residual not retained" path, reachable only from pre-2.1.0 archives, are removed.

## 2026-11-03: The legacy spectrum viewer is retired
- `peakfit plot spectrum` and its modules (`plot.spectra_viewer`, `plot.qt_core`,
  `plot.widgets`, `plot.reconstruction`) are removed, with no alias. `peakfit gui`
  is the single graphical interface; the PDF plot commands remain.
- Lost: overlaying a CLI fit result (`--results`) on the spectrum. The viewer
  recomputed amplitudes from the saved fitting state rather than reading the
  recorded result. Opening CLI results in `peakfit gui` is possible future work,
  not planned.
- `peakfit.plot` no longer imports Qt; an import-linter contract keeps it so.

## 2026-11-03: Only scientific archives are read
- Pickled fitting state (`io.state`), the JSON-only `ResultsLoader` fallback and its
  placeholder shapes, and the legacy MCMC branch are removed. `peakfit mcmc`, the
  plots and the GUI read only `scientific.npz`; nothing outside the GUI's trusted
  child-process pipes unpickles data. The legacy MCMC branch silently used a noise
  of 1.0 when none was recorded and searched nearby directories for inputs; MCMC now
  stops with an error instead.
- `git_commit` is removed from run metadata: it recorded the repository of the
  current directory, usually the user's data, not PeakFit. Runs now record no
  source revision: `software_version` is the package version, which stays the same
  across development commits, so it does not identify the executed source.
  Recording the exact source revision is separate work. The field's removal
  changes the recorded format, so the
  scientific archive moves to 4.0.0 and fit JSON to 5.0.0; earlier versions are
  rejected by version. The project manifest is unchanged at 6.0.0.
- Project-reader defaults for keys the 6.0.0 writer always records are removed, as
  are the older-format fixtures in `tests/data/storage-compat/`; version rejection
  is tested with synthetic archives and manifests.
- The example fit's outcome, samples and context are bit-identical before and after;
  only metadata changes.

## 2026-11-03: Previous-fit parameter files are removed
- `[parameters].from_file` is removed. It read fit JSON by keys the writer stopped
  emitting, so it had silently loaded nothing; it also matched parameters by name
  and depended on a JSON export rather than `scientific.npz`. The GUI already
  rejected it and continues fits by stable peak ID.
- `[parameters]` now rejects unknown keys, so a leftover `from_file` fails at load.
- Scientific archives and projects record the configuration and validate it on
  reopening; those written before this change record `from_file: null`. The
  scientific archive moves to 5.0.0 and the project to 7.0.0, so they are rejected
  by version before configuration validation. Fit JSON stays at 5.0.0: its schema
  keeps the configuration as an opaque mapping and nothing rebuilds a
  configuration from it.
- If CLI fits need previous-fit starts, read them from `scientific.npz`.

## 2026-11-03: Integrity failures raise one actionable error
- `ScientificIntegrityError(ValueError)` replaces production `np.testing.assert_allclose`
  checks in the archive reader, continuation and GUI model verification. `require_close`
  keeps numpy's rule (shapes match or one is scalar, NaNs equal) and every call keeps
  its tolerance; `tests/test_scientific_integrity.py` pins both.
- `read_scientific` and `open_analysis` turn malformed-file errors into that error,
  naming the file and what to do next and chained to the original. They catch only
  what malformed input raises: a damaged ZIP or float encoding, a missing field
  (`KeyError`), a wrongly typed value (`TypeError`) and the readers' own
  `ValueError` checks. Decoded JSON objects and arrays are type-checked before
  their methods or attributes are used, so `AttributeError`, `IndexError` and any
  other exception remain visible as defects.
- `prepare_noise_level` is the one validation point for fitting noise; it raises
  instead of replacing an invalid configured level by an estimate. Invariant checks
  on recorded outcomes stay where they are.
- The example fit's outcome, samples and context digests are identical before and
  after.

## 2026-11-03: GUI job failures are logged; adoption is checked structurally
- Worker processes send back the failure's type, the message to show and its
  formatted traceback (`gui.failures.child_failure`); the parent logs the traceback.
  In-process job boundaries log with `exc_info`. GUI errors go to the `peakfit.gui`
  logger with no handler of its own, so they reach the launching terminal.
- `ValueError` and `OSError` messages are shown as raised. Any other exception is a
  defect: its message asks for a report and points to the logged traceback.
- Canceled operations are named by what was canceled. Closing while an analysis is
  saving or opening explains why the window stays open.
- Run records no longer store `completion_warnings`. Reopening already rejects any
  post-fit position that disagrees with the adoption the outcome implies; comparing
  warning text added nothing but made rewording a warning break reopening. The
  project format moves to 8.0.0; 7.0.0 projects are rejected by version.

## 2026-11-03: Model parameters are never silently defaulted
- `ShapeBase.get_cluster_parameters` raises when a model parameter or a peak's
  lineshape on the axis is missing, instead of substituting `spec.default` (or 0 for
  a missing shape). The lineshape templates read `cs`, `lw`, `j` and `eta` without
  fallbacks (`lw = 25` or `20`, `j = 0`, `eta = 0.5`), which were unreachable.
- A phase not fitted on an axis has no parameter and no entry in the cluster
  parameters; the phased templates state the model explicitly as no phase
  correction. Every phase default was 0, so routine fits are bit-identical: the
  example fit's outcome, samples and context digests match main.
- `peakfit mcmc` selected parameters by peak-name prefix and dropped the
  cluster-shared `cluster_<id>.<axis>.phase`, so a phase-fitted run sampled the
  likelihood at zero phase. It now adds each shape's `parameter_names()`; this
  changes MCMC results for phase-fitted runs only, which were wrong.
- `FitParameters` still uses `spec.default` as index placeholders; it is removed in
  the engine-boundary work.

## 2026-11-03: One At bound rule for review (D8)
- `engine.diagnostics.bounds.at_bound` is the only review rule: a varied parameter
  is At bound when its distance from the nearest bound is at most 10⁻⁶ of its bound
  interval. It was the GUI rule. The CLI's live progress and review summary used
  10⁻⁶ × (1 + |value|), which tied the tolerance to the value's magnitude rather
  than the allowed range, and they flagged fixed parameters; both are removed with
  `Parameter.is_at_boundary`.
- Fixed parameters are never flagged. Without two finite bounds there is no
  interval to scale the tolerance, so one-sided or unbounded parameters are never
  flagged; a value outside its bounds or in a zero-width interval is.
- VarPro's active-bound test (absolute 10⁻⁷) decides whether symmetric standard
  errors are withheld. That is an uncertainty decision, not a display warning, and
  is unchanged; aligning it would need its own scientific validation.
- On the bundled example and the three example-05 configurations the flagged
  parameters are identical under the old CLI rule, the old GUI rule and the new
  rule (none, except `1011N-H.F3.lw` for `per_peak`). Fit digests and the golden
  regression are unchanged.

## 2026-11-03: Each fit run owns its correction state
- `run_pipeline_iter` shallow-copies its input clusters with zero corrections. Pass 1
  is then really revision 0, and correction updates never reach the caller's
  clusters. Before, `update_cluster_corrections` updated the caller's clusters in
  place and the next run snapshotted them as revision 0. On the bundled example with
  three passes, a second run on the same prepared clusters changed all 664
  parameters (χ² 1825695.05 → 1825690.58). The CLI and GUI were not affected only
  because they build fresh clusters or fit a pickled copy in a child process.
- `Parameters.update` copies parameters instead of storing and re-parenting the
  other set's objects, so a run's parameters no longer alias its optimizer results.
- Single runs are unchanged: the example digest, the golden regression and a
  three-pass example fit (values, errors, amplitudes and terminal corrections) are
  bit-identical to before.

## 2026-11-03: VarPro is the only optimizer (D7)
- Basin hopping is removed with its configuration (`BasinHoppingConfig`,
  `[fitting].optimizer_seed`), constants, `--optimizer`, tests and Example 03. Its
  standard errors came from a finite-difference Hessian inverted without rank or
  bound masking, and its `final_cost` was Σr² where VarPro records ½Σr², so the same
  published quantities meant different things. There was no scientific case for
  keeping and repairing a second, less reliable optimizer.
- Optimizer selection is removed rather than left with one choice: tasks carry a
  `VarProConfig`, `engine.fitting.varpro_fit.fit_varpro` fits a cluster, and the
  pipeline, `run_fit` and the CLI take no optimizer name.
- Optimizer provenance keeps every field. `optimizer_kind` is recorded as
  `"varpro"`; `iterations`, which only basin hopping filled, is recorded as
  unavailable because least squares reports evaluations, not iterations.
- Recorded configurations no longer contain `fitting.optimizer_seed`, which the
  configuration model forbids, so the scientific archive moves to 6.0.0 and the
  project to 9.0.0 and earlier versions are rejected by version. Fit JSON keeps its
  opaque configuration mapping and stays at 5.0.0.
- Routine fits are unchanged: the example's outcome, samples and context digests,
  the golden regression and a three-pass fit are bit-identical to main; archive
  metadata differs only by the version and the absent key.

## 2026-11-03: Engine boundary and parameter representation
- `CorrectionSnapshot` and `PipelineCompletion` (built only from engine types) move to
  `engine.scientific.completion`, and `RunSummary` (a projection of
  `FinalFitOutcome`) to `engine.scientific.summary`. `engine` and `io` no longer
  import `fit`, even for annotations, so import-linter now counts type-only imports.
  Five mutual type references between sibling modules (domain types, lineshape
  grid, editor mixins) are listed explicitly in the acyclic contract; restructuring
  them is not worth it here.
- `FitParameters` is removed: VarPro's map was the same `ParameterMap.from_names` of
  the varied names, and `FittingState.params` only fed a self-check. The serialized
  state `version` goes with it.
- Amplitudes are always solved, never parameters, so the amplitude-parameter paths
  never ran: `ParameterMap.from_peaks`, VarPro's amplitude-parameter update with its
  amplitude covariance, and MCMC's amplitude exclusion. MCMC chain labels, the one
  remaining amplitude name, are built with `ParameterId` and unchanged
  (`<peak>.F1.I<plane>`). `Cluster.parameter_names` selects a cluster's nonlinear
  model parameters from its shapes' declared names and replaces the two filters in
  evaluation and finalization and MCMC's selection.
- Pipeline progress uses typed `PassStart`/`CorrectionUpdate` events; the CLI owns
  their styling. `run_fit` no longer silences the `peakfit` logger, which nothing
  logs to during a fit.
- No numerical or format change: the example digest (outcome, samples, context),
  the golden regression and a three-pass fit match main, and archive metadata
  differs only by input paths.

## 2026-11-03: One carrier for a completed fit
- `CompletedFit` carries a completed fit. The records around it keep only what they
  add: `PipelineResult` the continuation state; `FitRun` the spectra and output
  directory (its duplicated `outcome`/`simulation_snapshot` fields, unread
  `continuation_state`, `success` and test-only `peak_identities` are removed);
  `DatasetFitResult` the reviewed inputs and run identity. `outcome` stays a
  property on `FitRun` and `DatasetFitResult`; the `final_outcome` and
  `simulation_snapshot` forwarders are removed.
- `find_review_clusters` reads only cluster outcomes, so it takes the
  `FinalFitOutcome`. Tests that compare CLI and GUI exports byte for byte call the
  shared writer through a test helper with the GUI's peak identities.
- `FinalFitOutcome.by_cluster_id` is derived in `__post_init__`, so it cannot
  disagree with `clusters`; it was never serialized.
- `FinalModelSnapshot` stored the outcome's noise, terminal revision and parameter
  values only to check that they matched the outcome; simulation reads them from the
  outcome. The copies and checks are removed. This is in memory only.
- No numerical or format change: the example digest, the golden regression and a
  three-pass fit match main; archive metadata differs only by input paths.

## 2026-11-03: One fit-preparation path
- `engine.algorithms.clustering.prepare_clusters` computes the noise
  (`prepare_noise_level`), the clustering threshold (the configured contour level,
  else `contour_factor` × noise, required positive and finite) and the clusters; the
  CLI's `load_data` and the GUI's `prepare_dataset_fit` both call it. The threshold
  formula was written twice and only the GUI checked it: `peakfit fit --contour -5`
  went on to fit with a meaningless threshold and is now rejected.
- Peak construction stays separate on purpose: the CLI reads the peak list, the GUI
  converts working peaks to engine peaks with stable identities.
- A test prepares the bundled example both ways and finds identical noise, threshold,
  cluster membership, sample coordinates and data. Fit digests, the golden regression
  and a three-pass fit are unchanged.

## 2026-11-03: Lineshape models in one module
- The six `model.py` modules duplicated each other (SP1/SP2 and Gaussian/Lorentzian
  differed only in names and kernels) and each re-declared the same parameter specs.
  The twelve classes now live in `engine.lineshapes.models`, built from the
  templates and `templates.parameter_specs(lw_default, *extras)`; the kernels stay
  in their packages, and NoApod's `make_state` moved next to its kernel.
- The module-level `function`/`function_doublet`/`bounds`/`PARAM_NAMES`/`NAME`
  were used only by tests. The equivalence tests now evaluate the classes, which
  also covers the no-derivative path used for display and simulation. Without
  them `LineshapeContext` carried only the grid, so `param_specs` take the grid and
  `LineshapeContext`, `require_grid` and `get_apodization_state` are removed.
- Bit-identical: every built-in shape's specs, values and derivatives (with and
  without derivatives) match byte for byte before and after; fit digests, the
  golden regression and a three-pass fit match main. No tolerance was relaxed.

## 2026-11-03: One child-process runner
- `gui.child_process.run_child` replaces the fit and archive bridges, which
  duplicated spawn, pipe, polling, liveness and reaping but differed in handshake,
  serializer and error transport. Every child now sends `ready`, receives the
  encoded payload, streams progress and ends with one result or `child_failure`
  record. Callers choose cancellation: fits terminate the child, archive
  operations forward a request the operation polls. Cancellation before the
  payload is sent always terminates, since the child cannot yet read a request.
- `FitProcessResult`'s startup, serialization and wall-time fields were never read;
  the fit process delivers the `DatasetFitResult` itself.
- One policy for historical runs: `StoredRun.load` runs on the requesting view's
  background worker, as fitting's parent-run lookup already did. Numerical review
  no longer spawns a process for it.

## 2026-11-03: Fitting state without polling
- The fitting panel refreshed on a 200 ms timer and after every workspace change,
  re-syncing the canvas and inspector each time. That masked missing notifications
  (a stage label written from the publication worker, the setup counting as seen)
  and re-synced the canvas while other work was in flight, which made the
  stale-delivery test flaky. `DatasetFitting` now emits
  `state_changed` only on transitions, and the workspace renders from its state;
  the timer runs only while busy, for elapsed time.
- The editing lock was computed in the fitting renderer, `_update_actions`, the lens
  update and per-call guards. `_update_actions` now applies one busy state to every
  edit control, which re-enables the Edit menu's Add peak after a fit. The pin
  button follows the lens directly instead of only ever being disabled.
- The fit setup counts as seen when the dataset page is shown (a Show event)
  rather than when a poll happened to observe it.
- The workspace creates its `DatasetFitting` before its widgets, so the `hasattr`
  construction guards and the `fit_busy` alias are gone.

## 2026-11-04: One Modified state; workspace capture and restore
- The toolbar's Modified tracked peaks changed since the last *export* (plus plane
  values), while the window title and close prompt tracked the analysis since the
  last *save or open*. After a peak or results export the toolbar said clean while
  closing still offered to save. The design spec already defines Modified as
  "unsaved analysis" and says exporting results does not mark it saved. Now the
  window computes one `unsaved()` from `EditorWorkspace.analysis_state()` against the
  state at the last save or open; the title, the toolbar (with what differs as its
  tooltip) and the close prompt all read it. `PeakEdits.dirty`/`mark_saved`,
  `plane_values_modified` and the unused `has_changes` are gone.
- Persistence and the window wrote about twenty workspace and fitting internals
  to save, open, rebind saved history, hand over scratch evidence, keep runs and
  import plane values. They now call `capture()`, `restore(analysis)`,
  `fitting.adopt_saved`, `fitting.hand_over_evidence`, `fitting.keep_run` and
  `set_plane_values`.
- `review.json` stored `id()` of the in-memory spectrum, meaningless in another
  process and already discarded by the reader (which binds the loaded spectrum).
  It is no longer written. Old projects still open (the reader never read it) and
  older readers accept new ones, so the project version stays 9.0.0.

## 2026-11-04: Close waits for an emitting delivery
- CI segfaulted twice in `ViewerComputation._deliver` while a test fixture deleted
  the workspace. A worker thread checked `_closed`, then emitted on the lane while
  the GUI thread closed it and deleted its owner; emitting on a QObject being
  destroyed is undefined. `suppress(RuntimeError)` only catches a wrapper already
  known to be deleted.
- The closed check and the emit now hold one lock, which `close()` takes to set
  `_closed`. After `close()` returns no emit is running or can start. The wait is
  bounded by posting one queued event; the lane never waits for the GUI thread.

## 2026-11-04: Keep a validated run when its recovery copy fails
- Publication validated the run, then spooled it to the session's recovery project;
  any spool error (disk full, permission, a failed child) failed the whole
  publication and discarded a result that had already passed validation. The spool
  protects against crashes; failing it should cost that protection, not the run.
- Now a spool error publishes the run with its journal entry. The run stays
  resident in history (as on datasets without a spectrum source), and
  `DatasetFitting.recovery_error` drives a "▲ Not in recovery copy" mark and
  tooltip until a later spool or a Save holds it. Validation failures and
  cancellation still publish nothing. Non-child errors are logged with their
  traceback; child errors were already logged by `run_child`.
- A run kept this way is resident in memory. Repeated spool failures accumulate
  resident runs until Save; this is accepted rather than evicting evidence that
  exists nowhere else.

## 2026-11-04: One model-plane cache and one normalized residual
- The canvas owned the only model-plane cache; the lens and the unexplained-signal
  search read it through a private attribute and discarded the planes they computed,
  so one (run, plane) could be projected up to three times. `fit_display.ModelPlanes`
  is now owned by the workspace and passed to the canvas and lens; every view stores
  what it computes, on the GUI thread, and reuses what is there. Bounds are unchanged
  (eight planes, 128 MiB); reads now refresh recency. Requests issued at the same
  moment by two views can still both compute; only sequential reuse is promised.
- `(observed − model) / noise` was written five times; `normalized_residual` is the
  one definition. Displayed model, residual, trace and unexplained-region arrays
  hash identically before and after on the synthetic two-cluster dataset.
- Moving presentation text out of `workspace.py` was considered and deferred: a
  large diff with no behaviour gain.

## 2026-11-04: MCMC formulation (D1–D4)
- MCMC samples the profile likelihood of the nonlinear parameters, justified as the
  integrated likelihood under the conditional Jeffreys amplitude prior, and draws
  amplitudes from N(â(θ), σ²(ΦΦᵀ)⁻¹) for each sample. The flat-amplitude-prior
  marginal is rejected: it depends on the shape normalization, its bias grows
  with the number of planes, and it rewards collinear shapes. At 50 planes and
  SNR 5 its 68% coverage was about 0.58, against 0.69 for the profile
  (`docs/mcmc-formulation.md` §3, `tools/validate_mcmc_formulation.py`).
- Degenerate and near-degenerate shapes: samples are never dropped (that would bias
  the posterior towards separated peaks). Per-peak amplitude intervals are withheld
  where a peak loads on a numerically weak direction, and flagged where member
  amplitudes are strongly correlated; summed amplitudes are reported instead.
  MCMC intervals are not required to be wider than the analytical ones.
- One emcee ensemble, with no R̂ across coupled walkers. A kept length of at least
  50 τ_max is a minimum, not proof of convergence. Stationarity, stuck-walker and
  label-switching checks make a run provisional instead of trustworthy-looking.
- Results are conditional on the recorded noise, frozen corrections and the
  lineshape model; χ²_red is reported, not used to rescale.
- The record stores θ chains and log-probabilities losslessly with complete
  summaries that need no source spectrum. Amplitude draws are regenerated from the
  parent run, the observations, the chain, a recorded `SeedSequence` and the
  software versions, rather than stored.
- The single-peak coverage study supports the formulation only; coverage on
  overlapping peaks (V6) is an acceptance criterion for the implementation.

## 2026-11-04: The sampled posterior (MCMC PR 22)
- `engine.scientific.sampling.sampling_problem(completed, cluster_id)` builds one
  cluster's posterior from the `CompletedFit`: the fit's geometry, observations and
  frozen corrections, and the cluster's final parameters (phases included). The
  likelihood uses `Cluster.evaluate` and `factorize_shapes`, the functions VarPro
  uses, without the Jacobian work. Building it recomputes χ² at θ̂ and requires the
  recorded value to a relative 10⁻¹⁰.
- `conditional_amplitudes` returns the profile log-likelihood and N(â, σ²G⁻¹)
  from the same factorization. A peak is not determined, and its draws are NaN,
  when either condition holds:
  - it loads above √ε on the null space (below the rank cutoff, or omitted by a
    reduced SVD when there are fewer points than peaks);
  - weak directions (s ≤ √ε·s₁, where the fit withholds its covariance) carry more
    than 10⁻⁴ of its conditional variance.

  A loading test alone is not enough: a direction with s/s₁ = 10⁻¹² and a loading of
  10⁻⁹ contributes about 10⁶ times the variance of a strong one. Draws omit weak
  directions, understating a kept amplitude's standard deviation by at most
  0.005%. No invariant prevents a cluster from having fewer points than peaks, so
  that case is handled rather than assumed away.
- The existing emcee driver now samples this problem. Its amplitude blobs are still
  â(θ), and walker start, diagnostics and storage are unchanged until PR 23–24. It
  no longer turns evaluation exceptions inside the bounds into −∞.

## 2026-11-04: Sampler driver and ensemble diagnostics (MCMC PR 23a)
- PR 23 is split: 23a is the driver and diagnostics, 23b the amplitude summaries
  (§5–§6) with V5 and V6.
- `engine.scientific.sampler.run_ensemble` takes a picklable log-posterior and one
  `SeedSequence`. The sequence spawns a sampler stream, which seeds the starts and
  emcee's moves, and a reserved amplitude-draw stream. A multi-cluster run spawns
  one sequence per cluster. Cancellation is checked and typed progress reported
  every step. `--workers` stays opt-in; pool workers receive the log-posterior once
  through the pool initializer, so steps don't pickle the problem.
- Starts are N(θ̂, (0.1·σ_fit)²), with 10⁻³ of the bound span when a standard error
  is missing. They are redrawn until strictly inside the bounds with a finite
  posterior, never clipped. Fewer than 2 × dimensions walkers is an error.
- Automatic burn-in is 5 τ_max of the whole chain, discarded once. The kept chain
  needs ≥ 50 τ_max (emcee's `tol=50`).
- Stationarity compares the medians of the kept chain's halves. The Monte Carlo
  error is estimated from the second half: estimated from the first half, it
  included the drift and let a chain falling from 10⁴ to 0 pass. On a stationary
  target it raised no false alarm in 40 seeded runs.
- Peaks with overlapping position bounds that change order along their
  best-separated axis are reported, and both peaks' per-peak intervals are
  withheld. On the bundled data, 53N-H and 147N-H (cluster 424) exchange in 59% of
  samples.
- The CLI shows medians and intervals at the precision of each posterior standard
  deviation. `peakfit plot mcmc` shows τ and ESS instead of R̂ (since PR 24, as
  recorded rather than re-checked).

## 2026-11-04: MCMC amplitude summaries (PR 23b)
- `engine.scientific.amplitudes.summarize_amplitudes` evaluates the conditional
  amplitude distribution at up to 4,000 kept samples (agreed cost bound), with one
  draw each from the run's amplitude-draw stream.
  - The samples are balanced: ⌊4,000 / walkers⌋ evenly spaced steps per walker.
    Spacing indices evenly through the flattened (steps, walkers) chain (16 × 2,000)
    had given walker 7 no sample and walker 15 one, and sampled each other walker in
    a single stretch of about 285 steps.
  - A sum's identifiability is checked at those evaluated samples, a single peak's
    at every kept sample (blobs), and the reports say which. That stream is
  child 1 of the run's `SeedSequence`, so draws reproduce from the recorded seed.
- Moments use the law of total variance, and quantiles come from the draws. Monte
  Carlo errors use the smaller of the sample count and the sampler's smallest ESS.
- A peak undetermined in any kept sample (from the per-sample identifiability
  blobs) is withheld with that fraction. Undetermined peaks group by co-occurrence,
  and their sum is reported if `ConditionalAmplitudes.determines` accepts it at
  every evaluated sample.
- Determined member amplitudes with |ρ| ≥ 0.9 in a plane are flagged and their
  group's sum reported. Their positions and linewidths stay visible, marked
  "not separately resolved": strong amplitude correlation alone does not show them
  unidentifiable. Withholding stays reserved for exchanges and demonstrated
  identifiability failures.
- V6 is a recorded study, `tools/validate_mcmc_coverage.py`, rather than a CI test:
  about seven minutes on eight cores. Separated cases cover at nominal rates. Near
  coalescence flags all 200 runs, with 199 provisional. The one run that passed
  explored a single ordering and undercovered; its intervals carry the caution
  (`docs/validation/mcmc-coverage.md`).
- `ConditionalAmplitudes` now keeps the singular vectors and judges any combination
  of amplitudes with the same two rules as a single peak. emcee squeezes a one-peak
  blob, so the peak axis is restored by count.

## 2026-11-04: MCMC run records (PR 24)
- Each `peakfit mcmc` run writes one immutable record,
  `<fit run>/mcmc/<UTC timestamp>-<random>.npz`, so names sort chronologically and a
  run never replaces another. `peakfit plot mcmc` reads the latest, or `--record`.
- The record reuses the scientific archive's encoding and reader limits
  (`io.mcmc_record`), stores each cluster's whole chain, log-probabilities and
  identifiability blobs losslessly, and the complete diagnosis and amplitude
  summaries. Results therefore read back without the parent or its spectrum.
- It binds to its parent by run ID and `scientific.npz` SHA-256, and records the
  spectrum SHA-256 for compact parents. Writing re-hashes the parent and refuses a
  record whose parent changed during sampling.
- Publication hard-links the flushed partial file to the record's name. Unlike a
  rename after an existence check, the link fails atomically if the name exists, so
  a concurrent writer's record is never replaced (tested with a forced collision).
- The reader is strict: exact field sets, typed scalars, array dtypes and shapes
  consistent with the recorded settings, and spawn keys matching the record's seed.
- Clean break: the HDF5 chain files (`chains/cluster_*_chains.h5`),
  `peakfit mcmc --save-chains/--no-save-chains`, `peakfit plot mcmc --burn-in`,
  `ResultsLoader.load_mcmc_chains` and the `h5py` dependency are removed. Existing
  chain files are no longer read; rerun `peakfit mcmc`. The plot report shows the
  recorded status and reasons instead of re-checking the chain, and thinning, which
  no run produced, is gone from the plots. The storage benchmarks that compare
  HDF5 take it from the optional `benchmarks` dependency group
  (`uv run --group benchmarks`).

## 2026-11-04: MCMC acceptance review, S1 and S2 reporting
- `peakfit plot mcmc` no longer presents an interval the terminal withholds. The
  marginal pages drew a median and 68% CI for every parameter, including those of a
  provisional run or exchanged peaks (seen on 53N-H/147N-H). Histograms are still
  drawn; a non-final parameter shows why instead, and every page of a provisional
  cluster says so. `PosteriorSample.unreported` is the one rule for terminal and PDF.
- Each MCMC posterior records its parent cluster's reduced χ², and the terminal and
  report page show it with the noise convention: MCMC uses the recorded σ, never
  rescaled; the fit's exported amplitude errors are scaled by √χ²ᵣ above 1. On the
  bundled 2N-H (χ²ᵣ 2.68), MCMC amplitude SDs were 1.05 × the unscaled conditional
  error but 0.64 × the exported `intensity_err`, which `mcmc.md` called "usually
  wider". MCMC record **1.1.0**: 1.0.0 records are refused; rerun `peakfit mcmc`.
- The uncertainty scaling itself is unchanged, pending a separate decision.

## 2026-11-04: σ-only uncertainties (ADR 0001)
- Amplitude errors are no longer multiplied by √max(1, χ²ᵣ). Nonlinear errors,
  amplitude errors and MCMC all use the supplied or estimated σ, and χ²ᵣ is reported
  as a goodness-of-fit diagnostic, now also in `intensities.csv` and the GUI profile
  note. The analytical amplitude covariance and errors are computed as before.
- `scaled_amplitude_standard_errors` and `amplitude_uncertainty_scale` are removed
  from the engine, archive, JSON and GUI. Scientific archive 7.0.0, fit JSON 6.0.0,
  analysis project 10.0.0; earlier files are refused. MCMC records stay 1.1.0.
- Validation (`docs/validation/sigma-uncertainties.md`):
  - On well-behaved synthetic data, the σ-only errors match the amplitude scatter
    to 6%, and MCMC to 1%.
  - On the CEST example, the change shrinks every error by exactly √χ²ᵣ (a median
    of 1/1.41). The successive-plane differences there contain genuine CEST
    profile variation and no replicates exist, so they calibrate neither
    convention. Correlated noise is a measured property of the spectrum, but its
    effect on these amplitude errors is untested.

## 2026-11-04: Processing-derived noise model (correlated noise, step 1)
- `engine.algorithms.noise_correlation` computes each axis's noise correlation from
  its recorded processing, and a cluster's separable R. `io.readers.spectrum.read_noise_model`
  reads it from an NMRPipe header (`docs/correlated-noise.md` §2.3).
  - Supported: `SP` or no apodization, first-point scale, zero fill and first-order
    phase.
  - Consistency check: |`APOD` − `TDSIZE`·S/`FTSIZE`| < 1.
  - Anything else gives "not modelled", with the reason. There is no partial or
    measured fallback.
- Not wired in: nothing is stored or published until the sandwich errors are
  validated (step 2) and published (step 3).

## 2026-11-04: Sandwich covariances (correlated noise, step 2)
- `linear_algebra.correlated_amplitude_covariance` gives the conditional amplitude
  sandwich G⁻¹ΦRΦᵀG⁻¹ from the existing factorization, which now keeps its left
  singular vectors.
- `varpro.identifiable_covariance(jacobian, correlation)` gives the nonlinear
  sandwich J⁺(R ⊗ I)J⁺ᵀ through the same SVD gate. Both are withheld exactly where
  today's covariances are, and R = I reproduces them to round-off.
- Validated with PeakFit's own fitting engine (`docs/validation/sandwich-errors.md`):
  - calibrated for isolated and overlapping 2D clusters, where σ-only understates
    the scatter 2.4–3.6-fold;
  - identical withholding in every case;
  - conditional amplitudes calibrated even near coalescence, where the nonlinear
    sandwich under-covers.
- Not wired in: nothing is published until step 3.

## 2026-11-05: Published sandwich errors (correlated noise, step 3; ADR 0002)
- **The noise model travels with σ.** The CLI and GUI read it from the spectrum
  header, `run_pipeline` requires it, and `FinalFitOutcome` records it.
  - Nonlinear and conditional amplitude errors are the sandwich where it is
    modelled.
  - Fitted values, σ, reduced χ² and the MCMC sampler are unchanged.
  - Every test and tool states its noise model explicitly; none defaults to "not
    modelled".
- **Formats.** Scientific archive 8.0.0, fit JSON 7.0.0 and analysis project 11.0.0
  record the noise model; the reader refuses an inconsistent one. A
  `noise_correlation` column goes in `parameters.csv` and `intensities.csv`.
  Projects restore the recorded model without re-deriving it.
- **Statements.** The CLI pre-fit summary, the report, the GUI inspector and the
  profile note state the noise model. MCMC always states its independent-noise,
  uncalibrated status; its record is unchanged.
- **Validation** (`docs/validation/sandwich-errors.md`):
  - published = sandwich in every realization;
  - calibrated for `sp1`, `sp2` and `no_apod`;
  - one recorded `sp1` failure (0.88) was replicated as a fluctuation.
- **Near coalescence.** 87 of 150 fits carry no warning and cover their positions
  0.65 · 0.82, through bias; amplitude correlation does not identify them. This is
  documented as a limitation of strongly overlapping or near-coalescent peaks, not
  of overlap as such, with no boundary claimed. No resolution-based warning: an
  advisory review flag, separate from covariance withholding, needs its own study
  showing it useful before any threshold is set.
- **Golden reference.** Regenerated: fitted values, classifications, warnings and
  χ² are bit-identical; only the errors changed.

## 2026-11-05: NMRPipe phase sign, digital filter and header limits (correlated noise)
- **First-order phase sign.** The noise correlation applied `P1` with the opposite
  sign to NMRPipe, which stores frequency in decreasing order. It is now
  ρ(k) = Re[C(k) e^{−iφk/M}]/C(0), verified against spectra NMRPipe 13.0 processed.
  - The earlier convention overstated amplitude errors by up to 5% on axes with `P1`
    180 and 9% with `P1` 360.
  - The validation tools emulated NMRPipe with the same mirrored sign for both noise
    and model, so their calibration evidence stands; they now use NMRPipe's sign and
    were rerun.
- **Bruker digital filter.** When FT corrects the filter delay (`FDDMXFLAG` 1, the
  default for Bruker data converted with `-decim`), NMRPipe applies an extra
  first-order phase of −360°·`FDDMXVAL` on the direct axis. Ignoring it understated
  errors by 10–38% on the tested chains.
  - It is recorded as `AxisProcessing.filter_delay`.
  - `FDDMXFLAG` 0 means FT did not apply it; any other state is not modelled.
- **Recorded window terms the model lacks** (`SP -elb`, `-glb`, `-goff`, `-df`) leave
  the run not modelled, with the reason.
- **Regression fixtures.** NMRPipe-processed impulse responses in `tests/data/nmrpipe`
  (`tools/make_nmrpipe_noise_fixtures.py`) give NMRPipe's exact noise covariance. The
  model matches it to 5 × 10⁻⁵.
- **What the header cannot show.** These leave a header indistinguishable from
  ordinary processing, so no header check is added:
  - linear prediction (`TDSIZE` and `APOD` give the extended length);
  - earlier windows, and `SP -size`, `-start` and `-one`;
  - digital-filter correction during conversion;
  - solvent filters;
  - NUSZF extrapolation and NUS reconstruction.

  [correlated-noise.md §2.5](correlated-noise.md#25-what-the-final-header-cannot-show)
  records their measured effect.
- **Noise correlation is not calibrated uncertainty.** With LP and NMRPipe's IST
  (default settings, plane by plane), parameter scatter exceeded the published errors
  1.5–6.6-fold in synthetic series. The extrapolation and reconstruction also changed
  fitted ¹⁵N linewidths:
  - LP: unbiased for isolated peaks at high SNR; +24% to +192% at low SNR or in
    overlap.
  - IST extrapolation: narrower lines, −27% to −51% without noise.
  - IST reconstruction of 50% Poisson-gap NUS: broader lines, +158% to +190% without
    noise.

  Peak areas, and relative amplitudes of isolated peaks across planes, were
  preserved. SMILE was not tested: its plugin failed on the test machine. These
  results apply to the tested conditions only. A user-declared reconstruction mode
  is a separate decision.
- **Formats.** Scientific archive 9.0.0, fit JSON 8.0.0 and analysis project 12.0.0
  record the filter delay; earlier files are refused.
- **Validation and golden reference.** The sandwich study was rerun.
  - Two of 18 calibration checks fail in the recorded run, each by a hair: `sp1`
    isolated ¹H position, scatter/SD 0.88; `sp1` overlapping linewidth, 68%
    coverage 0.716 against a limit of 0.715.
  - Independent replications pass both.
  - In the golden reference, fitted values, classifications, warnings and χ² are
    bit-identical. Amplitude errors fall 0.4–2.5% (median 1.8%) and parameter errors
    move −2.1% to +2.5%, from the ¹⁵N axis's `P1` 180.

## 2026-11-05: sp1/sp2 acquisition time from the acquired signal
- **Defect.** The `sp1`, `sp2` and `no_apod` lineshapes integrate the signal under
  the window over 0 ≤ t ≤ aq, with aq = `APOD`/`SW` and the recorded window.
  NMRPipe evaluates the sine bell on samples j = 0 … N − 1, which the signal reaches
  at (j − g + d)·dw:
  - d = `P1`/360, the sampling delay;
  - g, the Bruker digital-filter delay that FT corrects after the window.

  Linewidths were overstated at short acquisitions: +4% to +41% for 64-point `sp2`
  axes sampled from t = 0, and 5–83% on 512–1024-point Bruker direct axes. Half-dwell
  axes were overstated too, at short lengths, in PeakFit's cluster-sized fits.
- **Rule** ([lineshapes.md](lineshapes.md)):
  - `sp1` and `sp2` use aq = (N − 1 + d − g)·dw, with the window's span over the
    signal. They use `TDSIZE`, the full spectral width (so extraction no longer uses
    the rounded `APOD`), `P1`, `FDDMXVAL`/`FDDMXFLAG` and `APODDF`.
  - `no_apod` keeps N·dw, less g.
  - The kernels are unchanged, and there is no new lineshape family.
  - The convention follows the sampling delay `P1` records, not a `P1` ≈ 180
    threshold, so a small manual `P1` changes aq by `P1`/360 of a dwell.
- **Evidence** ([validation/acquisition-time.md](validation/acquisition-time.md)):
  NMRPipe-processed noiseless signals, fitted by PeakFit.
  - Exact for t = 0 (first point ×½) and half-dwell (×1): within 0.6% at 64 points,
    and 0.3% on indirect axes at 32.
  - First-order for other delays.
  - Positions and relative amplitudes are unchanged.
  - GB3 real data: the gross dependence of linewidth on acquisition length disappears.
- **Golden reference** regenerated. The CEST example is Varian data (`var2pipe`), with
  no digital-filter correction; Bruker handling is validated by the Bruker fixtures
  and GB3. Linewidths +0.3% median, amplitudes of unwarned
  clusters +0.3–0.8%, positions ≤ 0.49 SE, classifications and warnings unchanged.
- **Real-data reference values changed.**
  - `test_coincidence`: the 5N-H/76N-H pair now re-collapses from default starts like
    the other retained pairs. A resolved fit still has lower χ² (18045 against
    18454), so this is a known optimizer limitation. It is recorded as a strict
    `xfail` and tracked in #135.
  - `test_experimental_overlap`: the released local solutions reached with
    historical settings move by +0.96, −0.02 and +0.09 Hz in ¹⁵N. Cluster 269, which
    holds the 36N-H/93N-H identifiability pair, now carries the uncertainty warning
    from either start.
  - `test_continuation`: the 102N-H/52N-H pair's reduced χ² is 1.5440 instead of
    1.5296. Its qualitative behaviour is unchanged: resolved when continued,
    collapsed when only positions are carried.
- **Formats.** Scientific archive 10.0.0 and analysis project 13.0.0. Both record the
  derived spectral parameters, which continuation and project reopening compare with
  the spectrum as read now. Under the new rule, earlier files failed that comparison
  as an apparent calibration change, so they are now refused by version: refit.
  - Fit JSON (8.0.0) records no spectral parameters and stays valid.
  - MCMC records (1.1.0) are bound to their parent archive, which is refused.
- **Digital-filter metadata.** Missing, unreadable or unrecognised `FDDMXVAL` or
  `FDDMXFLAG` is refused when reading the spectrum, never taken as "no correction".
- **Not checked: first-point scaling against the sampling delay.** A mismatch
  m = (`C1` + 1) − ½ − d adds a constant offset whose linewidth bias grows as m/N.
  It is documented with measurements in [lineshapes.md](lineshapes.md), not
  diagnosed:
  - the offset depends on the true delay, which the header does not record;
  - m computed from `P1` is wrong whenever `P1` includes manual phasing;
  - any warning would need an arbitrary threshold.

## 2026-11-05: Review cleanup
- **No information criteria.** AIC, BIC and the log-likelihood are no longer
  computed or published, per cluster or per run. The removed values were computed
  from an independent-Gaussian likelihood, which the processing-derived correlation
  contradicts (correlated-noise.md §4). AIC and BIC are not themselves invalid for
  correlated observations; they would need the correlated likelihood, and nothing in
  PeakFit used them or documented them.
- **Reduced χ² unchanged.** Its calculation and the review threshold stay as they
  were. It is computed from unwhitened residuals, so under correlated noise it is a
  normalized residual diagnostic and heuristic review flag, not a calibrated
  chi-squared test ([outputs.md](outputs.md#unavailable-and-unreliable-uncertainties)).
- **Formats.** Scientific archive 11.0.0, fit JSON 9.0.0 and analysis project
  14.0.0; earlier files are refused by version: refit. The MCMC sampler's own
  likelihood is unaffected.
- **The noise model enforces its own rules.** `AxisProcessing` and `NoiseModel`
  refuse anything outside the supported noise-model domain:
  - a model with both axes and a reason, or neither;
  - an FT size below the acquired size;
  - an unknown window, a non-positive power, or a non-finite value.

  The archive validator keeps only the check that needs the samples: one axis per
  spectral coordinate.
- **No silent "independent points".** `fit_cluster`, `identifiable_covariance`,
  `_calculate_errors` and `calculate_amplitudes_with_uncertainty` require their
  `correlation` argument, so every caller says `None` or passes R.
- **One review threshold.** `engine.results.HIGH_REDUCED_CHI_SQUARED` replaces three
  copies of 5.0 (fit review, CLI and terminal summary).

## 2026-11-05: Default linewidth start of 2 Hz (#135)
- **Change.** sp1, sp2, no_apod and their doublets start every linewidth at 2 Hz
  instead of 20 Hz. Configured starts and continued runs take precedence as before.
  Gaussian, Lorentzian and pseudo-Voigt keep 25 Hz.
- **Why.** A start broader than the true lines makes overlapping shapes nearly
  identical from the first step, and the fit can stop in the collapsed limit. On the
  bundled data this caused 5N-H/76N-H and 36N-H/93N-H to collapse although resolved
  minima of lower χ² exist. On synthetic pairs with known truth, 20 Hz failed in
  16 of 108 cases and starts of 1–5 Hz in none, including on 60 Hz lines, with fewer
  function evaluations ([validation/linewidth-start.md](validation/linewidth-start.md)).
- **Consequence.** 53N-H/147N-H also resolves. Its χ² is 20% above the collapsed
  limit, which fits better only through diverging amplitudes. It loses the
  uncertainty warning; its misfit shows in χ²ᵣ and the residuals.
- **Not chosen.** A retry or several starts for difficult clusters, with the lower χ²
  kept: a second optimizer path and a selection rule, while the default stayed wrong
  everywhere else. A start scaled to the acquisition time, or the median width of
  isolated peaks: more machinery, with no advantage measured over a fixed narrow
  value.
- **Golden reference** regenerated: only the three formerly warned clusters change.
  The collapse-retention tests pin the former 20 Hz start, which still collapses
  the bundled pairs.

## 2026-11-06: Per-member amplitude identifiability (S1)
- **Change.** The amplitude profile classifies each member as identifiable, not
  established, or not identifiable, and draws only amplitudes the data determine.
  It withholds the others with a statement of which class applies, keeps the
  recorded values under Numerical diagnostics, and shows no combined amplitude.
  Hollow markers, dashed bars and the cluster-level normalization rule keyed on the
  uncertainty warning are removed.
- **Why.** The cluster-level uncertainty warning mixes active bounds, missing
  nonlinear σ and decomposition problems. A rank-deficient cluster can still
  determine some members, and an ill-conditioned or near-collinear one is not
  thereby unidentifiable. On the bundled data the real collapsed pairs keep full
  rank (s_min/s₀ ≈ 10⁻⁴) and finite σ; drawing their ±6×10¹² amplitudes dominated
  the axes ([validation/amplitude-identifiability.md](validation/amplitude-identifiability.md)).
- **Rule.** `engine/diagnostics/identifiability.py` reuses `factorize_shapes` and
  its conventions: null space below the rank cutoff ε·max(m, n)·s₀, ill-conditioned
  directions s ≤ √ε·s₀, and ε·max(m, n) as the numerical zero for a unit vector's
  projection; near-collinearity is the existing 1 − |cos| ≤ 10⁻⁴ rule. No new
  threshold. The classification is conditional on the fitted lineshapes and says
  nothing about identifiability when nonlinear parameters vary.
- **Not chosen.** Withholding every member whenever the joint covariance is
  withheld: it would call determined amplitudes unidentifiable. A summed amplitude
  for degenerate pairs: not shown until its identifiability is validated.

## 2026-11-06: Comparable change summary after a refit (S2)
- **Change.** "Changes from run …" lists changed clusters first, with new and lost
  numerical warnings (including the decomposition marks of S1) leading, then edited
  clusters. Recorded χ²ᵣ, |r|max and |ρ| appear before → after only when both runs
  share members, fitting samples, σ_stat, noise model and varied-parameter count;
  otherwise the first difference is named. The run-level median χ²ᵣ is removed.
- **Why.** In the bundled degenerate refit (147N-H moved, refit), the 53N-H/147N-H
  cluster's χ²ᵣ fell from 3.00 to 2.53 while its fitting samples grew from 67 to 68
  per plane and the pair collapsed. The two values describe different problems, and
  the lower one accompanies an unreliable decomposition.

## 2026-11-06: One spectral view with native slices; cluster focus replaces the lens
- **Slices.** The row and column slices are lines through native samples with no
  markers; Overlay draws the cached total-model plane at the same samples. Nothing
  is evaluated off the sampling grid, so the slice shows what was fitted.
- **Scale.** (Replaced on 2026-11-11 by a range per slice.) Both slices share the region intensity scale: min(0, D) to max(0, D)
  over the visible 2D region and every plane, padded 5 % of the span. Asymmetric,
  because a symmetric ±max|D| range halves the room of one-signed spectra; zero
  included so sign is never ambiguous; all planes so the scale holds while
  stepping planes; recomputed only on a deliberate view change so hover never
  rescales. Not tied to fitting masks, which are optimizer bookkeeping. A manual
  wheel scale persists until reset, a new dataset or an unrelated focused cluster.
- **Residual.** Residual lanes share one ±N σ_stat scale, set separately from the
  intensity scale. In Residual mode the slices themselves show the residual and the
  lanes hide, keeping the lane preference for Overlay.
- **Annotations.** (Caption and scale bars removed on 2026-11-11.) A zero baseline, one compact scale caption with units and
  clipping counts (data ▲ and model △ separately), exact values in the corner
  readout. The 2D ppm axes stay quantitative; slices carry no intensity tick axis.
- **Focus.** The three-panel lens is replaced by cluster focus: the same view
  reframed in place on displayed members ± max(3 × fitted FWHM, 8 samples), capped
  at a quarter of the spectral width and clipped to the spectrum. One view means
  one set of gestures, marks, modes and settings; the lens duplicated them and its
  separate plots could not share the overview's slices. Leaving focus restores the
  overview's view state exactly and never reverts edits or fits.
- **Window.** PeakFit never enlarges the window. Its minimum is the visible parts'
  minimums; space is taken from residual lanes (24 px), slices (72 px), then the
  other side panel; otherwise the action is declined with a message. Layouts
  restore spectral area, inspector, navigator in that order. 1160×720 is the
  validated reference with both side panels, slices and lanes.
- **Removed.** “Fit current traces” and the per-view trace gain. The stored
  `trace_limit` in project dataset headers is now unused; its removal is a
  separate storage cleanup.

## 2026-11-06: `trace_limit` removed from datasets and project headers
- `Dataset.trace_limit` and the `"trace_limit"` dataset-header key are removed; slices use the region intensity scale computed from the visible data. Breaking: new projects no longer write the key and `Dataset(...)` loses that argument; older projects still open (the reader never read the key) and keep it in their carried-over header until re-imported.

## 2026-11-06: Model planes from per-axis shapes
- **Change.** The GUI evaluates a model plane with `project_final_grid`: for each
  cluster, its per-axis shapes, then amplitude-weighted outer products, instead of
  evaluating every peak at all 139 776 grid points. The per-axis shapes are
  recomputed for every plane requested; nothing is cached beyond the existing
  model-plane cache.
- **Why.** On the bundled data a first visit to a plane in Overlay took 1.3 s,
  97 % of it model evaluation (1.26 s; contours 2–4 ms), so stepping planes stalled
  at every new plane. It now takes 52 ms p50 (59 ms p95) to paint, of which the
  model plane is about 10 ms (per-axis shapes about 9 ms, outer products 0.14 ms).
- **Same model.** A peak shape is defined as the product of its per-dimension
  shapes (`Cluster.evaluate_peaks`), so only the summation order changes. Tests
  require agreement within 10⁻¹² of the largest model value on every plane, for
  Gaussian, Lorentzian and pseudo-Voigt shapes with a folded peak, and for the
  bundled SP2 data (observed 1.2×10⁻¹⁵). The worst case for a change of summation
  order, strongly cancelling amplitudes, is tested on the degenerate 53N-H/147N-H
  refit (amplitudes about ±9×10¹² cancelling to a model of 2×10⁶): max |Δ|
  3.9×10⁻⁷ a.u., i.e. 3.3×10⁻¹¹ σ_stat and 1.5×10⁻¹³ of the largest model value,
  within the same criterion. Clusters must agree on the number of fitted planes,
  as in the pointwise path. The pointwise `project_final_outcome`
  and the CLI's full-grid simulation are unchanged, so no output file changes.

## 2026-11-11: A dataset opens without a confirmation; a suspicious mapping is checked
- **Change.** The "Open validated dataset" box is removed. It repeated the chosen
  files with 8-decimal ppm ranges and six peak names, so it could not show whether
  the peak-axis mapping was wrong. Instead, `gui.spectrum.mapping_warning` counts
  the peaks outside the direct axis window; when more than half are, opening or
  importing peaks asks first, with the count for each mapping.
- **Why.** A swapped mapping is the one mistake to catch at this step, and on the
  bundled data it puts 166 of 166 peaks outside the direct (H) window against 0 for the
  right mapping. The indirect axis is not counted because folded peaks lie outside
  it legitimately. The check reports and never chooses; opening anyway stays possible.

## 2026-11-11: Each slice scales to its own samples; no intensity scale marks
- **Change.** Each 1D slice takes its own range: min(0, D) to max(0, D) of its
  visible samples over every plane, padded 5 %. It follows the row or column shown
  and the view, and holds across planes. The wheel sets a gain on both slices
  instead of a manual range. The scale bars with their +/− signs are gone, and the
  caption above the row slice appears only for a residual or lane scale, a gain or
  clipping. The readout shows intensity and model in one power of ten. The menu
  item is now **Default slice intensities**; saved views store `gain` instead of
  `scale`.
- **Why.** The shared region scale, set by the strongest peak in view, flattened
  every weaker row or column into the baseline, so the slices rarely showed a
  usable lineshape. A slice is read for shape and fit quality; exact values are in
  the readout. Keeping all planes in the range preserves the amplitude change from
  plane to plane, and zero stays included so sign is never ambiguous. Without a
  shared scale the bars carried no number and only added clutter.
- **Cost.** Rescaling each slice on hover redraws it whole: on the bundled dataset,
  hover delivery-to-paint p95 rose from 5.8 to 9.2 ms (macOS, 2× DPR). The
  validation target moves from 8 to 10 ms, still within one 60 Hz frame and
  PERF-1's 16.7 ms; readable lineshapes are worth it.
- **Breaking.** A manual slice scale saved by an earlier project is ignored (the
  view opens at gain 1).

## 2026-11-11: Values stay on screen; explanations go to tooltips
- **Rule.** A screen shows values and states once. Explanations of fixed rules
  move to a tooltip or an “About …” disclosure; counts already shown elsewhere
  are not repeated.
- **Loaded screen.** The toolbar says “Not fitted” (counts are in the Peaks tab,
  the plane field and the Analysis page); the Analysis page has no run section
  before a fit and no “Select a peak …” hint; the lineshape note is one line;
  the noise method, the passes note and the fitting-region note are tooltips; the
  navigator footer appears only for a filter or a multiple selection. Noise and
  clustering level use ×10ⁿ instead of e+ notation.
- **Why.** The loaded screen repeated the peak count four times and spent most of
  the inspector on captions, so the values and the Fit button were hard to find.

## 2026-11-11: Peak page: short qualifiers stay, explanations go to tooltips
- **Change.** Applies the 2026-11-11 text rule to the selected peak, partly revising
  T3 (#144): a qualifier that changes what a number means stays beside it in a few
  words, and its explanation moves to a tooltip or About. The profile note reads
  “±1σ conditional on lineshapes · noise correlation modelled” (χ²ᵣ is with the
  measures; off-scale points are listed by plane). Measures read “χ²ᵣ 3” (sample
  count in the tooltip), “Largest residual 7.5 σ · guide ≈ 4.3 σ” and “ρ −0.77
  with 147N-H, conditional on lineshapes”. Empty review families take one line.
  “Unfolded ppm” is shown only beside a displayed alias. “Review cluster ›” no
  longer repeats the size in the header; the status line is gone before a fit;
  the cluster list shows its run line only when Outdated and its footer only for
  a filter or warnings. Off-scale tooltips use ×10ⁿ.
- **Why.** The fitted peak page spent four lines on the profile note and one
  caption per measure, so the values were hard to find; the qualifiers that
  matter scientifically still sit next to the numbers they qualify.
- **Layout.** One grammar for the inspector: every line on one left edge; forms,
  the Fitted and starts tables and the review share one label column (review rows
  replace the bulleted list, on the peak and cluster pages alike); sections start
  with the same spacing; disclosures are text triangles in small type on the left
  edge; captions wrap only between phrases; empty labels leave no gap. The peak
  editor is a plain widget, not an untitled group box with its own margins.

## 2026-11-12: Cluster page: one line per member
- **Change.** Members and the parameter table are one table: number, a link to the
  member's page, then its recorded values; the units sit under it without the
  “Recorded” prefix. Two members show one ρ row with “conditional on lineshapes”
  under it, like the peak page; three or more keep the matrix with the same short
  caption, the rest in its tooltip. A single-peak cluster has no correlation
  section instead of “Not applicable”. Provenance is a text disclosure on the left
  edge, and sections use the inspector's shared spacing.
- **Why.** Members, parameters and matrix were three lists keyed by the same
  numbers, so reading a member's values meant matching numbers across sections; a
  2×2 matrix shows one value four times. Names keep their natural width rather than
  the shared label column, because the parameter columns need the room in a 310 px
  inspector.

## 2026-11-12: Analysis page: the run and its changes as label rows
- **Change.** The active run is label rows (Run, Model, Converged, Usable, Fitted,
  Warnings) instead of one wrapped paragraph; Fitted reads “2026-11-12 01:56 UTC ·
  3.0 s” instead of an ISO stamp. “Changes from parent run” is a section like the
  others, with the parent's ID in a **Parent run** row instead of an uppercased title
  (“E48D0167”). The whole-run counts are rows, and a count that did not change reads
  “121, unchanged” instead of “121 → 121”. Position differences round before
  formatting, so none reads “−0.0000”; when all round to zero, one row says so. The
  positions caveat and the unexplained-signal explanation move to tooltips; the
  short qualifier “Diagnostic only” stays. The fit setup's values use the
  inspector's label column.
- **Why.** The run block repeated the toolbar in a paragraph that wrapped
  mid-phrase; “0 → 0” and five “−0.0000 ppm” lines after an unchanged refit said
  nothing at length.

## 2026-11-12: One profile transform for GUI and CLI; correct R2eff errors
- **Change.** Reference ratios and R2eff live in `engine.scientific.profiles`; the
  GUI amplitude profile and `peakfit plot cest|cpmg` both use them. CPMG R2eff
  errors are σ_r / (r·T): the CLI omitted the 1/r of d(−ln r)/dr and understated
  σ by the factor r (×0.3 on a plane attenuated to 30 %). CLI plots now keep an
  unavailable amplitude error unavailable instead of reading it as 0, report a
  ratio of an exactly zero intensity with its own σ instead of 0, and skip, with
  the reason, a peak whose reference fails the GUI's σ_R/|R| ≤ 0.1 limit.
- **Why.** The two front ends computed the same quantity separately and disagreed
  at the edges; one implementation, tested against Monte Carlo, removes the
  disagreement and the R2eff error. A test pinned the wrong R2eff σ and was
  corrected with this explanation.
