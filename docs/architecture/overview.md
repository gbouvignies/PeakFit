# Current Architecture Overview

This page describes PeakFit as audited on 2026-07-26. It is a map of the current
implementation, not a target design. Findings use these labels:

- **Verified** — supported directly by source, tests, configuration, or observed
  execution.
- **Likely** — strongly suggested by the evidence but not conclusively established.
- **Unknown** — requires scientific, historical, or additional runtime evidence.
- **Recommendation** — a proposed future action, not current behavior.

## Public entry points and workflows

**Verified.** The installed console script `peakfit` and `python -m peakfit` both
dispatch to the Typer application in `peakfit.cli.app`. The package root otherwise
exports only `__version__`.

The CLI exposes:

| Entry point | Current workflow |
| --- | --- |
| `peakfit init [PATH]` | Generate a validated default TOML configuration. |
| `peakfit fit SPECTRUM [PEAKLIST]` | Validate inputs, load an NMRPipe spectrum, read a supplied peak list or run experimental automatic peak picking, fit peak clusters, and write a result directory. |
| `peakfit mcmc RESULTS` | Load a fitted state, select whole peak clusters, sample nonlinear lineshape parameters with `emcee`, solve amplitudes analytically, report diagnostics, and optionally save HDF5 chains. |
| `peakfit plot intensity RESULTS` | Plot fitted amplitude against plane value or plane index from `tables/intensities.csv`. |
| `peakfit plot cest RESULTS` | Normalize fitted amplitudes to reference points and plot a CEST profile. |
| `peakfit plot cpmg RESULTS --time-t2 T` | Convert fitted amplitudes to effective transverse relaxation rates and plot a CPMG profile. |
| `peakfit plot mcmc RESULTS` | Read saved HDF5 chains and write MCMC diagnostic pages. |
| `peakfit plot spectrum --spectrum FILE [--results RESULTS]` | Open the Qt spectrum viewer, optionally with peaks and reconstructed fit overlays. |

**Verified.** `fit` accepts `.ft2` and `.ft3` NMRPipe inputs and Sparky, CSV,
JSON, or Excel peak lists. Omitting the peak list activates the experimental
automatic picker. The observed help still describes the overall CLI as
“pseudo-3D”, while repository guidance and domain language use the broader
“pseudo-ND”.

## Principal domain state

| Concept | Current representation and aliases |
| --- | --- |
| Pseudo-ND spectrum | `Spectra`; `Spectrum` is an alias. `data` uses plane-first shape before clustering, and `z_values` stores plane values. |
| Spectral metadata | `SpectralParameters` per dimension and derived `DimensionInfo`/`DimensionContext` conversion objects. The pseudo dimension is labelled `F1`; spectral axes begin at `F2`. |
| Peak | Mutable `Peak` with a name, current and starting positions, and one `Shape` implementation per spectral axis. |
| Peak cluster | Mutable `Cluster` with peaks, spectral grid indices, a strict `(n_points, n_series)` data matrix, and mutable cross-talk corrections. Construction rejects other dimensionalities and point-count mismatches. Code generally shortens the term to “cluster”; some user documentation says “overlap cluster”. |
| Lineshape | The `Shape` protocol, `ShapeBase`, and twelve configured singlet/doublet implementations across Gaussian, Lorentzian, pseudo-Voigt, SP1, SP2, and no-apodization families. |
| Parameters | Mutable scalar `Parameters`/`Parameter` objects plus vectorized `FitParameters` and `ParameterMap`. Canonical identifiers have `peak.axis.label` or `cluster_N.axis.label` form; amplitudes use the pseudo axis and an index. |
| Fit state | `FittingState`, which stores mutable clusters, both parameter representations, noise, and a version for continuation and diagnostics. It is not completed scientific truth. |
| Optimizer result | `FitResult` for one cluster, including a first-class run-local `cluster_id`, fitted scalar parameters, residuals, convergence metadata, and amplitude parameter count. `FitEvaluation` independently classifies that result as converged, usable non-converged, or unusable and carries a typed analytical evaluation only when usable. |
| Completed fit outcome | `FinalFitOutcome` is the immutable authoritative completed result. It has ordered per-cluster outcomes plus `cluster_id` lookup, final nonlinear values, a copied ticket-03 analytical evaluation for usable clusters, and terminal optimizer provenance. |
| Run state and output model | `PipelineResult` carries mutable continuation state alongside `FinalFitOutcome`, but is not itself scientific authority. `LoadedData`, `FitRun`, and `RunSummary` describe orchestration; `FitResults` and related dataclasses still describe writer input; Pydantic schemas describe JSON output. |

**Verified.** Peak positions, cluster corrections, scalar parameters, parameter
caches, and portions of UI state are mutable. `Parameter` keeps a parent
back-reference so changing `vary` invalidates a collection cache. `FittingState`
contains scalar and vector parameter representations that are constructed from
one another but have no continuing synchronization invariant.

## Fit data flow

1. **CLI and configuration.** `fit_command` performs Typer argument handling and
   blocking preflight validation. `build_fit_config` either loads a strict
   `PeakFitConfig` from TOML or builds one from CLI values. With a TOML file,
   only output directory, formats, headless mode, noise, and contour are
   overridden by CLI values; other fit flags do not override the file.
2. **Spectrum input.** `read_spectra` uses `nmrglue` to read NMRPipe data,
   converts it to `float32`, adds a leading plane axis for an ordinary
   frequency-domain input, reads or synthesizes plane values, builds spectral
   metadata, and excludes configured planes.
3. **Noise and lineshapes.** `load_data` estimates or accepts the noise level,
   resolves a contour level, and selects a lineshape per spectral axis from
   explicit configuration or NMRPipe apodization metadata.
4. **Peaks.** `read_list` parses a supplied peak list and creates `Peak` plus
   axis-specific shape objects. Without a peak list, `auto_pick_peaks` performs
   residual-driven, contour-connected ROI growth, trial VARPRO fits, and F-test
   decisions; the optional Qt stepper can intervene between cycles.
5. **Clustering.** `create_clusters` thresholds the absolute signal across
   planes, adds dilated masks around supplied peak positions, merges wrapped
   connected components, assigns peaks, and extracts each component as a
   point-by-plane matrix.
6. **Parameter setup and fitting.** `run_fit` creates scalar parameters and
   executes `run_pipeline_iter`, using a process pool when requested. The
   pipeline rejects duplicate input cluster identifiers, carries `cluster_id`
   through optimizer tasks and results, validates the exact returned identity
   set, and sorts completed results by identifier for presentation. Each fit
   step rebuilds cluster-local parameters, applies global and step constraints,
   calls either VARPRO or basin hopping, classifies each returned result through
   the shared analytical evaluator, merges parameters only from numerically
   usable results, and updates cross-talk corrections between passes.
7. **Numerical evaluation.** A cluster multiplies its axis lineshapes into a
   peak-by-point design matrix. VARPRO optimizes nonlinear lineshape parameters
   and solves per-plane amplitudes by QR-based linear least squares. The shared
   `evaluate_analytical_model` operation re-solves amplitudes from corrected
   data, validates finite values and compatible shapes, and returns model
   values, residuals, chi-squared statistics, and amplitude-uncertainty scaling
   inputs as one typed value. Optimizer convergence remains independent from
   this numerical usability decision.
8. **Final state and results.** The pipeline freezes its terminal correction
   snapshot, validates terminal optimizer results and ticket-03 evaluations by
   `cluster_id`, and assembles one immutable `FinalFitOutcome`. It returns that
   outcome alongside mutable `FittingState` continuation state. `build_fit_results`
   still separately re-evaluates every cluster for legacy writer input; migrating
   that projection is deliberately deferred.
9. **Persistence.** Writers produce schema-versioned `summary/fit.json`,
   `tables/parameters.csv`, `tables/intensities.csv`, optional shifts and
   Markdown output, a run README, and optionally a simulated spectrum.
   `metadata/fitting_state.pkl` is always written for stateful post-fit use.
10. **Downstream use.** Profile plots read `tables/intensities.csv`. MCMC and
    spectrum reconstruction prefer the pickle; if it is absent they fall back
    to a minimal state reconstructed from JSON. MCMC chains are stored in HDF5
    and consumed by `plot mcmc`.

## Current responsibility boundaries

| Boundary | Current ownership | Audit result |
| --- | --- | --- |
| CLI orchestration | `peakfit.cli` | **Verified.** Typer parsing, UI selection, exception translation, config overrides, and workflow calls live here. `fit_command` also directly coordinates result persistence. |
| Validation and parsing | `fit.validation` and `io.readers` | **Verified risk.** Spectrum validation reuses the real reader; peak-list validation has a separate set of parsers whose dimensionality and naming rules differ from the parsers used by the fit. |
| Domain state | `engine.domain` | **Verified.** Domain objects and parameter constraints live here. Several central objects are deliberately mutable. |
| Lineshape evaluation | `engine.lineshapes`, `engine.types`, `Cluster.evaluate` | **Verified.** The `Shape` protocol is a substantial seam shared by multiple genuinely different model families. |
| Optimization and fitting | `engine.algorithms`, `engine.fitting`, `fit.pipeline` | **Verified.** The engine owns numerical methods; the pipeline owns steps, cluster task preparation, parallel mapping, parameter merging, and correction updates. |
| Result construction | `engine.algorithms.evaluation`, `engine.results`, `fit.result_models`, `fit.results` | **Verified risk.** The pipeline now shares one numerical evaluation and usability classification, but output construction still independently recomputes amplitudes and statistics without consuming that evaluation or final optimizer provenance. Its migration belongs to the final-outcome and consumer work. |
| Persistence | `io.schemas`, `io.writers`, `io.state` plus `fit.fitting` | **Verified.** Structured writers live in `io`, but run-level selection and state persistence are coordinated by `fit.fitting`. |
| Plotting | `plot` | **Verified.** Profile transformation and Matplotlib output are separate from the Qt spectrum viewer. |
| Automatic picking | `auto_pick`, with `fit` and optional `ui` adapters | **Verified.** The algorithm is isolated from Qt; reporting callbacks and the optional stepper are supplied by higher layers. |
| MCMC | `engine.algorithms.mcmc` and `mcmc.analysis` | **Verified.** Sampling math is in the engine; result-directory loading, cluster selection, and formatting are in the workflow package. |
| Terminal and GUI UI | `ui`, CLI, and Qt modules under `plot` | **Verified.** Rich output is mostly behind a shared `Reporter` protocol, but `fit.pipeline` emits Rich markup in status strings. Qt is imported lazily for optional interactive paths. |

## Coupling, state, and cycles

- **Verified.** Import-linter analyzed 133 files and 254 dependencies: the
  configured package layering and sibling-module independence contracts both
  pass. No configured import cycle is currently reported.
- **Verified.** `fit_cluster_worker`, `fit_single_cluster_task`, and
  `run_pipeline` are pass-through layers around optimizer dispatch and the
  iterator pipeline. They currently support multiprocessing/presentation
  adaptation but add little domain behavior themselves.
- **Verified.** `Cluster.corrections` is updated after every pipeline pass, and
  `Peak.update_positions` mutates both peak positions and shape centers.
- **Verified.** MCMC multiprocessing uses a module-level mutable `_mcmc_state`
  initialized separately in each worker. Serial runs reuse the same object.
- **Verified.** UI verbosity is process-global. Interactive fit and MCMC views
  also use mutable closures to bridge progress events into Rich live displays.
- **Verified.** `ResultsLoader` calls JSON loading canonical, but its reconstructed
  clusters contain dummy one-point data and reconstructed shapes. Stateful
  numerical workflows therefore depend in practice on the pickle when present.
- **Likely.** The duplicated scalar/vector parameter state and reconstruction
  fallback can drift or be mistaken for equivalent representations even though
  they preserve different information.

## Scientific and compatibility-sensitive behavior

- **Verified.** All lineshape `lw` parameters are defined as FWHM in hertz; for
  apodized models this means the no-apodization-equivalent FWHM.
- **Verified.** Chemical shifts are in ppm. J coupling and linewidth are in
  hertz. Phased lineshapes evaluate complex kernels, but the fitting residual
  path uses real-valued cluster data.
- **Verified.** Peak clusters are contour-connected components over the union of
  signal-above-threshold and dilated peak-position masks, including wrapped
  spectral boundaries. Cluster identity and composition are therefore sensitive
  to noise, contour choice, peak positions, and wrapping.
- **Verified.** Per-plane amplitudes are linear least-squares quantities and are
  exported as “intensity”. The repository does not establish that they equal a
  broader experiment-independent physical intensity.
- **Verified.** Refinement changes the data supplied to each cluster by
  subtracting the current modeled contribution of peaks assigned elsewhere.
- **Verified.** Basin hopping and MCMC are stochastic unless seeds are controlled;
  MCMC currently creates an unseeded NumPy generator.
- **Verified.** The JSON contract is schema version `3.0.0`. Model parameters and
  amplitudes are intentionally split between `parameters.csv` and
  `intensities.csv`; JSON cluster entries contain lineshape parameters but not
  per-plane amplitudes.
- **Verified.** Config-file precedence, canonical parameter identifiers, result
  directory layout, pickle availability, input-file relocation heuristics, and
  default output formats are downstream compatibility surfaces.

## Architectural risks and missing protection

1. **Resolved — point/series axis contract.** Cluster data have one canonical
   representation, `(n_points, n_series)`. `Cluster` validates dimensionality
   and grid point counts and owns the point, series, observation, and amplitude
   counts consumed by optimizer and persisted statistics. Unequal-dimension
   tests cover cluster creation, merging, fitting statistics, uncertainty
   scaling, and reconstruction. State version `2.0` and output schema `3.0.0`
   explicitly invalidate development artifacts with the incorrect semantics.
2. **Verified — validation/parser duplication.** Preflight peak-list readers can
   accept rows or names that the real readers later reject or interpret
   differently. There are no equivalence tests across supported formats.
3. **Partially resolved — result truth is split.** `FinalFitOutcome` now keeps
   terminal classification, provenance, and shared analytical values separate
   from mutable continuation state. CLI review and `RunSummary` project that
   outcome, while durable output writers still use their older paths and can
   describe convergence differently until their planned migrations.
4. **Verified — state persistence has two unequal paths.** Pickle preserves
   numerical state; JSON reconstruction is intentionally minimal and excludes
   amplitudes and real cluster grids. Tests only protect the canonical JSON path,
   not numerical equivalence or pickle-free MCMC/reconstruction.
5. **Verified — duplicated parameter representations.** `FittingState` persists
   scalar and vector parameters without an executable consistency check.
6. **Verified — stochastic workflows have thin protection.** One integration
   test proves a short MCMC run can start, but there are no fixed-seed posterior,
   saved-chain round-trip, burn-in, or basin-hopping regression tests.
7. **Verified — GUI workflows lack executable tests.** The optional auto-pick
   stepper and interactive spectrum viewer are not exercised by the current
   suite.
8. **Verified — several input contracts are unprotected.** No tests cover real
   CSV/JSON/Excel peak-list parsing, plane-value length mismatches, plane
   exclusion with supplied values, NMRPipe dimensional variants, or config-file
   versus CLI precedence.
9. **Unknown — scientific reference status.** The repository does not record
   whether the example peak parameters, amplitudes, residual thresholds, or
   expected fit quality were independently validated against a scientific
   reference.

## Completed bounded follow-up

The point-by-series data-shape follow-up established one strict cluster
representation and centralized count algebra without changing lineshape,
optimization, MCMC, or reconstruction mathematics. Count-derived statistics
and amplitude uncertainty scaling now use the series axis.
