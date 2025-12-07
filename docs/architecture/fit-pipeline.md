# Fit Pipeline Architecture

## Current Flow (Critical Path)

- CLI entry: `cli/commands/fit.py` → `FitService.fit`
- Orchestration: `services/fit/pipeline.FitPipeline.run`
- Load: `core.domain.spectrum.read_spectra`, `core.domain.peaks_io.read_list`, noise via `core.algorithms.noise.prepare_noise_level`
- Cluster: `core.algorithms.clustering.create_clusters`
- Protocol: `core.fitting.protocol` (steps/fix-vary) → `services.fit.fitting.FitRunner`
- Strategies: `core.fitting.strategies` → `core.fitting.optimizer` / `core.fitting.computation`
- Persist: `io.state.StateRepository` (cache/state.pkl)
- Outputs: `services.fit.writer` → `io.writers.*` (JSON/CSV/Markdown; optional YAML/legacy)
- UI/logging: `ui.*` intertwined with pipeline for console/progress and HTML export

## Layering (desired)

- Core (pure, no IO/UI): domain, fitting, lineshapes, results, diagnostics
- Services (orchestration): pipelines that depend on core and accept injected adapters
- Adapters: CLI, UI/reporters, IO writers/readers
- Contrib/Experimental: optional algorithms and visualization outside the core namespace

## Current Coupling Hotspots

- `FitPipeline` and `FitRunner` call `ui.*` directly (console, live display)
- Output writers include legacy/YAML formats by default in `results_writer`
- Plotting and analysis live alongside core and are imported by CLI commands

## Contrib / Experimental Layout Proposal

- `contrib/lineshapes`: sp1, sp2, no_apod/apodization variants (shim exports remain in `core.lineshapes.__init__` initially)
- `contrib/optimizers`: global optimizers (basin-hopping, differential evolution) registered via `register_strategy`
- `contrib/plotting`: plotting helpers and `services/plot` consumers
- `contrib/analysis`: post-fit analytics (MCMC/profile/uncertainty services and CLI analysis commands)
- `legacy/`: legacy writer and opt-in output formats

## Near-Term Refactors

1. **Reporter injection**: allow `FitPipeline`/`FitRunner` to use an injected `Reporter` (default console) so UI becomes an adapter, not a dependency
2. **Pure loading path**: `services/fit/loader.py` should be console-free; return dataclass with spectra/peaks/noise/contour
3. **Output gating**: gate YAML via `output.formats` and legacy outputs via `output.include_legacy`; keep JSON/CSV/Markdown as defaults per output-system instructions
4. **Tests to freeze numerics**: add regression suites for lineshapes, amplitude solve, residuals, VarPro vs leastsq parity, cross-talk corrections, multi-plane amplitudes
5. **Headless runs**: support a headless flag (no Rich live display) routed through Reporter for non-interactive/batch usage

## Headless and UI Flags

- CLI: `peakfit fit ... --headless/--interactive` controls live display without changing config
- Config: `output.headless = true` disables live UI by default (can be overridden by CLI)
- Behavior: headless skips Rich Live displays and relies on the injected Reporter/console logging

## Deprecated Namespaces → Contrib

- `peakfit.plotting`, `peakfit.analysis`, `peakfit.lineshapes`, `peakfit.optimizers` now emit deprecation warnings and forward to `peakfit.contrib.*`
- Prefer importing contrib modules directly; keep adapters lazy/optional to avoid hard dependencies in core paths

## Extension Points Checklist

- New lineshape: register in lineshape registry, provide N-D support, add tests and doc entry
- New optimizer: implement `OptimizationStrategy`, register via `register_strategy`, add deterministic seed handling
- New output: add writer under `io/writers`, bump schema version per output-system instructions, document in `docs/output_system.md`

## Constraints to Honor

- N-D generic handling (no hardcoded 2D paths); one pseudo-dimension supported today
- Output-system directory/layout and schema versioning rules
- No hidden files; cache under `cache/`
- Type hints and small, single-responsibility functions
