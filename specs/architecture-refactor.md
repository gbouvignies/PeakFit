# Feature: Architecture refactor to vertical slices + minimal engine

## Problem
- The current layered layout (CLI/UI → Services → Core → IO) has accumulated overlap and thin wrappers.
- UI concerns (Rich/PySide) and orchestration code leak across layers, making the code harder to reason about.
- Output/plotting/reporting dependencies are heavier than necessary for end users.
- The "doctor" and standalone "check" commands add surface area without improving reliability.

## Non-Goals
- Changing scientific algorithms or fitting correctness.
- Preserving backwards compatibility of public APIs or CLI options.
- Making GUI/CLI components optional; the install should be single-path.

## Behavior Rules
- **Engine purity:** `peakfit.engine` contains pure computation only (domain, algorithms, lineshapes, math). No I/O, Rich, Qt, or filesystem writes.
- **Fit orchestration:** `peakfit.fit` owns config, validation, data loading, pipeline orchestration, and output writing.
- **Mandatory validation:** Input validation runs automatically at the start of every fit; there is no standalone `check` CLI command.
- **Plot spectrum preserved:** `plot spectrum` remains an interactive Qt + Matplotlib viewer.
- **CLI UX preserved:** Rich output quality is maintained in the CLI layer.
- **No cross-slice coupling:** `fit`, `mcmc`, and `plot` do not import each other directly.
- **Single install path:** No optional extras; dependencies required for features are mandatory.

## Edge Cases
- Headless environments for Qt-based viewer.
- Large spectra and dense peak lists impacting viewer performance.
- Missing/partial results directories when plotting or running MCMC.
- Mixed-format peak lists requiring consistent validation.

## Risks
- Import cycles during the move to vertical slices.
- Breaking CLI commands while rewiring validation and orchestration.
- Regressions in plotting/report generation when replacing PDF tooling.

## Test Plan
- Add unit tests for validation + fit orchestration entrypoints.
- Add smoke tests for `peakfit fit`, `peakfit plot spectrum`, and `peakfit mcmc` CLI entrypoints.
- Run required gates after file moves: `uv run pre-commit run` and `uv run pytest`.

## Acceptance Checks
- `peakfit fit` runs validation first and fails fast on invalid inputs.
- `peakfit plot spectrum` launches the Qt + Matplotlib viewer.
- `peakfit mcmc` remains a first-class CLI command.
- Core algorithms and lineshapes can be imported from `peakfit.engine` without importing Rich/Qt.
