# PeakFit Architecture Guide

This is a short map of the current repository, not an immutable contract. `AGENTS.md`
is the source of truth for AI-assisted work and explicitly allows architecture changes
when they simplify the project while preserving scientific correctness and useful
workflows.

## Purpose

PeakFit fits lineshape models to pseudo-ND NMR spectra through a Python package and CLI.
The important domains are spectra and peak inputs, clustering, lineshapes, optimization,
fit orchestration, uncertainty analysis, plotting, and output generation.

## Current Package Map

- `src/peakfit/cli/` - Typer command-line entrypoints and terminal presentation.
- `src/peakfit/ui/` - Rich terminal helpers used by the current CLI.
- `src/peakfit/fit/` - validation, data loading, fitting workflow, result assembly, and output coordination.
- `src/peakfit/mcmc/` - uncertainty sampling, diagnostics, and MCMC-facing workflows.
- `src/peakfit/plot/` - plotting and the spectrum viewer.
- `src/peakfit/io/` - parsing, serialization, and file-format handling.
- `src/peakfit/engine/` - domain models, numerical algorithms, lineshapes, and fitting math.
- `src/peakfit/shared/` - small cross-cutting helpers.

## Current Flow

1. `cli` parses user intent and builds configuration.
2. `fit` validates inputs, loads spectra/peaks, prepares clusters, and runs direct fit pipeline functions.
3. `engine` performs numerical work: clustering, lineshapes, optimizers, residuals, and statistics helpers.
4. `fit.results` assembles fit output data; `fit.result_models` owns the runtime output
   dataclasses; `io` reads inputs and writes structured artifacts.
5. `mcmc` and `plot` run post-fit workflows from canonical fit outputs.
6. `ui` renders terminal output where the CLI needs it.

## Invariants To Preserve Or Change Explicitly

- Numerical and scientific behavior should remain correct and tested.
- Expensive fit work should fail early on invalid inputs.
- Input parsing errors should be clear and actionable.
- Long-running jobs should show understandable progress and failures.
- Pure numerical code should not accidentally depend on terminal UI, Typer, Qt, or filesystem side effects.
- Public behavior changes should be documented with migration notes when needed.

## Simplification Notes

The current vertical-slice shape is a starting point. It is acceptable to merge packages,
rename modules, delete thin wrappers, or move responsibilities when that makes the code
easier to understand and test. Prefer direct data flow and domain names over generic
services, managers, factories, adapters, or registries.
