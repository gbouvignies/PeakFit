# Output Architecture Notes

This document describes the current output system at a high level. The concrete user-facing
layout is documented in [docs/output_system.md](../output_system.md).

## Current Shape

- Fit results are assembled into structured result models.
- `build_output_plan()` resolves concrete files from requested formats and available data.
- Writers serialize selected data to JSON, CSV, Markdown, and optional simulated spectra.
- The fit workflow coordinates fit artifacts, serialized state, and the output `README.md`.
- Output files support both humans reviewing a run and tools consuming fit results.
- Markdown reports are intentionally compact. Complete numeric detail belongs in JSON and CSV.

## Deliberate Simplifications

- There is no writer manager class; direct functions keep output flow visible.
- There is no output manifest; the file layout is fixed and documented.
- Run metadata, fit statistics, and MCMC diagnostics live in `summary/fit.json`.
- Per-plane amplitudes are exported only in `tables/intensities.csv`.
- Optional Markdown reports are bounded summaries, not full numeric exports.

## Next Architecture Pass

Keep output planning close to the writer functions unless a larger workflow split becomes
necessary. Avoid reintroducing manager, registry, or compatibility layers without a concrete
user workflow that justifies them.
