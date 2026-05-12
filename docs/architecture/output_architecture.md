# Output Architecture Notes

This document describes the current output system at a high level. The concrete user-facing
layout is documented in [docs/output_system.md](../output_system.md), and the proposed
redesign is tracked in [specs/output-revamp-plan.md](../../specs/output-revamp-plan.md).

## Current Shape

- Fit results are assembled into structured result models.
- Writers serialize selected data to JSON, CSV, Markdown, and related artifacts.
- The fit workflow currently coordinates output writing.
- Output files support both humans reviewing a run and tools consuming fit results.

## Known Cleanup Direction

The output revamp should reduce duplicated data, avoid placeholder files and empty
directories, make format and verbosity options real, and keep dense data in table or array
formats instead of bloating summary reports.

## Next Architecture Pass

Decide whether output planning belongs in the fit workflow, a small output module, or a
data-driven writer layer. Keep the result model and file layout easy to inspect from a
completed run.
