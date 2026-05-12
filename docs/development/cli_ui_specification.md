# CLI UI Design Notes

These notes capture the current intent for PeakFit terminal output. They are examples and
principles, not a fixed UI specification. The CLI may be simplified when a clearer workflow
or smaller implementation is available.

## Main Principle

PeakFit fitting can run long enough that users may look away and return later. Output should
therefore leave a useful history, make progress visible, and highlight failures or suspicious
clusters without overwhelming the terminal.

## Useful Pattern: Stream With Summary

A practical shape for long-running fits is:

- a short pre-fit summary of inputs, method, and output destination;
- streaming per-cluster or per-step events that remain visible in the scrollback;
- concise progress and aggregate status;
- a final summary focused on failures, outliers, parameter-bound hits, and where results were written.

This pattern is not mandatory. It should be replaced if another design makes the main user
workflow easier to understand.

## Display Guidance

- Keep routine success compact.
- Make warnings and failed clusters easy to find.
- Explain high reduced chi-squared, boundary hits, missing files, and parse errors in actionable terms.
- Avoid decorative panels or live dashboards that hide the historical record of what happened.
- Use stable terminology, especially for domain metrics such as reduced chi-squared.

## Current Implementation Pointers

The current implementation has UI helpers under `src/peakfit/ui/`. These are useful context
when changing the CLI, but they may be merged, renamed, or removed as part of simplification.
