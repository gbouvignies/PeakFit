# Terminal Output Principles

This guide records useful conventions from the current Rich-based CLI. These are examples
and principles, not mandatory architecture rules. If simplifying the CLI means replacing
or merging `peakfit.ui` helpers, do that deliberately and update this document.

## Principles

- Make long-running jobs readable in logs as well as interactive terminals.
- Surface failures, warnings, outliers, and next actions more prominently than routine success.
- Keep progress concise; avoid decorative output that hides useful information.
- Use consistent wording for scientific metrics such as reduced chi-squared.
- Prefer shared output helpers when they reduce duplication, but avoid preserving wrappers
  that only exist to mirror the old UI structure.

## Current Helper Pattern

The current CLI imports concrete helpers from `peakfit.ui` submodules for messages,
tables, progress, and error rendering. This keeps dependencies explicit while preserving
consistent wording:

```python
from peakfit.ui.messages import error, info, success, warning

info("Loading spectrum...")
success("Fitting complete")
warning("Using default parameters")
error("Failed to load spectrum")
```

## Error Output

Good error output should name the failed operation, include the actionable cause, and avoid
burying the useful message inside a large traceback unless verbose/debug output was requested.
