# Terminal Output Style Guide for PeakFit

This guide defines the current conventions for console output. All CLI code
should use the centralized UI helpers in `peakfit.ui`.

## 1. Use the Centralized UI API

**Do**

```python
from peakfit.ui import (
    console,
    show_standard_header,
    info,
    success,
    warning,
    error,
    create_table,
    print_summary,
    create_progress,
    show_error_with_details,
)
```

**Don’t** create your own `Console()` instance.

## 2. Verbosity and Headers

Use `set_verbosity()` and `show_standard_header()` for consistent banners:

```python
from peakfit.ui import Verbosity, set_verbosity, show_standard_header

set_verbosity(Verbosity.VERBOSE if verbose else Verbosity.NORMAL)
show_standard_header("Fitting")
```

## 3. Status Messages

Use the message helpers for all user‑facing output:

```python
info("Loading spectrum...")
success("Fitting complete")
warning("Using default parameters")
error("Failed to load spectrum")
```

## 4. Tables

```python
table = create_table("Validation Summary")
table.add_column("Check", style="key")
table.add_column("Status", style="value")
table.add_row("Spectrum", "[success]✓[/success]")
console.print(table)

print_summary(
    {"Total clusters": 42, "Successful": 40, "Failed": 2},
    title="Fitting Results",
)
```

## 5. Progress Indicators

```python
with create_progress() as progress:
    task = progress.add_task("Processing clusters...", total=n_clusters)
    for cluster in clusters:
        process(cluster)
        progress.advance(task)
```

Use `console.status()` for short, indeterminate tasks.

## 6. Errors and Diagnostics

Prefer rich error helpers:

```python
show_error_with_details("Loading spectrum", exc)
```

For missing files with suggestions, use `show_file_not_found()`.

## 7. Theme Tokens

Use theme styles (no hardcoded colors). Common tokens:

- `success`, `warning`, `error`, `info`, `neutral`
- `header`, `subheader`, `panel.border`
- `key`, `value`, `metric`, `metric.good`, `metric.warn`, `metric.bad`
- `path`, `url`, `code`, `dim`, `emphasis`

## 8. Checklist

- No `print()` for user output
- No new `Console()` instances
- Use `peakfit.ui` helpers for headers, messages, tables, and progress
- Use theme tokens for styles
