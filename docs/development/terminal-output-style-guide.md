# Terminal Output Style Guide for PeakFit

This document establishes the **official style guide** for all terminal output in PeakFit. **Every contributor must follow these guidelines** to maintain consistency across the application.

---

## Core Principle

**Terminal output should feel like it comes from a single, polished, professional application - not a collection of different scripts.**

---

## 1. Branding & Version Display

### When to Show the Banner

PeakFit uses **Option C: Controlled by Verbosity** (the most flexible approach):

```bash
# Default (no banner):
$ peakfit fit spectrum.ft2 peaks.list
✓ Loaded spectrum: spectrum.ft2
...

# Verbose (shows banner):
$ peakfit fit spectrum.ft2 peaks.list --verbose
🎯 PeakFit v2.0.0 | https://github.com/gbouvignies/PeakFit
...

# Version flag (always shows):
$ peakfit --version
🎯 PeakFit v2.0.0
https://github.com/gbouvignies/PeakFit
```

### Implementation

```python
from peakfit.ui import PeakFitUI as ui

def my_command(verbose: bool = False):
    # Show banner based on verbosity
    ui.show_banner(verbose)

    # Rest of command...
```

---

## 2. Using the Centralized UI System

### Import Statement

**ALWAYS** use the centralized UI system:

```python
from peakfit.ui import PeakFitUI as ui
from peakfit.ui import console
```

**NEVER** create your own `Console()` instance.

---

## 3. Standard Message Patterns

### Status Messages

Use these methods for all status messages:

```python
# Success (green checkmark)
ui.success("Fitting complete")
ui.success(f"Saved results to: [path]{output_dir}[/path]")

# Error (red X)
ui.error("Failed to load spectrum")
ui.error(f"File not found: [path]{filepath}[/path]")

# Warning (yellow warning sign)
ui.warning("Using default parameters")
ui.warning(f"No reference points found for {file.name}")

# Info (blue info icon)
ui.info("Loading spectrum...")
ui.info(f"Found {len(files)} result files")
```

### Headers

```python
# Main header
ui.show_header("Fitting Clusters")

# Subheader (if needed)
ui.show_subheader("Processing Results")
```

---

## 4. Tables

### Creating Tables

**ALWAYS** use the standard table creation method:

```python
# With header
table = ui.create_table("Validation Summary")
table.add_column("Check", style="metric")
table.add_column("Status", style="value", justify="center")
table.add_row("Spectrum file", "[success]✓[/success] Found")
table.add_row("Peak list", "[success]✓[/success] Valid (45 peaks)")
console.print(table)

# Summary table (key-value pairs)
ui.print_summary({
    "Total clusters": n_clusters,
    "Successful": n_success,
    "Failed": n_failed,
    "Total time": format_time(elapsed),
    "Output directory": str(output_dir),
}, title="Fitting Results")
```

### Table Style Rules

[GOOD] **DO**:
- Use `box.ROUNDED` (automatically set by `ui.create_table()`)
- Use `"metric"` style for label columns
- Use `"value"` style for data columns
- Always include a title
- Right-align numeric columns: `justify="right"`

[BAD] **DON'T**:
- Create tables with `Table()` directly
- Use hardcoded colors like `style="green"`
- Omit titles

---

## 5. Progress Indicators

### Progress Bars

```python
# Standard progress bar
with ui.create_progress() as progress:
    task = progress.add_task(
        "Processing clusters...",  # Always end with "..."
        total=n_clusters
    )

    for cluster in clusters:
        # Do work
        process(cluster)

        # Advance
        progress.advance(task)
```

### Spinners

```python
# For indefinite operations
with console.status("[info]Loading spectrum...[/info]"):
    spectrum = load_spectrum(path)
```

---

## 6. Error Handling

### Standard Error Display

```python
# Simple error
ui.error(f"File not found: [path]{filepath}[/path]")

# Error with details
ui.show_error_with_details(
    context="Spectrum loading",
    error=exception,
    suggestion="Check that the file exists and is readable"
)

# File not found with suggestions
ui.show_file_not_found(
    filepath=Path("spectrum.ft2"),
    similar_files=[Path("spectrum.ft1"), Path("spectrum.ft3")]
)
```

---

## 7. Consistent Icons

Use these icons consistently:

| Icon | Meaning | Usage |
|------|---------|-------|
| ✓ | Success | `ui.success()` |
| ✗ | Error | `ui.error()` |
| ⚠ | Warning | `ui.warning()` |
| ℹ | Info | `ui.info()` |
| 🎯 | Branding | Banner only |
| 📋 | Next steps | `ui.print_next_steps()` |
| ⏱️ | Performance | `ui.print_performance_summary()` |

---

## 8. Color Scheme

Use theme names, not hardcoded colors:

[GOOD] **DO**:
```python
console.print(f"[success]Completed[/success]")
console.print(f"[path]{filepath}[/path]")
console.print(f"[code]peakfit fit ...[/code]")
```

[BAD] **DON'T**:
```python
console.print(f"[green]Completed[/green]")  # Wrong!
console.print(f"[bold cyan]{filepath}[/bold cyan]")  # Wrong!
```

### Available Theme Colors

- `success` - Bold green
- `warning` - Bold yellow
- `error` - Bold red
- `info` - Cyan
- `header` - Bold cyan
- `subheader` - Bold white
- `emphasis` - Bold
- `dim` - Dimmed text
- `code` - Bold magenta
- `value` - Green
- `metric` - Cyan
- `path` - Blue underline

---

## 9. Command Output Structure

### Template for ALL Commands

```python
from peakfit.ui import PeakFitUI as ui, console

def command_function(args, verbose: bool = False, quiet: bool = False):
    """Every command follows this structure."""

    # 1. Show banner (if verbose)
    ui.show_banner(verbose)

    # 2. Validate inputs (same format everywhere)
    ui.info("Validating inputs...")
    validate_inputs(args)
    ui.success("Inputs validated")

    # 3. Main operation with progress
    ui.show_header("Processing Data")

    with ui.create_progress() as progress:
        task = progress.add_task(
            f"Processing {n_items} items...",
            total=n_items
        )

        for item in items:
            # Do work
            process(item)
            progress.advance(task)

    # 4. Results summary (same format everywhere)
    ui.print_summary({
        "Total items": n_items,
        "Successful": n_success,
        "Failed": n_failed,
        "Total time": format_time(elapsed),
    }, title="Results")

    # 5. Next steps (if appropriate)
    if not quiet:
        ui.print_next_steps([
            "View results: [code]peakfit plot results.csv[/code]",
            "Documentation: [code]https://github.com/gbouvignies/PeakFit[/code]",
        ])
```

---

## 10. Best Practices

### DO:

[GOOD] Use the centralized UI system for ALL output
[GOOD] Add `--verbose` flag to all commands
[GOOD] Use consistent icons and colors
[GOOD] Include helpful suggestions in error messages
[GOOD] Show next steps after completing operations
[GOOD] Use progress indicators for long operations
[GOOD] Format file paths with `[path]` style
[GOOD] Format code examples with `[code]` style

### DON'T:

[BAD] Use `print()` directly
[BAD] Use `logger.info()` for user-facing messages
[BAD] Create your own `Console()` instances
[BAD] Use hardcoded colors like `[green]` or `[red]`
[BAD] Mix different table styles
[BAD] Show the banner unless `--verbose` is specified
[BAD] Forget to add progress indicators for slow operations

---

## 11. Code Review Checklist

Before submitting a PR, verify:

- [ ] No `print()` statements (use `ui.*` or `console.print()`)
- [ ] No `logger.info/warning/error` for user-facing messages
- [ ] All colors use theme names (not hardcoded)
- [ ] All icons are consistent
- [ ] All tables use `ui.create_table()`
- [ ] All progress bars use `ui.create_progress()`
- [ ] Banner shown only if `--verbose`
- [ ] Errors use `ui.error()` with helpful context
- [ ] Success messages use `ui.success()`
- [ ] File paths use `[path]` style
- [ ] Code examples use `[code]` style

---

## 12. Examples

### Example: Validation Command

```python
from peakfit.ui import PeakFitUI as ui, console

def run_validate(spectrum_path: Path, peaklist_path: Path, verbose: bool = False):
    # Banner
    ui.show_banner(verbose)

    ui.show_header("Validating Input Files")

    # Check spectrum
    ui.info(f"Checking spectrum: [path]{spectrum_path}[/path]")
    try:
        data = load_spectrum(spectrum_path)
        ui.success(f"Spectrum readable - Shape: {data.shape}")
    except Exception as e:
        ui.error(f"Failed to read spectrum: {e}")
        raise SystemExit(1)

    # Summary
    ui.print_summary({
        "Spectrum shape": str(data.shape),
        "Peaks": "45",
    }, title="Validation Summary")

    ui.success("Validation passed!")
```

### Example: Fitting Command

```python
from peakfit.ui import PeakFitUI as ui, console

def run_fit(spectrum_path: Path, verbose: bool = False):
    # Banner
    ui.show_banner(verbose)

    # Load data
    ui.show_header("Loading Data")
    with console.status("[info]Loading spectrum...[/info]"):
        spectra = read_spectra(spectrum_path)
    ui.success(f"Loaded spectrum: {spectrum_path.name}")

    # Fit clusters
    ui.show_header("Fitting Clusters")
    with ui.create_progress() as progress:
        task = progress.add_task(
            f"Processing {len(clusters)} clusters...",
            total=len(clusters)
        )

        for cluster in clusters:
            fit_cluster(cluster)
            progress.advance(task)

    # Summary
    ui.print_summary({
        "Total clusters": len(clusters),
        "Successful": n_success,
        "Total time": f"{elapsed:.2f}s",
    }, title="Fitting Results")

    ui.success("Fitting complete!")
```

---

## 13. Future Additions

When adding new features:

1. **Always** use the centralized UI system
2. **Follow** the established patterns
3. **Add** `--verbose` flag to new commands
4. **Update** this style guide if introducing new patterns
5. **Test** that output is consistent with existing commands

---

## Questions?

See the implementation in `src/peakfit/ui/style.py` for the full API reference.

For issues or suggestions, open an issue at: https://github.com/gbouvignies/PeakFit/issues
