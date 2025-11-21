# PeakFit UI/UX Style Guide

This document defines the user interface and user experience patterns for PeakFit CLI commands.

## Design Principles

1. **Professional**: Feel like a commercial-grade scientific tool
2. **Intuitive**: First-time users can accomplish tasks without reading docs
3. **Informative**: Always clear what's happening and what to do next
4. **Consistent**: Same patterns and conventions throughout
5. **Helpful**: Error messages guide users to solutions
6. **Beautiful**: Visually appealing and well-organized output

## Color Scheme

PeakFit uses a consistent color palette across all commands:

- **Primary (Cyan/Blue)**: Headers, logo, emphasis
- **Success (Green)**: Successful operations, confirmations
- **Warning (Yellow)**: Warnings, notes, caution
- **Error (Red)**: Errors, failures
- **Info (Dim Cyan)**: Informational messages, hints
- **Dimmed (Gray)**: Secondary information, metadata

### Usage Examples

```python
from peakfit.messages import console

# Success messages
console.print("[green]✓[/] Operation completed successfully")

# Errors
console.print("[red]✗ Error:[/] File not found")

# Warnings
console.print("[yellow]⚠ Warning:[/] Large dataset detected")

# Info
console.print("[dim]ℹ Auto-detected lineshape:[/] [cyan]sp1[/]")

# Emphasis
console.print("[bold cyan]Processing clusters...[/]")
```

## Logo and Branding

### ASCII Logo

The PeakFit logo appears:
- In documentation (README.md)
- On startup with `--verbose` flag
- In error pages (if web UI added)

```
   ___           _     _____ _ _
  / _ \ ___  __ _| | __|  ___(_) |_
 / /_)/ _ \/ _` | |/ /| |_  | | __|
/ ___/  __/ (_| |   < |  _| | | |_
\/    \___|\__,_|_|\_\|_|   |_|\__|
```

### Display Function

The logo is displayed using the PeakFitUI class from `ui/style.py`:

```python
from peakfit.ui.style import PeakFitUI

ui = PeakFitUI()
ui.print_banner()  # Shows logo with version and description
```

## Command Structure

### Argument Pattern

**Positional arguments**: Always required, file paths
```bash
peakfit fit SPECTRUM PEAKS    # Good
peakfit fit --spectrum X --peaks Y    # Bad (too verbose)
```

### Option Pattern

**Options**: Always with `--flag` format
```bash
--output-dir PATH
--config FILE
--parallel
--jobs N
```

### Boolean Flags

Use `--flag` and `--no-flag` pattern:
```bash
--parallel / --no-parallel
--verbose / --quiet
--overwrite / --no-overwrite
--show / --no-show
```

## Help Text Structure

Every command must have:

1. **One-line description** (clear and concise)
2. **Longer description** (2-3 sentences explaining when/how to use)
3. **Required arguments** (with type and description)
4. **Optional arguments** (with defaults and when to change them)
5. **Examples** (at least 2-3 common usage patterns)
6. **See also** (optional: links to related commands)

### Template

```python
@app.command()
def command_name(...) -> None:
    """One-line description of what this command does.

    Longer description explaining when to use this command and what it
    accomplishes. Include key concepts and typical use cases.

    Common parameter values: X (for case A), Y (for case B).

    Examples:
      Basic usage:
        $ peakfit command arg1 arg2

      Advanced usage with options:
        $ peakfit command arg1 arg2 --option value

      Edge case handling:
        $ peakfit command arg1 arg2 --special-flag

    See also:
      • peakfit other-command - Related functionality
      • Documentation: https://peakfit.readthedocs.io/command
    """
```

## Error Messages

### Principle: Every error must be actionable

❌ **Bad**: `Error: Invalid configuration`

✅ **Good**:
```
✗ Error: Invalid configuration in config.toml

Problem: Unknown parameter 'linshape' in [fitting] section
         Did you mean: 'lineshape'?

Valid options:
  • lineshape: auto, gaussian, lorentzian, voigt
  • optimizer: least_squares, basin_hopping

Documentation: https://peakfit.readthedocs.io/config
```

### Error Message Functions

```python
from peakfit.ui.style import PeakFitUI

ui = PeakFitUI()

# Simple error
ui.error("Configuration file is invalid")

# Or use console directly for quick messages
from peakfit.messages import console
console.print("[bold red]✗[/] Configuration file is invalid")
```

## User Feedback Patterns

### Auto-Detection Messages

When the CLI auto-detects a parameter:

```python
from peakfit.messages import print_auto_detection

print_auto_detection("lineshape", "sp1", "spectrum header")
# Output: ℹ Auto-detected lineshape from spectrum header: sp1
```

### Smart Defaults

When using a smart default:

```python
from peakfit.messages import print_smart_default

print_smart_default("--jobs", "8", "detected 8 CPU cores")
# Output: → Using --jobs: 8 (detected 8 CPU cores)
```

### Success Messages

```python
from peakfit.ui.style import PeakFitUI

ui = PeakFitUI()
ui.success("Created configuration file: peakfit.toml")
# Output: ✓ Created configuration file: peakfit.toml

# Or use console directly
from peakfit.messages import console
console.print("[bold green]✓[/] Created configuration file: peakfit.toml")
```

### Next Steps

After completing an operation, suggest next steps:

```python
from peakfit.messages import print_next_steps

print_next_steps([
    "Review configuration: [cyan]peakfit.toml[/]",
    "Run fitting: [cyan]peakfit fit spectrum.ft2 peaks.list[/]",
    "View results: [cyan]peakfit plot intensity Fits/[/]",
])
```

## Progress Indicators

### Simple Progress Bar

```python
from rich.progress import Progress

with Progress() as progress:
    task = progress.add_task("[cyan]Fitting clusters...", total=100)
    for i in range(100):
        # Do work
        progress.update(task, advance=1)
```

### Rich Progress with Details

```python
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeRemainingColumn(),
) as progress:
    task = progress.add_task("Fitting clusters", total=100)
    # ...
```

## Summary Tables

### Performance Summary

```python
from peakfit.messages import print_performance_summary

print_performance_summary(
    total_time=125.5,
    n_items=100,
    item_name="clusters",
    successful=98
)
```

Output:
```
⏱️  Performance Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total clusters        100
Successful            98 (98.0%)
Failed                2
Total time            2m 5s
Average per cluster   1.255s
```

### Data Summary

```python
from peakfit.messages import print_data_summary

print_data_summary(
    spectrum_shape=(2048, 256, 40),
    n_planes=40,
    n_peaks=150,
    n_clusters=45,
    noise_level=1250.5,
    contour_level=6252.5
)
```

## Confirmation Prompts

For potentially destructive operations:

```python
from peakfit.messages import print_confirmation_prompt

if output_dir.exists():
    if not print_confirmation_prompt(f"Output directory '{output_dir}' exists. Overwrite?"):
        console.print("[yellow]Aborted.[/yellow]")
        return
```

## File Not Found Handling

```python
from peakfit.messages import print_file_not_found_with_suggestions
from pathlib import Path
import difflib

spectrum_file = Path("specturm.ft2")  # Typo!

if not spectrum_file.exists():
    # Find similar files
    parent = spectrum_file.parent or Path(".")
    similar = difflib.get_close_matches(
        spectrum_file.name,
        [f.name for f in parent.glob("*.ft2")],
        n=5,
        cutoff=0.6
    )
    similar_paths = [parent / name for name in similar]

    print_file_not_found_with_suggestions(spectrum_file, similar_paths)
```

Output:
```
✗ Error: File not found: specturm.ft2

Did you mean one of these?
  • spectrum.ft2
  • test_spectrum.ft2

Available *.ft2 files in .:
  • spectrum.ft2
  • test_spectrum.ft2
  • cest_spectrum.ft2
```

## Icons and Symbols

Use Unicode symbols consistently:

- ✓ Success checkmark
- ✗ Error cross
- ⚠ Warning triangle
- ℹ Information
- 📋 List/checklist
- 📄 Document/file
- 🎯 Target (logo title)
- ⏱️ Timer/performance
- → Arrow (smart defaults)

## Testing UI/UX

When adding new UI elements:

1. **Test in different terminal sizes**: 80x24, 120x40, 200x50
2. **Test with light/dark themes**: Ensure colors are readable
3. **Test error paths**: Verify error messages are helpful
4. **Test with real users**: Can they accomplish tasks?
5. **Test examples**: All examples in help text must work

## Accessibility

- **Color not required**: Info conveyed with text + color
- **Clear hierarchy**: Headers, sections, emphasis
- **Readable fonts**: Monospace for code, regular for text
- **Sufficient contrast**: All colors pass WCAG AA
- **Screen reader friendly**: Rich's alt text support

## Anti-Patterns to Avoid

❌ **Don't**: Use emojis excessively
✅ **Do**: Use sparingly (logo, section markers)

❌ **Don't**: `Error: File not found`
✅ **Do**: Explain what was expected and suggest fixes

❌ **Don't**: Show cryptic stack traces to users
✅ **Do**: Show user-friendly error, log technical details

❌ **Don't**: Use inconsistent terminology
✅ **Do**: "spectrum" everywhere (not "spectra" sometimes)

❌ **Don't**: Assume user knowledge
✅ **Do**: Explain abbreviations on first use (CEST, CPMG)

❌ **Don't**: Show raw paths
✅ **Do**: Highlight important parts with colors

## Version Information

When displaying version:

```python
from peakfit import __version__
from peakfit.messages import console

console.print(f"[bold cyan]PeakFit[/] [dim]v{__version__}[/]")
```

## Documentation Links

Always provide:
- Main documentation: https://github.com/gbouvignies/PeakFit
- Specific command docs: https://peakfit.readthedocs.io/{command}
- Issue tracker: https://github.com/gbouvignies/PeakFit/issues

## Summary

This style guide ensures PeakFit provides a consistent, professional, and helpful
user experience. When in doubt:

1. **Be helpful**: What would help the user right now?
2. **Be clear**: Can this be misunderstood?
3. **Be consistent**: Does this match existing patterns?
4. **Be beautiful**: Does this look professional?
5. **Be actionable**: What should the user do next?

---

**Last Updated**: 2025-11-18
**Version**: 1.0
**Maintainers**: PeakFit Development Team
