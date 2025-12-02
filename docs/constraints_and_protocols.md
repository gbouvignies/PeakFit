# Parameter Constraints and Multi-Step Fitting Protocols

PeakFit supports advanced parameter control through constraints and multi-step fitting protocols. This allows you to:

- Set custom starting values, bounds, and fix/vary status for parameters
- Define position windows relative to peak positions (per-peak and per-axis)
- Create multi-step fitting workflows (e.g., fix positions first, then release)
- Load starting values from previous fits

## Quick Start

### Basic Position Window Control

The simplest way to control position bounds is with a global position window:

```toml
[parameters]
position_window = 0.1  # ±0.1 ppm for all positions
```

### Per-Dimension Windows

Different dimensions often need different constraints (e.g., ¹H vs ¹⁵N):

```toml
[parameters.position_windows]
F2 = 0.5   # ±0.5 ppm for ¹⁵N (indirect dimension)
F3 = 0.05  # ±0.05 ppm for ¹H (direct dimension)
```

### Per-Peak Control

For peaks that need special treatment:

```toml
[parameters.peaks."G45N-HN"]
position_window = 1.0  # This peak can move more

[parameters.peaks."2N-H"]
position_window = 0.02  # This peak needs tight constraints
```

### Per-Peak, Per-Axis Control

The most specific level of control:

```toml
[parameters.peaks."Problem-Peak".position_windows]
F2 = 2.0   # Allow large movement in ¹⁵N
F3 = 0.01  # But keep ¹H very tight
```

## Complete Configuration Reference

### Position Windows

Position windows define how far a peak can move from its starting position. The bounds are computed as:
```
min = starting_position - window
max = starting_position + window
```

**Priority order (low to high):**
1. Code default (±1 FWHM)
2. `parameters.position_window` (global)
3. `parameters.position_windows.{axis}` (per-axis)
4. `parameters.peaks.{name}.position_window` (per-peak)
5. `parameters.peaks.{name}.position_windows.{axis}` (per-peak, per-axis)

### Direct Parameter Constraints

For non-position parameters, or when you need explicit bounds:

```toml
[parameters.defaults]
# Pattern-based defaults (glob syntax)
"*.*.lw" = { min = 5.0, max = 100.0 }  # All linewidths
"*.*.eta" = { value = 0.5, vary = false }  # Fix all eta values
"*.F2.*" = { vary = true }  # Vary all F2 parameters

[parameters.peaks."2N-H"]
# Per-peak, per-parameter constraints
"F2.cs" = { value = 120.5, vary = false }  # Fix this position
"F3.lw" = { min = 10.0, max = 50.0 }  # Custom linewidth bounds
```

### Pattern Syntax

Patterns use glob-style matching:
- `*` matches any sequence of characters
- `?` matches any single character
- `.` is literal

**Examples:**
- `*.*.cs` - all chemical shift parameters
- `*.*.lw` - all linewidth parameters  
- `*.F2.*` - all F2 (indirect) dimension parameters
- `2N-H.*.*` - all parameters for peak "2N-H"
- `G4?N-HN.*.cs` - CS parameters for peaks matching "G4?N-HN"

### Loading from Previous Fits

Start from a previous fit result:

```toml
[parameters]
from_file = "previous_fit/fit_summary.json"
```

This loads parameter values as starting points while respecting any other constraints you define.

## Multi-Step Fitting Protocols

Protocols define a sequence of fitting steps with different parameter constraints at each step.

### Basic Two-Step Protocol

A common workflow: fit linewidths with fixed positions, then refine everything:

```toml
[[fitting.steps]]
name = "fix_positions"
fix = ["*.*.cs"]
iterations = 1

[[fitting.steps]]
name = "full_optimization"
vary = ["*"]
iterations = 2
```

### Complex Protocol Example

```toml
[[fitting.steps]]
name = "linewidths_only"
description = "Optimize linewidths with everything else fixed"
fix = ["*"]
vary = ["*.*.lw"]
iterations = 1

[[fitting.steps]]
name = "lineshape_params"
description = "Optimize lineshape parameters (lw, eta)"
fix = ["*.*.cs"]
vary = ["*.*.lw", "*.*.eta"]
iterations = 1

[[fitting.steps]]
name = "full_refinement"
description = "Final refinement of all parameters"
vary = ["*"]
iterations = 3
```

### Step Configuration

Each step supports:

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Human-readable step name (for logging) |
| `description` | string | Optional description |
| `fix` | list[string] | Patterns for parameters to fix |
| `vary` | list[string] | Patterns for parameters to vary |
| `iterations` | int | Number of refinement iterations |

**Note:** `vary` patterns are applied after `fix` patterns, so you can do:
```toml
fix = ["*"]           # Fix everything
vary = ["*.*.lw"]     # Except linewidths
```

## Full Example Configuration

```toml
# peakfit.toml - Complete example

[fitting]
lineshape = "auto"
refine_iterations = 1  # Ignored when steps are defined

# Multi-step protocol
[[fitting.steps]]
name = "initial_fit"
description = "Fix positions, optimize linewidths"
fix = ["*.*.cs"]
iterations = 1

[[fitting.steps]]
name = "refine_all"
description = "Full optimization with tight position constraints"
vary = ["*"]
iterations = 2

[clustering]
contour_factor = 5.0

[output]
directory = "Fits"
formats = ["json", "csv"]
save_simulated = true

# Parameter constraints
[parameters]
# Global position window
position_window = 0.1

# Per-dimension windows (override global)
[parameters.position_windows]
F2 = 0.5   # ¹⁵N: ±0.5 ppm
F3 = 0.05  # ¹H: ±0.05 ppm

# Pattern-based defaults
[parameters.defaults]
"*.*.lw" = { min = 5.0, max = 100.0 }
"*.*.eta" = { min = 0.0, max = 1.0 }

# Per-peak constraints
[parameters.peaks."2N-H"]
position_window = 0.02  # Very tight for this peak
"F2.cs" = { vary = false }  # Fix 15N position completely

[parameters.peaks."G45N-HN"]
"F3.lw" = { value = 20.0, min = 15.0, max = 30.0 }

[parameters.peaks."W50N-HE1".position_windows]
F2 = 1.0   # Large movement allowed
F3 = 0.02  # But keep 1H tight
```

## CLI Options

Quick constraints can also be set via command line:

```bash
# Global position window
peakfit fit spectrum.ft2 peaks.list --position-window 0.1

# Per-dimension windows
peakfit fit spectrum.ft2 peaks.list \
    --position-window-f2 0.5 \
    --position-window-f3 0.05

# Fix patterns
peakfit fit spectrum.ft2 peaks.list --fix "*.*.cs" --fix "*.*.eta"

# Start from previous fit
peakfit fit spectrum.ft2 peaks.list --start-from previous/fit_summary.json
```

## Best Practices

1. **Start with position windows**: Most fitting problems benefit from reasonable position constraints. Use per-dimension windows matching the expected peak movement.

2. **Use multi-step protocols for difficult fits**: When peaks overlap or initial estimates are poor, a protocol that fixes positions first often helps.

3. **Be conservative with bounds**: Overly tight bounds can prevent convergence; overly loose bounds can lead to unphysical results.

4. **Load from previous fits**: When refitting similar data or refining fits, use `from_file` to start from known-good values.

5. **Document your protocol**: Use the `description` field in steps to explain your fitting strategy for reproducibility.
