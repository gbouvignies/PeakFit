# Example 5: Parameter Constraints and Multi-Step Protocols

## Overview

This example demonstrates PeakFit's advanced parameter control features:

- **Position windows** to constrain how far peaks can move
- **Per-peak constraints** for fine-grained control
- **Multi-step protocols** for complex fitting workflows
- **Pattern-based constraints** using glob syntax

These features are particularly useful when:

- Some peaks need tighter constraints than others
- Initial fits fail and you need a staged approach
- You want to fix certain parameters while varying others
- Loading starting values from a previous fit

## Quick Start

```bash
bash run.sh
```

This runs three fitting scenarios demonstrating different constraint configurations.

## Scenario 1: Position Windows

The most common use case is constraining how far peak positions can move from their initial values.

### Configuration (`configs/position_windows.toml`)

```toml
[fitting]
lineshape = "auto"
refine_iterations = 1

[parameters]
# Global position window: all peaks can move ±0.1 ppm
position_window = 0.1

# Per-dimension windows (override global)
[parameters.position_windows]
F2 = 0.5   # 15N: allow ±0.5 ppm movement
F3 = 0.05  # 1H: tight ±0.05 ppm constraint

[output]
directory = "Fits/scenario1"
```

### Why Different Windows Per Dimension?

- **¹H (F3)**: High resolution, peak positions are usually well-determined → tight constraints
- **¹⁵N (F2)**: Lower resolution, may need more room to move → looser constraints

### Command

```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --config configs/position_windows.toml
```

---

## Scenario 2: Per-Peak Constraints

Some peaks may need special treatment—either tighter or looser constraints.

### Configuration (`configs/per_peak.toml`)

```toml
[fitting]
lineshape = "auto"

[parameters]
# Default position windows
position_window = 0.1

[parameters.position_windows]
F2 = 0.3
F3 = 0.03

# Override for specific peaks
[parameters.peaks."2N-H"]
# This peak is well-resolved, use very tight constraints
position_window = 0.01

[parameters.peaks."2N-H"]
# Fix position completely (vary = false)
"F2.cs" = { vary = false }
"F3.cs" = { vary = false }

[parameters.peaks."W50N-HE1"]
# This tryptophan peak may need to move more
position_window = 0.2

[parameters.peaks."W50N-HE1".position_windows]
F2 = 1.0   # Allow significant 15N movement
F3 = 0.05  # But keep 1H reasonably tight

[parameters.peaks."G45N-HN"]
# Set custom linewidth bounds for this glycine
"F2.lw" = { min = 10.0, max = 80.0 }
"F3.lw" = { min = 15.0, max = 50.0 }

[output]
directory = "Fits/scenario2"
```

### Pattern-Based Defaults

You can also use glob patterns to set defaults for groups of parameters:

```toml
[parameters.defaults]
# Fix all eta (lineshape mixing) parameters
"*.*.eta" = { value = 0.5, vary = false }

# Set bounds for all linewidths
"*.*.lw" = { min = 5.0, max = 100.0 }

# Fix all chemical shifts for peaks starting with "G"
"G*.*.cs" = { vary = false }
```

---

## Scenario 3: Multi-Step Protocol

For difficult fits, a staged approach often works better than optimizing everything at once.

### Configuration (`configs/multi_step.toml`)

```toml
[fitting]
lineshape = "auto"

# Multi-step protocol
[[fitting.steps]]
name = "linewidths_only"
description = "First optimize linewidths with fixed positions"
fix = ["*.*.cs"]           # Fix all chemical shifts
vary = ["*.*.lw"]          # Vary linewidths
iterations = 1

[[fitting.steps]]
name = "positions_and_linewidths"
description = "Release positions but keep eta fixed"
fix = ["*.*.eta"]          # Fix lineshape mixing
vary = ["*.*.cs", "*.*.lw"]
iterations = 1

[[fitting.steps]]
name = "full_refinement"
description = "Final refinement of all parameters"
vary = ["*"]               # Vary everything
iterations = 2

[parameters]
# Position windows apply throughout all steps
position_window = 0.1

[parameters.position_windows]
F2 = 0.3
F3 = 0.03

[output]
directory = "Fits/scenario3"
```

### How Protocols Work

1. **Step 1**: Fix positions, only fit linewidths → stable initial estimate
2. **Step 2**: Release positions with tight bounds → refine peak locations
3. **Step 3**: Full optimization → final polish

This staged approach helps when:

- Peaks are overlapped and initial guesses are poor
- Global optimization is too slow
- You want reproducible, staged refinement

---

## Scenario 4: Starting from Previous Fit

Load parameter values from a previous fit as starting points.

### Configuration (`configs/from_previous.toml`)

```toml
[fitting]
lineshape = "auto"
refine_iterations = 2

[parameters]
# Load starting values from previous fit
from_file = "../Fits/scenario1/fit_results.json"

# Apply tight position windows around loaded values
position_window = 0.02

[output]
directory = "Fits/scenario4"
```

### Use Cases

- **Refitting with different settings**: Start from a good fit
- **Processing similar datasets**: Use one fit to seed another
- **Iterative refinement**: Gradually tighten constraints

---

## Pattern Syntax Reference

Constraints use glob-style pattern matching on parameter names. Parameter names follow the format: `{peak_name}.{axis}.{type}`

| Pattern    | Matches                  | Example                          |
| ---------- | ------------------------ | -------------------------------- |
| `*.*.cs`   | All chemical shifts      | `2N-H.F2.cs`, `G45N-HN.F3.cs`    |
| `*.*.lw`   | All linewidths           | `2N-H.F2.lw`, `W50N-HE1.F3.lw`   |
| `*.*.eta`  | All lineshape mixing     | `Peak1.F2.eta`                   |
| `*.F2.*`   | All F2 (indirect) params | `2N-H.F2.cs`, `2N-H.F2.lw`       |
| `*.F3.*`   | All F3 (direct) params   | `2N-H.F3.cs`, `2N-H.F3.lw`       |
| `2N-H.*.*` | All params for peak 2N-H | `2N-H.F2.cs`, `2N-H.F3.lw`       |
| `G*.*.*`   | All glycine peaks        | `G45N-HN.F2.cs`, `G12N-HN.F3.lw` |
| `*`        | Everything               | All parameters                   |

---

## Constraint Priority

When multiple constraints apply to a parameter, they are resolved in priority order:

1. **Code defaults** (from lineshape models)
2. **Global position window** (`parameters.position_window`)
3. **Per-axis position windows** (`parameters.position_windows.F2`)
4. **Pattern-based defaults** (`parameters.defaults."*.*.lw"`)
5. **Per-peak position window** (`parameters.peaks."2N-H".position_window`)
6. **Per-peak axis windows** (`parameters.peaks."2N-H".position_windows.F2`)
7. **Per-peak parameter constraints** (`parameters.peaks."2N-H"."F2.cs"`)

Higher priority always wins!

---

## Command-Line Options

Quick constraints can also be set via CLI:

```bash
# Global position window
peakfit fit spectrum.ft2 peaks.list --position-window 0.1

# Per-dimension windows
peakfit fit spectrum.ft2 peaks.list \
    --position-window-f2 0.5 \
    --position-window-f3 0.05

# Fix patterns
peakfit fit spectrum.ft2 peaks.list \
    --fix "*.*.cs" \
    --fix "*.*.eta"

# Start from previous fit
peakfit fit spectrum.ft2 peaks.list \
    --start-from previous/fit_results.json
```

---

## Output Structure

Each scenario creates its own output directory:

```
Fits/
├── scenario1/              # Position windows
│   ├── fit_results.json
│   ├── parameters.csv
│   └── ...
├── scenario2/              # Per-peak constraints
│   ├── fit_results.json
│   └── ...
├── scenario3/              # Multi-step protocol
│   ├── fit_results.json
│   └── ...
└── scenario4/              # From previous fit
    ├── fit_results.json
    └── ...
```

---

## Inspecting Constraint Application

Check the log file to see which constraints were applied:

```bash
grep -i "constraint\|position_window\|protocol" Fits/scenario3/peakfit.log
```

The log shows:

- Which position windows were applied
- Protocol step progression
- Which parameters were fixed/varied at each step

---

## Tips for Effective Constraints

### Start Conservative

Begin with looser constraints and tighten gradually:

```toml
# First try
position_window = 0.5

# If fits look good, tighten
position_window = 0.1
```

### Use Multi-Step for Overlap

When peaks overlap, fix positions first to get stable linewidths:

```toml
[[fitting.steps]]
fix = ["*.*.cs"]
iterations = 1

[[fitting.steps]]
vary = ["*"]
iterations = 2
```

### Combine Windows with Protocols

Apply position windows AND use a multi-step protocol for maximum control:

```toml
[parameters]
position_window = 0.1

[[fitting.steps]]
name = "initial"
fix = ["*.*.cs", "*.*.eta"]
iterations = 1

[[fitting.steps]]
name = "refine"
vary = ["*"]
iterations = 2
```

---

## Troubleshooting

### "Fit stuck at bounds"

Position window may be too tight. Try:

```toml
[parameters.peaks."problem_peak"]
position_window = 0.5  # Increase window
```

### "Fits oscillate between steps"

Reduce iterations or tighten constraints:

```toml
[[fitting.steps]]
name = "final"
vary = ["*"]
iterations = 1  # Reduce from 3
```

### "Parameter not varying"

Check pattern priority—a more specific pattern may have fixed it:

```bash
# Debug: print which patterns matched
grep "2N-H.F2.cs" Fits/scenario3/peakfit.log
```

---

## Next Steps

- **Uncertainty analysis**: Add MCMC to your protocol → [Example 4](../04-uncertainty-analysis/)
- **Global optimization**: Combine with basin-hopping → [Example 3](../03-global-optimization/)
- **Full documentation**: [docs/constraints_and_protocols.md](../../docs/constraints_and_protocols.md)

---

**Questions?** Open an issue at https://github.com/gbouvignies/PeakFit/issues
