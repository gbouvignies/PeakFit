# Example 1: Basic Fitting

## Overview

This example demonstrates the simplest use case for PeakFit: fitting a 2D or pseudo-3D NMR spectrum with well-separated peaks using default settings.

**Note:** This is a template example. You'll need to provide your own data files to run it. See [Example 2](../02-advanced-fitting/) for a complete, ready-to-run example with real data.

## What This Example Teaches

[GOOD] **Basic workflow:**
1. Prepare your spectrum and peak list
2. Validate input files
3. Run the fit with default settings
4. Explore the new structured outputs

[GOOD] **New output formats:**
- `results.json` - Machine-readable structured data
- `results.csv` - Spreadsheet-compatible tabular data
- `results.md` - Human-readable Markdown report

[GOOD] **When to use basic fitting:**
- Well-resolved peaks (minimal overlap)
- Good signal-to-noise ratio (>10:1)
- Standard 2D experiments (HSQC, HMQC, etc.)

## Prerequisites

### Data Files Needed

Place your own data files in the `data/` subdirectory:

1. **Spectrum file:**
   - Format: NMRPipe `.ft2` or `.ft3`
   - Type: 2D or pseudo-3D
   - Example: `data/spectrum.ft2`

2. **Peak list:**
   - Format: Sparky (`.list`), NMRPipe, or CSV
   - Example: `data/peaks.list`

3. **Z-values** (optional, for pseudo-3D):
   - Format: Plain text, one value per line
   - Example: `data/z_values.txt`

### Sparky Peak List Format

```
Assignment  w1      w2
A1N-HN      115.630 6.868
A2N-HN      117.519 8.693
A3N-HN      118.234 7.542
...
```

## Running the Example

### Step 1: Prepare Your Data

Copy your spectrum and peak list to the `data/` directory:

```bash
cp /path/to/your/spectrum.ft2 data/
cp /path/to/your/peaks.list data/
```

### Step 2: Validate Inputs

Always validate before fitting:

```bash
peakfit validate data/spectrum.ft2 data/peaks.list
```

### Step 3: Run the Fit

```bash
peakfit fit data/spectrum.ft2 data/peaks.list --output results/
```

For pseudo-3D with Z-values:

```bash
peakfit fit data/spectrum.ft2 data/peaks.list \
  --z-values data/z_values.txt \
  --output results/
```

### Step 4: Explore the Outputs

```bash
# View the structured results
ls -la results/

# Read the human-readable report
cat results/results.md

# Check the JSON for programmatic access
cat results/results.json | python -m json.tool | head -50

# Open the CSV in a spreadsheet
open results/results.csv  # macOS
# xdg-open results/results.csv  # Linux
```

## Output Files

After fitting, you'll find these new structured outputs:

### results/results.json

Machine-readable structured data for programmatic access:

```json
{
  "version": "1.0",
  "metadata": {
    "timestamp": "2025-01-15T14:30:00Z",
    "method": "scipy",
    "elapsed_seconds": 12.5
  },
  "clusters": [
    {
      "cluster_id": 1,
      "peaks": ["A1N-HN"],
      "fit_statistics": {
        "chi_squared": 1.23,
        "reduced_chi_squared": 1.02
      },
      "amplitudes": [
        {
          "peak_name": "A1N-HN",
          "value": 1.5e6,
          "uncertainty": 2.3e4
        }
      ]
    }
  ]
}
```

### results/results.csv

Spreadsheet-compatible tabular data:

```csv
cluster_id,peak_name,parameter,value,uncertainty,units
1,A1N-HN,amplitude,1500000.0,23000.0,intensity
1,A1N-HN,position_x,115.632,0.001,ppm
1,A1N-HN,position_y,6.869,0.002,ppm
```

### results/results.md

Human-readable Markdown report:

```markdown
# PeakFit Results

**Date**: 2025-01-15 14:30:00
**Method**: SciPy Optimizer
**Runtime**: 12.5 seconds

## Summary
- Clusters fitted: 15
- Total peaks: 42
- Overall χ²: 1.05

## Cluster 1: A1N-HN
| Parameter | Value | Uncertainty |
|-----------|-------|-------------|
| amplitude | 1.50e+06 | ±2.30e+04 |
| position_x | 115.632 | ±0.001 |
```

## Controlling Output Verbosity

Use the `--verbosity` flag to control detail level:

```bash
# Minimal: Just essential results
peakfit fit data/spectrum.ft2 data/peaks.list \
  --output results/ \
  --verbosity minimal

# Standard: Include statistics (default)
peakfit fit data/spectrum.ft2 data/peaks.list \
  --output results/ \
  --verbosity standard

# Full: Everything including all metadata
peakfit fit data/spectrum.ft2 data/peaks.list \
  --output results/ \
  --verbosity full
```

## Including Legacy Output

For backward compatibility with existing scripts:

```bash
peakfit fit data/spectrum.ft2 data/peaks.list \
  --output results/ \
  --include-legacy
```

This adds the traditional `*.out` files alongside the new formats.

## Working with the Outputs

### Python: Reading JSON

```python
import json

with open('results/results.json') as f:
    results = json.load(f)

for cluster in results['clusters']:
    print(f"Cluster {cluster['cluster_id']}: {cluster['peaks']}")
    for amp in cluster['amplitudes']:
        print(f"  {amp['peak_name']}: {amp['value']:.2e} ± {amp['uncertainty']:.2e}")
```

### Python: Reading CSV with pandas

```python
import pandas as pd

df = pd.read_csv('results/results.csv')

# Filter to amplitudes only
amplitudes = df[df['parameter'] == 'amplitude']
print(amplitudes[['peak_name', 'value', 'uncertainty']])
```

### Shell: Quick extraction

```bash
# Extract all amplitudes with jq
jq '.clusters[].amplitudes[] | "\(.peak_name): \(.value)"' results/results.json

# Get chi-squared values
jq '.clusters[] | "\(.cluster_id): χ²=\(.fit_statistics.chi_squared)"' results/results.json
```

## Success Criteria

A successful fit should have:
- ✓ All (or most) peaks converged
- ✓ Reasonable χ² values (~1.0-2.0)
- ✓ Small residuals (observed - fitted)
- ✓ Chemical shifts match expectations

## When Basic Fitting Isn't Enough

| Problem | Solution |
|---------|----------|
| Many failed fits | Try [Example 3: Global Optimization](../03-global-optimization/) |
| Uncertain results | Try [Example 4: MCMC Uncertainty](../04-uncertainty-analysis/) |
| Poor convergence | Increase refinement: `--refine 3` |
| Severe overlap | Global optimization or adjust clustering |

## Troubleshooting

### "No peaks found"
- Check peak list format matches spectrum dimensions
- Verify peaks are within spectral bounds

### "Fitting failed for cluster X"
- Check `results.json` for error details
- Try global optimization (Example 3)

### "Results look wrong"
- Verify peak list coordinates are correct
- Check lineshape model with `--lineshape auto`

## Next Steps

After running this example:

1. **Try Example 2:** Complete workflow with real CEST data
2. **Explore outputs:** Practice reading JSON/CSV programmatically
3. **Try MCMC:** Get uncertainty estimates (Example 4)

---

**Ready to try with real data?** See [Example 2](../02-advanced-fitting/) for a complete, runnable example.
