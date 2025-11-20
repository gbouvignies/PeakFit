# Example 1: Basic Fitting

## Overview

This example demonstrates the simplest use case for PeakFit: fitting a 2D or pseudo-3D NMR spectrum with well-separated peaks using default settings.

**Note:** This is a template example. You'll need to provide your own data files to run it. See [Example 2](../02-advanced-fitting/) for a complete, ready-to-run example with real data.

## What This Example Teaches

✅ **Basic workflow:**
1. Prepare your spectrum and peak list
2. Validate input files
3. Run the fit with default settings
4. Check the results

✅ **When to use basic fitting:**
- Well-resolved peaks (minimal overlap)
- Good signal-to-noise ratio (>10:1)
- Standard 2D experiments (HSQC, HMQC, etc.)
- Default parameters work well

✅ **Key concepts:**
- Input file formats (NMRPipe, Sparky)
- Output file structure
- Result interpretation

## Prerequisites

### Data Files Needed

Place your own data files in the `data/` subdirectory:

1. **Spectrum file:**
   - Format: NMRPipe `.ft2` or `.ft3`
   - Type: 2D or pseudo-3D
   - Example: `data/spectrum.ft2`

2. **Peak list:**
   - Format: Sparky (`.list`), NMRPipe, or CSV
   - Coordinates: Must match spectrum dimensions
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

Where:
- **Assignment:** Peak name (e.g., "A1N-HN")
- **w1:** F1 chemical shift (ppm)
- **w2:** F2 chemical shift (ppm)

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

This checks:
- Files are readable
- Formats are correct
- Peaks are within spectral bounds
- No duplicate assignments

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

### Step 4: Check Results

```bash
# View fitted chemical shifts
cat results/shifts.list

# Check the log for any warnings
less results/peakfit.log

# List all output files
ls -lh results/
```

## Expected Output

After fitting, you'll find:

**results/shifts.list** - Fitted chemical shift positions:
```
Assignment  w1_fit    w2_fit
A1N-HN      115.632   6.869
A2N-HN      117.521   8.694
...
```

**results/{peak}.out** - Individual peak profiles (one per peak):
```
# Peak: A1N-HN
# F1: 115.632 ppm
# F2: 6.869 ppm
#
# Plane  Z-value  Intensity    Fitted      Residual
0        ...      1.234e6      1.245e6     -0.011e6
1        ...      1.256e6      1.251e6      0.005e6
...
```

**results/peakfit.log** - Detailed log with timestamps:
```
2025-11-20 12:34:56 | INFO  | PeakFit Session Started
2025-11-20 12:34:56 | INFO  | Loading spectrum...
2025-11-20 12:34:57 | INFO  | Loaded 42 peaks
2025-11-20 12:34:57 | INFO  | Created 15 clusters
...
```

## Success Criteria

A successful fit should have:
- ✓ All (or most) peaks converged
- ✓ Reasonable χ² values (~1.0-2.0)
- ✓ Small residuals (observed - fitted)
- ✓ Chemical shifts match expectations

## When Basic Fitting Isn't Enough

If you encounter:

❌ **Many failed fits**
→ Try [Example 3: Global Optimization](../03-global-optimization/)

❌ **Poor convergence**
→ Increase refinement iterations: `--refine 3`

❌ **Overlapping peaks**
→ PeakFit automatically clusters nearby peaks, but for severe overlap, try global optimization

❌ **Uncertain results**
→ Try [Example 4: Uncertainty Analysis](../04-uncertainty-analysis/)

## Advanced Options

### Fixing Peak Positions

To only fit intensities (fix positions):

```bash
peakfit fit data/spectrum.ft2 data/peaks.list \
  --fixed \
  --output results/
```

### Custom Lineshape

Force a specific lineshape model:

```bash
peakfit fit data/spectrum.ft2 data/peaks.list \
  --lineshape gaussian \
  --output results/
```

Options: `auto`, `gaussian`, `lorentzian`, `pvoigt`

### Adjusting Contour Level

Control peak clustering threshold:

```bash
peakfit fit data/spectrum.ft2 data/peaks.list \
  --contour-factor 5.0 \
  --output results/
```

Higher values → more aggressive clustering (group more peaks together)

### Using Configuration Files

For reproducibility, create a configuration file:

```bash
# Generate template
peakfit init config.toml

# Edit config.toml with your settings
# Then run:
peakfit fit data/spectrum.ft2 data/peaks.list \
  --config config.toml
```

## Troubleshooting

### "No peaks found"
- **Check:** Peak list format matches spectrum dimensions
- **Check:** Peaks are within spectral bounds (run validate)
- **Try:** Opening peak list in a text editor to verify format

### "Spectrum file not readable"
- **Check:** File is NMRPipe format (`.ft2` or `.ft3`)
- **Try:** `nmrPipe -in spectrum.ft2 -verb` to verify
- **Check:** File permissions

### "Fitting failed for cluster X"
- **Check:** Log file for specific error: `grep "cluster X" results/peakfit.log`
- **Try:** Global optimization (Example 3)
- **Try:** Adjusting contour level

### "Results look wrong"
- **Check:** Peak list coordinates are correct
- **Verify:** Lineshape model (auto-detected from header)
- **Check:** For systematic residuals in `.out` files

## Next Steps

After running this example:

1. **Visualize results:**
   - Plot fitted profiles
   - Check residuals for systematic deviations

2. **Try Example 2:**
   - Learn about pseudo-3D fitting with real data
   - See CEST analysis workflow

3. **Explore advanced features:**
   - Global optimization (Example 3)
   - Uncertainty analysis (Example 4)
   - Batch processing (Example 5)

## Reference

### Quick Command Reference

```bash
# Validate
peakfit validate SPECTRUM PEAKS

# Basic fit
peakfit fit SPECTRUM PEAKS --output DIR

# Pseudo-3D fit
peakfit fit SPECTRUM PEAKS --z-values ZFILE --output DIR

# With config
peakfit fit SPECTRUM PEAKS --config CONFIG

# Fixed positions
peakfit fit SPECTRUM PEAKS --fixed --output DIR

# Help
peakfit fit --help
```

### Required File Formats

**NMRPipe spectrum:**
- Binary format created by `nmrPipe`
- Extensions: `.ft2` (2D) or `.ft3` (3D)

**Sparky peak list:**
- Plain text
- Columns: Assignment, w1, w2
- Whitespace-separated

**Z-values (optional):**
- Plain text
- One numeric value per line
- Must match number of planes in spectrum

## Additional Resources

- **[Main Examples README](../README.md)** - Overview of all examples
- **[Example 2: Advanced Fitting](../02-advanced-fitting/)** - Complete example with real data
- **[Optimization Guide](../../docs/optimization_guide.md)** - Performance tuning
- **[GitHub Issues](https://github.com/gbouvignies/PeakFit/issues)** - Get help

---

**Ready to try with real data?** See [Example 2](../02-advanced-fitting/) for a complete, runnable example.
