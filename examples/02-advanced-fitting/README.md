# Example 2: Advanced Fitting with CEST Analysis

## Overview

This example demonstrates fitting a pseudo-3D CEST (Chemical Exchange Saturation Transfer) NMR spectrum using PeakFit. This is a realistic, production-ready example using real experimental data.

CEST experiments measure chemical exchange by applying saturation at different frequency offsets and measuring the resulting intensity changes. This creates a pseudo-3D dataset where each plane corresponds to a different B1 offset frequency.

## Dataset

**Spectrum:** `data/pseudo3d.ft2`
- **Type:** Pseudo-3D CEST experiment
- **Dimensions:** 131 planes × 256 × 546 points
- **Size:** ~70 MB
- **Sample:** Protein sample
- **Experiment:** CEST with B1 offsets from -5000 to +5000 Hz

**Peak list:** `data/pseudo3d.list`
- **Format:** Sparky format
- **Number of peaks:** 166 peaks
- **Assignment:** Residue-specific assignments (e.g., "2N-HN", "3N-HN")

**Z-values:** `data/b1_offsets.txt`
- **Type:** B1 offset frequencies
- **Range:** -5000 to +5000 Hz
- **Number of points:** 131 (one per plane)

**Configuration:** `data/peakfit.toml`
- **Purpose:** Optional configuration file for reproducibility
- **Contains:** Fitting parameters, output options, optimization settings

## Running the Example

### Quick Start

```bash
# From this directory
bash run.sh
```

This script will:
1. Clean any previous results
2. Run PeakFit on the CEST data
3. Attempt to plot CEST profiles

### Step-by-Step

```bash
# 1. Validate inputs (optional but recommended)
peakfit validate data/pseudo3d.ft2 data/pseudo3d.list

# 2. Run the fit
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --output Fits/

# 3. View results
cat Fits/shifts.list
less Fits/peakfit.log

# 4. Plot CEST profiles
peakfit plot cest Fits/ --ref 0
```

### Using Configuration File

For better reproducibility, use the provided configuration file:

```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --config data/peakfit.toml
```

The configuration file allows you to:
- Document your analysis parameters
- Share exact settings with collaborators
- Reproduce analyses months or years later
- Version control your analysis workflow

## Expected Output

### Fitting Process

The fitting should complete in 2-3 minutes and display:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Loading Data
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Loaded spectrum: pseudo3d.ft2
  ‣ Shape: (131, 256, 546)
  ‣ Z-values: 131 planes

✓ Loaded 166 peaks
  ‣ Created 45 clusters

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Fitting Clusters
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Progress bar showing cluster fitting...]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Fitting Complete
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Metric               ┃ Value        ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ Total clusters       │ 45           │
│ Successful fits      │ 45 (100%)    │
│ Total time           │ ~2-3 minutes │
└──────────────────────┴──────────────┘
```

### Output Files

After completion, the `Fits/` directory will contain:

**Peak profiles:**
- `Fits/2N-HN.out` - Fitted intensity profile for residue 2
- `Fits/3N-HN.out` - Fitted intensity profile for residue 3
- ... (one file per peak)

**Summary files:**
- `Fits/shifts.list` - Fitted chemical shift positions
- `Fits/peakfit.log` - Detailed log with timestamps
- `Fits/logs.html` - Interactive HTML report (if enabled)
- `Fits/.peakfit_state.pkl` - Internal state for resuming/analysis

### Profile File Format

Each `.out` file contains the fitted CEST profile:

```
# Residue: 2N-HN
# F1: 115.630 ppm (fitted)
# F2: 6.868 ppm (fitted)
#
# Z-value    Intensity    Fitted    Residual
-5000.0     1.234e6      1.245e6   -0.011e6
-4900.0     1.256e6      1.251e6    0.005e6
...
```

**Columns:**
1. **Z-value:** B1 offset frequency (Hz)
2. **Intensity:** Observed peak intensity
3. **Fitted:** Fitted intensity from model
4. **Residual:** Difference (observed - fitted)

## What This Example Teaches

### Key Concepts

✅ **Pseudo-3D fitting:**
- PeakFit treats pseudo-3D data as a series of 2D planes
- Peaks are fitted in each plane independently
- Z-values link the planes together

✅ **Z-values file:**
- One value per plane (must match number of planes)
- Can be any experimental parameter: B1 offset, CPMG delay, time, temperature, etc.
- Values are written to output files for plotting

✅ **Configuration files:**
- TOML format for human-readable settings
- Documents all parameters in one place
- Enables reproducible analysis

✅ **Clustering:**
- Nearby peaks are grouped into clusters
- Each cluster is fitted as a unit (accounts for overlap)
- Reduces computation and improves fits for overlapping peaks

### Workflow Best Practices

1. **Always validate first:**
   ```bash
   peakfit validate data/pseudo3d.ft2 data/pseudo3d.list
   ```

2. **Check logs for issues:**
   ```bash
   less Fits/peakfit.log
   ```

3. **Visualize results:**
   - Plot profiles to check fit quality
   - Look for systematic deviations (residuals)
   - Verify chemical shifts match expectations

4. **Document parameters:**
   - Use configuration files
   - Add comments explaining choices
   - Version control the config file

## Advanced Usage

### Fixing Peak Positions

If you want to fix peak positions during fitting (only fit intensities):

```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --fixed \
  --output Fits/
```

This is useful when:
- Peak positions are well-known
- You only care about intensity profiles
- Peak overlap is severe

### Custom Lineshape

Force a specific lineshape model:

```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --lineshape pvoigt \
  --output Fits/
```

Available lineshapes:
- `auto` - Auto-detect from spectrum header (default)
- `gaussian` - Gaussian peaks
- `lorentzian` - Lorentzian peaks
- `pvoigt` - Pseudo-Voigt (mixed Gaussian/Lorentzian)

### Excluding Noisy Planes

Exclude specific planes from fitting:

```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --exclude 0 --exclude 130 \
  --output Fits/
```

This is useful for:
- Removing reference plane with artifacts
- Excluding planes with low signal
- Focusing on specific Z-value range

### Refinement Iterations

Increase refinement iterations for better convergence:

```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --refine 3 \
  --output Fits/
```

Higher values (2-5) can improve fits for:
- Highly overlapping peaks
- Complex lineshapes
- Difficult data

## Troubleshooting

### Common Issues

**"No peaks found"**
- Check peak list format (must match spectrum dimensions)
- Verify peak coordinates are within spectral bounds
- Try: `peakfit validate data/pseudo3d.ft2 data/pseudo3d.list`

**"Z-values file has wrong number of entries"**
- Must have exactly one value per plane
- This dataset has 131 planes → need 131 Z-values
- Check: `wc -l data/b1_offsets.txt` (should show 131)

**"Fitting failed for some clusters"**
- Check log file for specific errors: `less Fits/peakfit.log`
- May indicate overlapping peaks → try global optimization (Example 3)
- Low signal-to-noise → adjust contour level

**"Results file is empty"**
- Check that fitting completed successfully
- Look for errors in `Fits/peakfit.log`
- Verify output directory is writable

**"CEST profiles look wrong"**
- Check Z-values are correct (should be symmetric around 0)
- Verify reference plane selection (--ref 0 uses first plane)
- Check for artifacts in specific planes

### Validation Steps

1. **Check file sizes:**
   ```bash
   ls -lh data/
   # pseudo3d.ft2 should be ~70 MB
   ```

2. **Count peaks:**
   ```bash
   grep -c "N-H" data/pseudo3d.list
   # Should show 166 peaks
   ```

3. **Verify Z-values:**
   ```bash
   head data/b1_offsets.txt
   tail data/b1_offsets.txt
   # Should range from -5000 to +5000
   ```

4. **Check output:**
   ```bash
   ls -1 Fits/*.out | wc -l
   # Should have 166 .out files (one per peak)
   ```

## Next Steps

After running this example:

1. **Try global optimization** (Example 3)
   - Compare results with basin-hopping optimizer
   - See if fits improve for difficult peaks

2. **Explore uncertainty analysis** (Example 4)
   - Estimate parameter uncertainties
   - Understand fit quality

3. **Learn batch processing** (Example 5)
   - Process multiple CEST experiments
   - Compare results across datasets

4. **Adapt for your data**
   - Replace with your own pseudo-3D spectrum
   - Adjust parameters as needed
   - Use configuration file for documentation

## Reference

### Commands Summary

```bash
# Validate
peakfit validate SPECTRUM PEAKS

# Basic fit
peakfit fit SPECTRUM PEAKS --z-values ZFILE --output DIR

# With config
peakfit fit SPECTRUM PEAKS --z-values ZFILE --config CONFIG

# Fixed positions
peakfit fit SPECTRUM PEAKS --z-values ZFILE --fixed

# Custom lineshape
peakfit fit SPECTRUM PEAKS --z-values ZFILE --lineshape MODEL

# Exclude planes
peakfit fit SPECTRUM PEAKS --z-values ZFILE --exclude N

# More refinement
peakfit fit SPECTRUM PEAKS --z-values ZFILE --refine N
```

### File Formats

**Sparky peak list (`.list`):**
```
Assignment  w1      w2
2N-HN       115.630 6.868
3N-HN       117.519 8.693
```

**Z-values file (`.txt`):**
```
-5000.0
-4900.0
-4800.0
...
```

**Configuration file (`.toml`):**
```toml
[fitting]
lineshape = "auto"
refine_iterations = 1

[output]
directory = "Fits"
save_html_report = true
```

## Additional Resources

- **[Main Examples README](../README.md)** - Overview of all examples
- **[Optimization Guide](../../docs/optimization_guide.md)** - Performance tuning
- **[GitHub Issues](https://github.com/gbouvignies/PeakFit/issues)** - Get help

---

**Questions?** Open an issue at https://github.com/gbouvignies/PeakFit/issues
