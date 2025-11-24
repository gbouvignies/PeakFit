# PeakFit Examples

This directory contains practical examples demonstrating PeakFit's capabilities, from basic fitting to advanced optimization and uncertainty analysis.

## Quick Start

All examples can be run from the command line. Navigate to an example directory and run the provided script:

```bash
cd 02-advanced-fitting
bash run.sh
```

## Examples Overview

### 1. Basic Fitting
**Directory:** `01-basic-fitting/`
**Difficulty:** Beginner
**Time:** < 1 minute
**Status:** Template - adapt for your data

Demonstrates the most common use case: fitting a simple 2D or pseudo-3D spectrum with well-separated peaks.

**What you'll learn:**
- Loading NMRPipe spectra and peak lists
- Running a basic fit with default settings
- Interpreting the results

**Command:**
```bash
peakfit fit data/spectrum.ft2 data/peaks.list --output results/
```

### 2. Advanced Fitting with CEST Analysis
**Directory:** `02-advanced-fitting/`
**Difficulty:** Intermediate
**Time:** 2-3 minutes
**Status:** Ready to run with real data

Shows how to fit pseudo-3D CEST (Chemical Exchange Saturation Transfer) data with Z-axis values and custom configuration.

**What you'll learn:**
- Handling pseudo-3D experiments (CEST, CPMG, relaxation, etc.)
- Using Z-values for plane-dependent experiments
- Working with configuration files for reproducibility
- Analyzing CEST profiles

**Command:**
```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --output Fits/
```

### 3. Global Optimization for Difficult Peaks
**Directory:** `03-global-optimization/`
**Difficulty:** Advanced
**Time:** 5-10 minutes
**Status:** Template - uses same data with different approach

Demonstrates global optimization methods (basin-hopping, differential evolution) for fitting highly overlapping or difficult peaks where local optimization fails.

**What you'll learn:**
- When to use global optimization
- Choosing between basin-hopping and differential evolution
- Tuning global optimizer parameters
- Comparing local vs. global results

**Command:**
```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --optimizer basin_hopping \
  --output Fits/
```

### 4. Uncertainty Quantification
**Directory:** `04-uncertainty-analysis/`
**Difficulty:** Advanced
**Time:** 10-20 minutes
**Status:** Template - requires MCMC implementation

Shows how to estimate parameter uncertainties using MCMC sampling or profile likelihood methods.

**What you'll learn:**
- Running uncertainty analysis on fitted parameters
- Understanding parameter correlations
- Interpreting confidence intervals
- Validating fit quality

**Command:**
```bash
# First, run the fit
peakfit fit data/spectrum.ft2 data/peaks.list --output fit_results/

# Then, run uncertainty analysis (if available)
peakfit analyze uncertainty fit_results/
```

### 5. Batch Processing Multiple Datasets
**Directory:** `05-batch-processing/`
**Difficulty:** Intermediate
**Time:** Variable
**Status:** Template - demonstrates workflow pattern

Demonstrates efficient processing of multiple experiments with shared peak assignments.

**What you'll learn:**
- Processing multiple spectra in batch
- Automating workflows with shell scripts
- Organizing results from multiple experiments
- Comparing results across datasets

## Data Format Requirements

PeakFit supports:
- **Spectra:** NMRPipe format (.ft2, .ft3)
- **Peak lists:** NMRPipe, Sparky, or CSV format
- **Z-values:** Plain text file, one value per line (for pseudo-3D experiments)

## Getting Started

### For First-Time Users

Start with **Example 2** (Advanced Fitting) since it contains real data and is ready to run:

```bash
cd 02-advanced-fitting
bash run.sh
```

This will fit the provided CEST spectrum and generate results in the `Fits/` directory.

### Adapting Examples for Your Data

Most examples are templates showing the workflow. To use them with your data:

1. **Replace the data files** in the `data/` subdirectory
2. **Update the commands** in the README and run script
3. **Adjust parameters** as needed for your experiment type

## Example Data

The examples use a real pseudo-3D CEST NMR spectrum from a protein sample:
- **Spectrum:** 131 planes × 256 × 546 points
- **Experiment:** CEST with B1 offsets from -5000 to +5000 Hz
- **Peaks:** 166 peaks organized into 45 clusters

This dataset is used across multiple examples to demonstrate different analysis workflows.

## Output Files

After running a fit, you'll typically find:

- **`Fits/{peak_name}.out`** - Individual peak fitting results with intensity profiles
- **`Fits/shifts.list`** - Fitted chemical shift positions
- **`Fits/peakfit.log`** - Detailed log file with timestamps
- **`Fits/logs.html`** - Interactive HTML report (if enabled)

## Troubleshooting

### Common Issues

**"Command not found: peakfit"**
- Make sure PeakFit is installed: `pip install peakfit`
- Or install from source: `pip install -e .` from the repository root

**"File not found" errors**
- Check you're in the correct example directory
- Verify data files exist in the `data/` subdirectory

**"Fitting failed for all clusters"**
- Check data quality and signal-to-noise ratio
- Try adjusting contour level or clustering parameters
- Consider global optimization (Example 3)

**Results look wrong**
- Verify peak list coordinates match your spectrum
- Check lineshape model (auto-detected from spectrum header)
- Review the log file for warnings

### Getting Help

For each example, see the `README.md` in that directory for detailed explanations and expected output.

For general help:
```bash
peakfit --help
peakfit fit --help
peakfit plot --help
```

## Validation

Before fitting, validate your input files:

```bash
peakfit validate spectrum.ft2 peaks.list
```

This checks:
- File formats are correct
- Peaks are within spectral bounds
- No duplicate assignments
- Proper file permissions

## Creating Your Own Workflows

These examples provide templates you can adapt for your own data. Key principles:

1. **Start simple** - Use default settings first
2. **Validate inputs** - Always check files before fitting
3. **Use configuration files** - For reproducibility and documentation
4. **Check logs** - Review `peakfit.log` for detailed information
5. **Visualize results** - Plot profiles to validate fits

## Performance Tips

For large datasets:
- Use configuration files to specify parameters once
- Parallel processing has been removed; use single-process execution
- Consider clustering parameters to reduce number of fits
- Monitor memory usage for very large spectra

See the [Optimization Guide](../docs/optimization_guide.md) for detailed performance tuning.

## Citation

If you use PeakFit in your research, please cite the appropriate reference (see main README).

## Additional Resources

- **[Main README](../README.md)** - Installation and quick start
- **[Optimization Guide](../docs/optimization_guide.md)** - Performance tuning
- **[Documentation](../docs/)** - Complete documentation
- **[GitHub Issues](https://github.com/gbouvignies/PeakFit/issues)** - Report bugs or request features

---

**Need help?** Open an issue at https://github.com/gbouvignies/PeakFit/issues
