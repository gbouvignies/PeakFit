# PeakFit Examples

This directory contains practical examples demonstrating PeakFit's capabilities, from basic fitting to advanced optimization and uncertainty analysis with MCMC.

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

Demonstrates the most common use case: fitting a 2D or pseudo-3D spectrum with well-separated peaks.

**What you'll learn:**

- Loading NMRPipe spectra and peak lists
- Running a basic fit with default settings
- Understanding the new structured output formats (JSON, CSV, Markdown)

**Command:**

```bash
peakfit fit data/spectrum.ft2 data/peaks.list --output results/
```

---

### 2. Advanced Fitting with CEST Analysis

**Directory:** `02-advanced-fitting/`
**Difficulty:** Intermediate
**Time:** 2-3 minutes
**Status:** Ready to run with real data

Shows how to fit pseudo-3D CEST (Chemical Exchange Saturation Transfer) data with Z-axis values and demonstrates the full output system.

**What you'll learn:**

- Handling pseudo-3D experiments (CEST, CPMG, relaxation, etc.)
- Using Z-values for plane-dependent experiments
- Exploring the new output formats:
  - `fit_results.json` - Machine-readable structured data
  - `parameters.csv` - Lineshape parameter estimates
  - `shifts.csv` - Chemical shifts (wide format)
  - `intensities.csv` - Fitted intensities
  - `report.md` - Human-readable formatted report
- Controlling output verbosity levels

**Command:**

```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --output Fits/ \
  --verbosity standard
```

---

### 3. Global Optimization for Difficult Peaks

**Directory:** `03-global-optimization/`
**Difficulty:** Advanced
**Time:** 5-10 minutes
**Status:** Template - uses same data with different approach

Demonstrates global optimization methods (basin-hopping, differential evolution) for fitting highly overlapping peaks where local optimization fails.

**What you'll learn:**

- When to use global optimization
- Choosing between basin-hopping and differential evolution
- Comparing results across optimization methods using structured outputs
- Using JSON output for programmatic comparison

**Command:**

```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --optimizer basin_hopping \
  --output Fits/ \
  --verbosity full
```

---

### 4. Uncertainty Analysis with MCMC

**Directory:** `04-uncertainty-analysis/`
**Difficulty:** Advanced
**Time:** 10-20 minutes
**Status:** Ready to run

Shows how to estimate parameter uncertainties using MCMC sampling and explores the comprehensive diagnostic outputs.

**What you'll learn:**

- Running MCMC-based uncertainty analysis
- Understanding MCMC diagnostics (R-hat, ESS, convergence status)
- Saving and loading MCMC chains
- Interpreting the `mcmc/` output directory:
  - `chains.npz` - Raw MCMC chains (NumPy format)
  - `chains_meta.json` - Chain metadata
  - `diagnostics.json` - Convergence diagnostics
- Viewing correlation matrices and posterior distributions

**Command:**

```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --method mcmc \
  --save-chains \
  --verbosity full \
  --output Fits/
```

---

### 5. Parameter Constraints and Multi-Step Protocols

**Directory:** `05-constraints-and-protocols/`
**Difficulty:** Intermediate
**Time:** 5-10 minutes
**Status:** Ready to run with real data

Demonstrates advanced parameter control including position windows, per-peak constraints, and multi-step fitting protocols for complex spectra.

**What you'll learn:**

- Setting position windows (global, per-dimension, per-peak)
- Using pattern-based constraints with glob syntax
- Creating multi-step fitting protocols
- Loading starting values from previous fits
- Combining constraints with staged optimization

**Command:**

```bash
# Run all scenarios
bash run.sh

# Or run individually with specific config
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --config configs/multi_step.toml
```

---

## Output Formats

PeakFit now generates structured outputs in multiple formats. See the [Output System Documentation](../docs/output_system.md) for complete details.

### Quick Reference

| Category    | File                                | Use Case                                         |
| ----------- | ----------------------------------- | ------------------------------------------------ |
| Summary     | `summary/fit_summary.json`          | Programmatic access, scripts, pipelines          |
| Summary     | `summary/analysis_report.md`        | Human-readable reports, documentation            |
| Summary     | `summary/quick_results.csv`         | Key results for spreadsheet import               |
| Parameters  | `parameters/parameters.csv`         | Lineshape parameters for analysis                |
| Parameters  | `parameters/amplitudes.csv`         | Fitted intensities for CEST/relaxation           |
| Statistics  | `statistics/fit_statistics.json`    | Chi-squared, AIC, BIC                            |
| Diagnostics | `diagnostics/mcmc_diagnostics.json` | R-hat, ESS, convergence                          |
| Metadata    | `metadata/run_metadata.json`        | Reproducibility info                             |
| Chains      | `chains/mcmc_chains.h5`             | Full MCMC chains                                 |
| Legacy      | `legacy/*.out`                      | Backward compatibility (with `--include-legacy`) |

### Verbosity Levels

Control output detail with `--verbosity`:

| Level      | What's Included                                     |
| ---------- | --------------------------------------------------- |
| `minimal`  | Essential results only (values, uncertainties)      |
| `standard` | Adds fit statistics and basic diagnostics (default) |
| `full`     | Everything: all diagnostics, metadata, raw data     |

### Example: Reading JSON Output

```python
import json

with open('Fits/summary/fit_summary.json') as f:
    results = json.load(f)

for cluster in results['clusters']:
    print(f"Cluster {cluster['cluster_id']}")
    print(f"  χ² = {cluster['fit_statistics']['chi_squared']:.3f}")
    for amp in cluster['amplitudes']:
        print(f"  {amp['peak_name']}: {amp['value']:.2e} ± {amp['uncertainty']:.2e}")
```

### Example: Loading MCMC Chains

```python
import numpy as np

# Load chains (NumPy fallback format)
data = np.load('Fits/chains/mcmc_chains.npz')
chains = data['chains']      # Shape: (n_chains, n_samples, n_params)
param_names = data['param_names']

print(f"Chains shape: {chains.shape}")
print(f"Parameters: {list(param_names)}")
# Example output: ['cs_F1_0', 'cs_F2_0', 'lw_F1_0', ...]

# Or use HDF5 format (preferred)
import h5py

with h5py.File('Fits/chains/mcmc_chains.h5', 'r') as f:
    chains = f['chains'][:]
    param_names = [name.decode() for name in f['param_names'][:]]
```

---

## Directory Structure

After running Example 2, you'll see:

```
02-advanced-fitting/
├── data/
│   ├── pseudo3d.ft2        # Input spectrum
│   ├── pseudo3d.list       # Peak list
│   ├── b1_offsets.txt      # Z-values
│   └── peakfit.toml        # Optional config
└── Fits/
    ├── README.md           # Auto-generated guide
    ├── summary/
    │   ├── fit_summary.json
    │   ├── analysis_report.md
    │   └── quick_results.csv
    ├── parameters/
    │   ├── parameters.csv
    │   ├── amplitudes.csv
    │   └── parameters.json
    ├── statistics/
    │   ├── fit_statistics.json
    │   └── residuals.csv
    ├── metadata/
    │   ├── run_metadata.json
    │   └── configuration.toml
    └── legacy/             # (if --include-legacy)
        ├── 2N-HN.out
        └── ...
```

After running Example 4 (MCMC), you'll also see:

```
Fits/
├── summary/
│   ├── fit_summary.json
│   ├── analysis_report.md
│   └── quick_results.csv
├── parameters/
├── statistics/
├── diagnostics/
│   ├── mcmc_diagnostics.json
│   ├── convergence.csv
│   └── warnings.txt
├── chains/
│   ├── mcmc_chains.h5
│   └── mcmc_chains.npz
├── figures/
│   ├── profiles/
│   ├── diagnostics/
│   └── correlations/
└── metadata/
```

├── mcmc/ # MCMC outputs
│ ├── chains.npz # Raw chains (if --save-chains)
│ └── diagnostics.json # Convergence info
└── figures/ # Figure catalog (if generated)
├── manifest.json
└── \*.pdf

````

---

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
````

This will fit the provided CEST spectrum and generate results in the `Fits/` directory.

### Adapting Examples for Your Data

Most examples are templates showing the workflow. To use them with your data:

1. **Replace the data files** in the `data/` subdirectory
2. **Update the commands** in the README and run script
3. **Adjust parameters** as needed for your experiment type

---

## Troubleshooting

### Common Issues

**"Command not found: peakfit"**

- Make sure PeakFit is installed: `uv sync --extra dev` (recommended for development) or `uv sync --all-extras` for all optional tooling
- Or install from source with pip: `pip install -e .` from the repository root

**"File not found" errors**

- Check you're in the correct example directory
- Verify data files exist in the `data/` subdirectory

**"Fitting failed for all clusters"**

- Check data quality and signal-to-noise ratio
- Try adjusting contour level or clustering parameters
- Consider global optimization (Example 3)

### Getting Help

For each example, see the `README.md` in that directory for detailed explanations and expected output.

```bash
peakfit --help
peakfit fit --help
```

---

## Additional Resources

- **[Output System Guide](../docs/output_system.md)** - Complete output format reference
- **[Main README](../README.md)** - Installation and quick start
- **[Optimization Guide](../docs/optimization_guide.md)** - Performance tuning
- **[GitHub Issues](https://github.com/gbouvignies/PeakFit/issues)** - Report bugs or request features

---

**Need help?** Open an issue at https://github.com/gbouvignies/PeakFit/issues
