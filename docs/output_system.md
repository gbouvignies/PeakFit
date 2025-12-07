# PeakFit Output System

This document describes the redesigned output system for PeakFit, which provides structured, machine-readable outputs alongside human-readable reports.

## Overview

The new output system generates multiple output formats from a single fitting run:

| Category    | File                                | Purpose                              |
| ----------- | ----------------------------------- | ------------------------------------ |
| Summary     | `summary/fit_summary.json`          | Complete structured results          |
| Summary     | `summary/analysis_report.md`        | Human-readable formatted report      |
| Summary     | `summary/quick_results.csv`         | Key results for spreadsheet import   |
| Parameters  | `parameters/parameters.csv`         | Lineshape parameter estimates (long) |
| Parameters  | `parameters/amplitudes.csv`         | Fitted intensities per plane         |
| Parameters  | `parameters/parameters.json`        | Parameters with full metadata        |
| Statistics  | `statistics/fit_statistics.json`    | Chi-squared, AIC, BIC                |
| Statistics  | `statistics/residuals.csv`          | Fit residual values                  |
| Diagnostics | `diagnostics/mcmc_diagnostics.json` | R-hat, ESS, convergence              |
| Diagnostics | `diagnostics/warnings.txt`          | Collected warnings                   |
| Metadata    | `metadata/run_metadata.json`        | Reproducibility info                 |
| Metadata    | `metadata/configuration.toml`       | Copy of input config                 |
| Chains      | `chains/mcmc_chains.h5`             | Full MCMC chains (HDF5)              |
| Legacy      | `legacy/*.out`                      | Backward-compatible text format      |

## Output Directory Structure

When `peakfit fit` completes, outputs are organized in a structured hierarchy:

```
output_YYYYMMDD_HHMMSS/
├── README.md                 # Auto-generated directory guide
├── summary/
│   ├── fit_summary.json      # Main results file
│   ├── analysis_report.md    # Human-readable report
│   └── quick_results.csv     # Key results for quick import
├── parameters/
│   ├── parameters.csv        # All parameters (long format)
│   ├── amplitudes.csv        # Intensities per plane
│   └── parameters.json       # Parameters with full metadata
├── statistics/
│   ├── fit_statistics.json   # Chi-squared, AIC, BIC
│   ├── residuals.csv         # Residual values
│   └── model_comparison.json # If multiple models
├── diagnostics/
│   ├── mcmc_diagnostics.json # R-hat, ESS, convergence
│   ├── convergence.csv       # Per-parameter convergence
│   └── warnings.txt          # Collected warnings
├── figures/
│   ├── profiles/             # Fit profiles per peak
│   ├── diagnostics/          # MCMC diagnostic plots
│   └── correlations/         # Correlation plots
├── metadata/
│   ├── run_metadata.json     # Reproducibility info
│   └── configuration.toml    # Copy of input config
├── chains/
│   ├── mcmc_chains.h5        # Full MCMC chains (HDF5)
│   └── mcmc_chains.npz       # NumPy archive (fallback)
└── legacy/                   # Legacy outputs (if --include-legacy)
    └── *.out
```

## Configuration

### Command Line Options

```bash
# Basic usage (default: minimal output)
peakfit fit spectrum.ft2 peaks.list

# Full output with all formats
peakfit fit spectrum.ft2 peaks.list --verbosity full

# Specify output directory
peakfit fit spectrum.ft2 peaks.list -o my_results

# Include legacy format for backward compatibility
peakfit fit spectrum.ft2 peaks.list --include-legacy

# Save MCMC chains (for MCMC method only)
peakfit fit spectrum.ft2 peaks.list -m mcmc --save-chains
```

### Verbosity Levels

The `--verbosity` option controls how much detail is included:

| Level      | Description                                              |
| ---------- | -------------------------------------------------------- |
| `minimal`  | Essential results only (parameter values, uncertainties) |
| `standard` | Includes fit statistics and basic diagnostics (default)  |
| `full`     | Complete output including all diagnostics and metadata   |

### Configuration File (peakfit.toml)

```toml
[output]
directory = "results"
verbosity = "standard"   # minimal, standard, or full
include_legacy = false   # Generate .out files
save_chains = true       # Save MCMC chains
save_figures = true      # Save plots
include_timestamp = true # Add timestamp to directory name
```

## Output Formats

### JSON Output (summary/fit_summary.json)

The JSON output provides complete, structured access to all fitting results:

```json
{
  "version": "1.0",
  "schema_version": "1.0",
  "metadata": {
    "timestamp": "2025-01-15T14:30:00Z",
    "peakfit_version": "0.9.0",
    "method": "mcmc",
    "elapsed_seconds": 45.2,
    "input_files": {
      "spectrum": "spectrum.ft2",
      "peaks": "peaks.list"
    }
  },
  "clusters": [
    {
      "cluster_id": 1,
      "peaks": ["G23", "S24"],
      "fit_statistics": {
        "chi_squared": 1.234,
        "reduced_chi_squared": 1.02,
        "degrees_of_freedom": 45,
        "aic": 156.7,
        "bic": 162.3
      },
      "parameters": [
        {
          "name": "cs_F1",
          "value": 8.234,
          "uncertainty": 0.002,
          "ci_lower_95": 8.23,
          "ci_upper_95": 8.238
        },
        {
          "name": "cs_F2",
          "value": 120.5,
          "uncertainty": 0.05,
          "ci_lower_95": 120.4,
          "ci_upper_95": 120.6
        }
      ],
      "amplitudes": [
        {
          "peak_name": "G23",
          "value": 1.5e6,
          "uncertainty": 2.3e4,
          "snr": 65.2
        }
      ],
      "mcmc_diagnostics": {
        "status": "GOOD",
        "n_samples": 10000,
        "n_chains": 4,
        "convergence": {
          "all_rhat_below_threshold": true,
          "effective_sample_size_adequate": true
        }
      }
    }
  ],
  "global_summary": {
    "n_clusters": 15,
    "n_peaks": 45,
    "total_parameters": 180,
    "overall_chi_squared": 1.05
  }
}
```

### CSV Output (parameters/parameters.csv)

Tabular output optimized for spreadsheet analysis:

```csv
cluster_id,peak_name,parameter,value,uncertainty,ci_lower_95,ci_upper_95,unit
1,G23,cs_F1,8.234,0.002,8.230,8.238,ppm
1,G23,cs_F2,120.5,0.05,120.4,120.6,ppm
1,G23,lw_F1,25.3,1.2,23.0,27.8,Hz
1,G23,lw_F2,18.5,0.8,16.9,20.1,Hz
```

### Markdown Report (summary/analysis_report.md)

Human-readable formatted report:

```markdown
# PeakFit Results

**Date**: 2025-01-15 14:30:00
**Method**: MCMC
**Runtime**: 45.2 seconds

## Summary

- Clusters fitted: 15
- Total peaks: 45
- Overall χ²: 1.05

## Cluster 1: G23, S24

### Fit Quality

- χ² = 1.234 (reduced: 1.02)
- MCMC Status: ✓ GOOD

### Parameters

| Parameter | Value | Uncertainty | 95% CI         |
| --------- | ----- | ----------- | -------------- |
| cs_F1     | 8.234 | ±0.002      | [8.230, 8.238] |
| cs_F2     | 120.5 | ±0.05       | [120.4, 120.6] |
| lw_F1     | 25.3  | ±1.2        | [23.0, 27.8]   |
```

### Legacy Output (\*.out)

For backward compatibility with existing analysis pipelines:

```
# PeakFit Results
# Generated: 2025-01-15T14:30:00
# Method: mcmc

CLUSTER 1
  Chi-squared: 1.234

  PEAK: G23
    Amplitude:    1.500e+06 ±  2.30e+04
    Position X:   8.234     ±  0.002
```

## MCMC Chain Storage

When using MCMC fitting with `--save-chains`:

### chains/mcmc_chains.h5 (HDF5 format - preferred)

```python
import h5py

with h5py.File('output/chains/mcmc_chains.h5', 'r') as f:
    chains = f['chains'][:]      # Shape: (n_chains, n_samples, n_params)
    param_names = [name.decode() for name in f['param_names'][:]]
```

### chains/mcmc_chains.npz (NumPy compressed archive - fallback)

```python
import numpy as np

# Load chains
data = np.load('output/chains/mcmc_chains.npz')
chains = data['chains']      # Shape: (n_chains, n_samples, n_params)
param_names = data['param_names']
```

### diagnostics/mcmc_diagnostics.json

```json
{
  "n_chains": 4,
  "n_samples": 10000,
  "n_parameters": 12,
  "parameter_names": ["cs_F1_0", "cs_F2_0", "lw_F1_0", ...],
  "convergence": {
    "all_rhat_below_threshold": true,
    "rhat_max": 1.02,
    "ess_min": 3200
  },
  "created": "2025-01-15T14:30:00Z"
}
```

```

## Figure Registry

Figures are organized by category in subdirectories:

```

figures/
├── profiles/ # Fit profiles per peak
│ ├── cluster_001_profile.pdf
│ └── cluster_002_profile.pdf
├── diagnostics/ # MCMC trace plots, autocorrelation
│ └── trace_cluster_001.pdf
└── correlations/ # Corner plots, parameter correlations
└── corner_cluster_001.pdf

````

## Programmatic Access

### Using FitResults Dataclass

```python
from peakfit.core.results import FitResults
from peakfit.io.writers import JSONWriter
import json

# Load results from JSON
with open('output/summary/fit_summary.json') as f:
    data = json.load(f)

# Or access directly after fitting
from peakfit.services.fit import FitPipeline

pipeline = FitPipeline(config)
results: FitResults = pipeline.run()

# Access structured data
for cluster in results.clusters:
    print(f"Cluster {cluster.cluster_id}")
    print(f"  Chi-squared: {cluster.statistics.chi_squared}")

    for param in cluster.parameters:
        print(f"  {param.name}: {param.value} ± {param.uncertainty}")
````

### Custom Output Writers

```python
from peakfit.io.writers import OutputWriter, Verbosity, WriterConfig

class CustomWriter:
    """Custom output writer implementation."""

    def __init__(self, config: WriterConfig):
        self.config = config
        self.verbosity = config.verbosity

    def write(self, results: FitResults, path: Path) -> None:
        """Write results in custom format."""
        # Implementation here
        pass
```

## Migration from Legacy Output

If you have scripts that parse the legacy `.out` format:

1. **Keep using legacy format**: Add `--include-legacy` to your command
2. **Migrate to JSON**: Use the structured JSON output for new scripts
3. **Use CSV**: For spreadsheet-based analysis, switch to CSV output

### Example Migration

**Before (parsing .out file):**

```python
# Fragile regex parsing
with open('results.out') as f:
    for line in f:
        if 'Amplitude:' in line:
            value = float(line.split()[1])
```

**After (using JSON):**

```python
import json

with open('output/summary/fit_summary.json') as f:
    results = json.load(f)

for cluster in results['clusters']:
    for amp in cluster['amplitudes']:
        value = amp['value']
        uncertainty = amp['uncertainty']
```

## Troubleshooting

### Missing Output Files

- Check that the output directory is writable
- Verify verbosity setting includes desired outputs
- For MCMC chains, ensure `--save-chains` is specified

### Large Output Files

- Use `--verbosity minimal` for reduced file sizes
- MCMC chains can be large; use compression or disable with `--no-save-chains`

### Compatibility Issues

- Legacy format is opt-in with `--include-legacy`
- JSON schema version is included for future compatibility

---

**Last Updated**: 2025-01-15
