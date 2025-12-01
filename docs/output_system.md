# PeakFit Output System

This document describes the redesigned output system for PeakFit, which provides structured, machine-readable outputs alongside human-readable reports.

## Overview

The new output system generates multiple output formats from a single fitting run:

| Format   | File           | Purpose                                    |
| -------- | -------------- | ------------------------------------------ |
| JSON     | `results.json` | Machine-readable, complete structured data |
| CSV      | `results.csv`  | Spreadsheet-compatible tabular data        |
| Markdown | `results.md`   | Human-readable formatted report            |
| Legacy   | `*.out`        | Backward-compatible text format            |

## Output Directory Structure

When `peakfit fit` completes, outputs are organized as follows:

```
output/
├── results.json        # Complete structured results
├── results.csv         # Parameter estimates table
├── results.md          # Human-readable report
├── results.out         # Legacy format (if enabled)
├── mcmc/               # MCMC outputs (if applicable)
│   ├── chains.npz      # Raw MCMC chains
│   ├── chains_meta.json# Chain metadata
│   └── diagnostics.json# Convergence diagnostics
└── figures/            # Generated plots
    ├── manifest.json   # Figure metadata catalog
    ├── cluster_001_profile.pdf
    ├── cluster_001_correlation.pdf
    └── ...
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

### JSON Output (results.json)

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

### CSV Output (results.csv)

Tabular output optimized for spreadsheet analysis:

```csv
cluster_id,peak_name,parameter,value,uncertainty,ci_lower_95,ci_upper_95,unit
1,G23,cs_F1,8.234,0.002,8.230,8.238,ppm
1,G23,cs_F2,120.5,0.05,120.4,120.6,ppm
1,G23,lw_F1,25.3,1.2,23.0,27.8,Hz
1,G23,lw_F2,18.5,0.8,16.9,20.1,Hz
```

### Markdown Report (results.md)

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

### chains.npz (NumPy compressed archive)

```python
import numpy as np

# Load chains
data = np.load('output/mcmc/chains.npz')
chains = data['chains']      # Shape: (n_chains, n_samples, n_params)
param_names = data['param_names']
```

### chains_meta.json

```json
{
  "n_chains": 4,
  "n_samples": 10000,
  "n_parameters": 12,
  "parameter_names": ["pos_x_0", "pos_y_0", "width_x_0", ...],
  "compression": "gzip",
  "dtype": "float64",
  "created": "2025-01-15T14:30:00Z"
}
```

## Figure Registry

The figure manifest (`figures/manifest.json`) catalogs all generated plots:

```json
{
  "generated": "2025-01-15T14:30:00Z",
  "n_figures": 45,
  "figures": [
    {
      "filename": "cluster_001_profile.pdf",
      "category": "profile",
      "title": "Cluster 1 Profile Fit",
      "cluster_id": 1,
      "dimensions": {
        "width_px": 800,
        "height_px": 600,
        "dpi": 150
      }
    }
  ],
  "by_category": {
    "profile": 15,
    "correlation": 15,
    "residual": 15
  }
}
```

## Programmatic Access

### Using FitResults Dataclass

```python
from peakfit.core.results import FitResults
from peakfit.io.writers import JSONWriter

# Load results from JSON
with open('results.json') as f:
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
```

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

with open('results.json') as f:
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
