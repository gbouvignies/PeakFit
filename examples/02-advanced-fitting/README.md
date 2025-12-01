# Example 2: Advanced Fitting with CEST Analysis

## Overview

This example demonstrates fitting a pseudo-3D CEST (Chemical Exchange Saturation Transfer) NMR spectrum using PeakFit. This is a realistic, production-ready example using real experimental data that showcases the **new structured output system**.

## Dataset

**Spectrum:** `data/pseudo3d.ft2`

- **Type:** Pseudo-3D CEST experiment
- **Dimensions:** 131 planes × 256 × 546 points
- **Size:** ~70 MB

**Peak list:** `data/pseudo3d.list`

- **Format:** Sparky format
- **Number of peaks:** 166 peaks

**Z-values:** `data/b1_offsets.txt`

- **Range:** -5000 to +5000 Hz
- **Number of points:** 131 (one per plane)

## Running the Example

### Quick Start

```bash
bash run.sh
```

### Step-by-Step

```bash
# 1. Validate inputs
peakfit validate data/pseudo3d.ft2 data/pseudo3d.list

# 2. Run the fit with new output system
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --output Fits/ \
  --verbosity standard

# 3. Explore the outputs
ls -la Fits/
```

## Output Files

After fitting, the `Fits/` directory contains the **new structured outputs**:

```
Fits/
├── fit_results.json    # Machine-readable structured results
├── parameters.csv      # Lineshape parameters (long format)
├── shifts.csv          # Chemical shifts (wide format)
├── intensities.csv     # Fitted intensities for all peaks
├── report.md           # Human-readable Markdown report
└── peakfit.log         # Detailed execution log
```

### fit_results.json - Structured Data

Complete structured results for programmatic access:

```json
{
  "version": "1.0",
  "metadata": {
    "timestamp": "2025-01-15T14:30:00Z",
    "peakfit_version": "0.9.0",
    "method": "scipy",
    "elapsed_seconds": 145.2,
    "input_files": {
      "spectrum": "pseudo3d.ft2",
      "peaks": "pseudo3d.list",
      "z_values": "b1_offsets.txt"
    }
  },
  "clusters": [
    {
      "cluster_id": 1,
      "peaks": ["2N-HN", "3N-HN"],
      "fit_statistics": {
        "chi_squared": 1.234,
        "reduced_chi_squared": 1.02,
        "degrees_of_freedom": 260
      },
      "parameters": [
        {
          "name": "cs_F1",
          "value": 115.632,
          "uncertainty": 0.001
        },
        {
          "name": "cs_F2",
          "value": 6.869,
          "uncertainty": 0.002
        },
        {
          "name": "lw_F1",
          "value": 25.3,
          "uncertainty": 1.2
        }
      ],
      "amplitudes": [
        {
          "peak_name": "2N-HN",
          "values": [1.5e6, 1.4e6, 1.3e6, ...],
          "uncertainties": [2.3e4, 2.1e4, 2.0e4, ...]
        }
      ]
    }
  ],
  "global_summary": {
    "n_clusters": 45,
    "n_peaks": 166,
    "total_parameters": 540,
    "overall_chi_squared": 1.05
  }
}
```

### results.csv - Tabular Data

Easy-to-analyze tabular format:

```csv
cluster_id,peak_name,plane,z_value,amplitude,uncertainty,chi_squared
1,2N-HN,0,-5000.0,1500000.0,23000.0,1.02
1,2N-HN,1,-4900.0,1480000.0,22000.0,1.01
1,2N-HN,2,-4800.0,1450000.0,21500.0,1.03
...
```

### results.md - Human-Readable Report

Markdown-formatted report for documentation:

```markdown
# PeakFit Results

**Date**: 2025-01-15 14:30:00
**Method**: SciPy Optimizer
**Runtime**: 2 min 25 sec

## Summary

| Metric          | Value |
| --------------- | ----- |
| Clusters fitted | 45    |
| Total peaks     | 166   |
| Overall χ²      | 1.05  |

## Cluster 1: 2N-HN, 3N-HN

**Fit Quality**: χ² = 1.234 (reduced: 1.02)

### Parameters

| Parameter | Value   | Uncertainty |
| --------- | ------- | ----------- |
| cs_F1     | 115.632 | ±0.001      |
| cs_F2     | 6.869   | ±0.002      |
| lw_F1     | 25.3    | ±1.2        |
| lw_F2     | 18.5    | ±0.8        |
```

## Verbosity Levels

Control output detail with `--verbosity`:

```bash
# Minimal: Essential results only
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --output Fits/ \
  --verbosity minimal

# Standard: Include statistics (default)
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --output Fits/ \
  --verbosity standard

# Full: Everything including all metadata
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --output Fits/ \
  --verbosity full
```

## Legacy Output Compatibility

For backward compatibility with existing scripts, add `--include-legacy`:

```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --output Fits/ \
  --include-legacy
```

This adds the traditional `*.out` profile files alongside the new formats:

```
Fits/
├── fit_results.json
├── parameters.csv
├── shifts.csv
├── intensities.csv
├── report.md
├── legacy/
│   ├── 2N-HN.out       # Legacy profile file
│   ├── 3N-HN.out
│   └── ...
└── peakfit.log
```

## Working with the Outputs

### Python: Analyze with JSON

```python
import json

with open('Fits/fit_results.json') as f:
    results = json.load(f)

# Get summary
print(f"Fitted {results['global_summary']['n_peaks']} peaks")
print(f"Overall χ² = {results['global_summary']['overall_chi_squared']:.3f}")

# Analyze each cluster
for cluster in results['clusters']:
    chi2 = cluster['fit_statistics']['chi_squared']
    peaks = ', '.join(cluster['peaks'])
    print(f"Cluster {cluster['cluster_id']}: {peaks} (χ²={chi2:.2f})")
```

### Python: DataFrame with pandas

```python
import pandas as pd

# Load intensities for CEST analysis
df = pd.read_csv('Fits/intensities.csv')

# Plot CEST profile for a specific peak
peak_data = df[df['peak_name'] == '2N-HN']
import matplotlib.pyplot as plt
plt.plot(peak_data['offset'], peak_data['intensity'])
plt.xlabel('B1 Offset (Hz)')
plt.ylabel('Intensity')
plt.title('CEST Profile: 2N-HN')
plt.show()

# Load parameters for detailed analysis
params_df = pd.read_csv('Fits/parameters.csv')
print(params_df[['peak_name', 'parameter', 'value', 'std_error']])
```

### Shell: Quick extraction with jq

```bash
# Get list of all peaks
jq -r '.clusters[].peaks[]' Fits/fit_results.json | sort -u

# Get chi-squared for each cluster
jq '.clusters[] | "\(.cluster_id): χ²=\(.fit_statistics.chi_squared)"' Fits/fit_results.json

# Count successful clusters
jq '[.clusters[] | select(.fit_statistics.chi_squared < 2.0)] | length' Fits/fit_results.json
```

## Expected Terminal Output

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

[Progress bar]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Fitted 45 clusters (166 peaks) in 2m 25s

Output files:
  ‣ Fits/fit_results.json  (structured data)
  ‣ Fits/parameters.csv    (parameter estimates)
  ‣ Fits/intensities.csv   (fitted intensities)
  ‣ Fits/report.md         (human-readable report)
```

## Advanced Options

### Using Configuration File

```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --config data/peakfit.toml
```

### Fixed Peak Positions

```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --fixed \
  --output Fits/
```

### Custom Lineshape

```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --lineshape pvoigt \
  --output Fits/
```

## Troubleshooting

### "Z-values file has wrong number of entries"

- Must have exactly 131 values (one per plane)
- Check: `wc -l data/b1_offsets.txt`

### "Fitting failed for some clusters"

- Check `Fits/fit_results.json` for error details
- Try global optimization: [Example 3](../03-global-optimization/)

### Output files missing

- Check `Fits/peakfit.log` for errors
- Verify output directory is writable

## Next Steps

1. **Global optimization** - [Example 3](../03-global-optimization/)
2. **MCMC uncertainty** - [Example 4](../04-uncertainty-analysis/)
3. **Read the output guide** - [docs/output_system.md](../../docs/output_system.md)

---

**Questions?** Open an issue at https://github.com/gbouvignies/PeakFit/issues
