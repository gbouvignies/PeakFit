# Example 5: Constraints and Fit Steps

This example demonstrates configuration-driven fitting with parameter constraints.

## Scenarios

1. `position_windows.toml` - global/per-axis position windows
2. `per_peak.toml` - targeted per-peak overrides
3. `multi_step.toml` - staged fitting steps

Each config writes to a stable output directory (`include_timestamp = false`):

- `Fits/scenario1`
- `Fits/scenario2`
- `Fits/scenario3`

## Quick Start

```bash
bash run.sh
```

## Manual Commands

```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --config configs/position_windows.toml

peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --config configs/per_peak.toml

peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --config configs/multi_step.toml
```

## What to Compare

For each scenario, inspect:

- `summary/fit.json`
- `tables/parameters.csv`

The run script prints each scenario's global chi-squared value for a quick comparison.
