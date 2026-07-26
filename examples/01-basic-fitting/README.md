# Example 1: Basic Fitting Template

This template shows the smallest working `peakfit fit` workflow for your own spectrum.

## Expected Input Files

Place your files under `data/`:

- `data/spectrum.ft2` (or `.ft3`)
- `data/peaks.list` (Sparky peak list)
- optional `data/z_values.txt` plane values file for pseudo-ND data

## Run

```bash
bash run.sh
```

The script automatically:

- validates required files
- runs `peakfit fit`
- uses plane values when `data/z_values.txt` is present
- prints the latest run directory

## Manual Commands

2D fit:

```bash
peakfit fit data/spectrum.ft2 data/peaks.list --output results
```

Pseudo-ND fit:

```bash
peakfit fit data/spectrum.ft2 data/peaks.list \
  --z-values data/z_values.txt \
  --output results
```

## Key Files to Check

After a run:

- `summary/fit.json`
- `tables/parameters.csv`
- `tables/intensities.csv`

Next: use `../02-advanced-fitting/` for a real dataset workflow.
