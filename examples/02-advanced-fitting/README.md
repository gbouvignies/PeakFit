# Example 2: Advanced Pseudo-ND CEST Workflow

This is the primary ready-to-run example for PeakFit.

It demonstrates:

- pseudo-ND fitting with plane values supplied through `--z-values`
- structured output files
- publication-ready PDF plots
- interactive `plot spectrum` review (Qt + Matplotlib)

## Dataset

- `data/pseudo3d.ft2` (pseudo-ND spectrum)
- `data/pseudo3d.list` (peak list)
- `data/b1_offsets.txt` (plane values: CEST offsets)

## Quick Start

```bash
bash run.sh
```

The script runs a fit and creates profile PDFs from the generated results.

## Manual CLI Workflow

```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --output Fits
```

Get the latest run directory:

```bash
LATEST_RUN="$(find Fits -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
```

Generate profile PDFs:

```bash
peakfit plot intensity "$LATEST_RUN" --output "$LATEST_RUN/figures/intensity_profiles.pdf"
peakfit plot cest "$LATEST_RUN" --output "$LATEST_RUN/figures/cest_profiles.pdf"
```

Interactive spectrum inspection (important fit-quality check):

```bash
peakfit plot spectrum --spectrum data/pseudo3d.ft2 --results "$LATEST_RUN"
```

## Config-Driven Variant

A valid config file is provided at `data/peakfit.toml`.

```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --config data/peakfit.toml
```

This variant writes to `Fits-config/` without a timestamp.

## Key Output Files

From `LATEST_RUN`:

- `summary/fit.json`
- `tables/parameters.csv`
- `tables/intensities.csv`
- `metadata/fitting_state.pkl` (used by `peakfit mcmc`)
