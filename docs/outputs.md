# PeakFit Output System

This document describes the output files produced by `peakfit fit` in the current codebase.

## Overview

PeakFit writes a structured results directory (default: `Fits` or a timestamped subdirectory when enabled). The content depends on selected output formats and available result data.

### Default Formats

The default output formats are:

- `json` → machine‑readable summaries
- `csv` → tabular data

Markdown reports are optional. Add `txt` to `[output] formats` or pass
`--format txt` to write `summary/report.md`. There is no TSV output; CSV is the
canonical tabular format.

## Output Directory Layout

```
<output_dir>/
├── README.md
├── summary/
│   ├── fit.json
│   └── report.md                 # optional: only if txt format is enabled
├── tables/
│   ├── clusters.csv              # status, statistics, and terminal provenance
│   ├── parameters.csv            # usable final nonlinear parameters
│   ├── intensities.csv           # usable final per-plane amplitudes
│   └── shifts.csv                # only when shift parameters are present
├── metadata/
│   └── fitting_state.pkl
```

### Notes

- `README.md` is generated for each run and summarizes the result status, files,
  and common next commands for that run directory.
- `metadata/fitting_state.pkl` is mutable continuation state for post-fit
  workflows. It is not used to reconstruct completed scientific values.
- `summary/fit.json` is schema `4.0.0`: each cluster is keyed by stable
  `cluster_id` and carries final classification, terminal correction revision,
  optimizer provenance, and either frozen analytical values or an unusable reason.
- `summary/report.md` is created only when `txt` is in `output.formats`; it is a
  bounded review report with fit-quality checks and key parameters, not a
  complete parameter export.
- `tables/parameters.csv` is intentionally lean: parameter identity, fitted
  value, uncertainty, fixed status, and optional units/bounds. Rich metadata
  such as parameter category and global/shared status belongs in
  `summary/fit.json`.
- CSV columns are ordered for reading: `peak_name` first, the main independent
  variable or parameter column next, and final classification/provenance columns
  (including `cluster_id`) last. `clusters.csv` retains one row even for an
  unusable outcome; numerical tables omit unusable rows.
- `tables/intensities.csv` may contain signed amplitudes. CEST plots preserve
  signed normalized intensities; CPMG plots use only points with positive
  `I/I0` ratios because `R2eff` is log-transformed.

## Optional Artifacts

These files are written only when explicitly enabled:

- `simulated.ftN` (at output dir root) when `output.save_simulated = true` and
  `nmrglue` is installed. It is projected from final nonlinear parameters and
  stored analytical amplitudes without solving amplitudes again.

## Configuration

### CLI

```
peakfit fit spectrum.ft2 peaks.list \
  --format json --format csv
```

### TOML

```toml
[output]
formats = ["json", "csv"]
save_simulated = false
include_timestamp = true
```

## Programmatic Access

The primary JSON entry point is:

- `summary/fit.json`

This file is the canonical machine-readable summary for downstream analysis.
Old schema `3.0.0` result documents are intentionally rejected; rerun the fit to
produce the unambiguous `4.0.0` outcome format.
Post-fit commands and `ResultsLoader` expect the run output directory
(`<output_dir>`), not the `summary/` subdirectory.

## Implementation Notes

- `build_output_plan()` resolves concrete files from requested formats and
  available data.
- Direct writer functions serialize JSON, CSV, optional Markdown, run-level
  README/state companions, and optional simulated spectra.
- The fixed layout is the output contract.
- Per-plane amplitudes are exported only in `tables/intensities.csv`.
