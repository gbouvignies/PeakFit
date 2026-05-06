# PeakFit Output System

This document describes the output files produced by `peakfit fit` in the current codebase.

## Overview

PeakFit writes a structured results directory (default: `Fits` or a timestamped subdirectory when enabled). The content depends on selected output formats and verbosity.

### Default Formats

The default output formats are:

- `json` → machine‑readable summaries
- `csv` → tabular data
- `txt` → Markdown report (`report.md`)

These are configured via `[output] formats` or the CLI `--format` option.

## Output Directory Layout

```
<output_dir>/
├── README.md
├── manifest.json
├── summary/
│   ├── fit_summary.json
│   └── report.md                 # only if txt format is enabled
├── parameters/
│   ├── parameters.csv            # model parameters only
│   ├── intensities.csv           # per-plane amplitudes
│   └── shifts.csv                # omitted for minimal verbosity
├── statistics/
│   └── fit_statistics.json       # omitted for minimal verbosity
├── diagnostics/
│   └── mcmc_diagnostics.json     # only if MCMC diagnostics are present
├── metadata/
│   ├── run_metadata.json
│   └── fitting_state.pkl
└── legacy/                       # only if legacy output is enabled
```

### Notes

- `README.md` is generated for each run and summarizes the results directory.
- `metadata/fitting_state.pkl` is the serialized state used by post-fit workflows.
- `summary/report.md` is created only when `txt` is in `output.formats`.
- `mcmc_diagnostics.json` is written only when the fit results include MCMC diagnostics.
- `legacy/` is written only when `output.include_legacy = true`.
- `manifest.json` indexes generated files and key run metrics.

## Optional Artifacts

These files are written only when explicitly enabled:

- `simulated.ftN` (at output dir root) when `output.save_simulated = true` and `nmrglue` is installed.
- `logs.html` when `output.save_html_report = true`.

## Configuration

### CLI

```
peakfit fit spectrum.ft2 peaks.list \
  --format json --format csv --format txt \
  --output-verbosity standard
```

### TOML

```toml
[output]
formats = ["json", "csv", "txt"]
verbosity = "standard"        # minimal, standard, full
include_legacy = false
save_simulated = false
save_html_report = false
include_timestamp = true
```

## Programmatic Access

The primary JSON entry point is:

- `summary/fit_summary.json`

This file is the canonical machine-readable summary for downstream analysis.
