# PeakFit Examples

These examples are aligned with the current CLI and output layout.

## CLI Workflow

PeakFit now follows a simple command flow:

1. `peakfit fit ...` (includes mandatory pre-fit checks)
2. `peakfit mcmc ...` (post-fit uncertainty analysis)
3. `peakfit plot ...` (PDF profile and MCMC diagnostic plots)

## Recommended Path

Start with `02-advanced-fitting/`; it is the canonical ready-to-run workflow:
fit a pseudo-ND dataset, inspect canonical CSV/JSON outputs, and generate CEST
and intensity plots. Use the other examples only when you need that specific task.

## Examples

| Directory | Workflow | Status |
| --- | --- | --- |
| `01-basic-fitting/` | Minimal fit template for your own data | Template |
| `02-advanced-fitting/` | Ready-to-run pseudo-ND CEST fit + plotting | Ready |
| `04-uncertainty-analysis/` | Optional post-fit MCMC analysis from saved results | Ready |
| `05-constraints-and-fit-steps/` | Optional constraint-driven and multi-step fits | Ready |

## Common Output Layout

A fit run produces a results directory with this structure:

```text
<run_dir>/
├── scientific.npz
├── summary/report.md
└── tables/
    ├── parameters.csv
    ├── intensities.csv
    ├── clusters.csv
    └── shifts.csv
```

Use `--detailed-evidence` to include portable terminal observations/predictions.
Older examples that explicitly request JSON remain supported, but JSON is not a
default. See [formats and migration](../docs/outputs.md).

By default, `peakfit fit --output Fits` writes to a timestamped subdirectory (for example `Fits/20260205_153012`).

Find your latest run with:

```bash
LATEST_RUN="$(find Fits -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
```

## Quick Start

```bash
cd 02-advanced-fitting
bash run.sh
```

Then inspect the peaks and fit quality in the desktop application.
This starts a separate analysis; it does not reopen the CLI fit. Fit the
dataset in the GUI to see model overlays:

```bash
peakfit gui data/pseudo3d.ft2 --peaks data/pseudo3d.list \
  --peak-axis-order indirect-direct --z-values data/b1_offsets.txt
```
