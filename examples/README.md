# PeakFit Examples

These examples are aligned with the current CLI and output layout.

## CLI Workflow

PeakFit now follows a simple command flow:

1. `peakfit fit ...` (includes mandatory pre-fit checks)
2. `peakfit mcmc ...` (post-fit uncertainty analysis)
3. `peakfit plot ...` (PDF plots and interactive spectrum review)

## Recommended Path

Start with `02-advanced-fitting/`; it is the canonical ready-to-run workflow:
fit a pseudo-ND dataset, inspect canonical CSV/JSON outputs, and generate CEST
and intensity plots. Use the other examples only when you need that specific task.

## Examples

| Directory | Workflow | Status |
| --- | --- | --- |
| `01-basic-fitting/` | Minimal fit template for your own data | Template |
| `02-advanced-fitting/` | Ready-to-run pseudo-ND CEST fit + plotting | Ready |
| `03-global-optimization/` | Optional optimizer comparison for difficult fits | Ready |
| `04-uncertainty-analysis/` | Optional post-fit MCMC analysis from saved results | Ready |
| `05-constraints-and-fit-steps/` | Optional constraint-driven and multi-step fits | Ready |

## Common Output Layout

A fit run produces a results directory with this structure:

```text
<run_dir>/
├── summary/
│   └── fit.json
├── tables/
│   ├── parameters.csv
│   ├── intensities.csv
│   └── shifts.csv
├── metadata/
│   └── fitting_state.pkl
```

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

Then inspect fit quality with the Qt + Matplotlib spectrum viewer:

```bash
LATEST_RUN="$(find Fits -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
peakfit plot spectrum --spectrum data/pseudo3d.ft2 --results "$LATEST_RUN"
```
