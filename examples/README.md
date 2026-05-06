# PeakFit Examples

These examples are aligned with the current CLI and output layout.

## CLI Workflow

PeakFit now follows a simple command flow:

1. `peakfit fit ...` (includes mandatory pre-fit checks)
2. `peakfit mcmc ...` (post-fit uncertainty analysis)
3. `peakfit plot ...` (PDF plots and interactive spectrum review)

There is no separate `doctor` step.

## Examples

| Directory | Workflow | Status |
| --- | --- | --- |
| `01-basic-fitting/` | Minimal fit template for your own data | Template |
| `02-advanced-fitting/` | Ready-to-run pseudo-3D CEST fit + plotting | Ready |
| `03-global-optimization/` | Compare `varpro` vs `basin_hopping` | Ready |
| `04-uncertainty-analysis/` | Post-fit MCMC analysis from saved results | Ready |
| `05-constraints-and-protocols/` | Constraint-driven and multi-step fits | Ready |

## Common Output Layout

A fit run produces a results directory with this structure:

```text
<run_dir>/
├── summary/
│   ├── fit_summary.json
│   └── report.md
├── parameters/
│   ├── parameters.csv
│   ├── intensities.csv
│   └── shifts.csv
├── statistics/
│   └── fit_statistics.json
├── metadata/
│   ├── run_metadata.json
│   └── fitting_state.pkl
└── diagnostics/
    └── mcmc_diagnostics.json   # only when present
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
