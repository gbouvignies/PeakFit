# PeakFit

PeakFit is a Python package and CLI for fitting lineshape models to pseudo-ND NMR
spectra.

The normal workflow is:

```bash
peakfit fit spectrum.ft2 peaks.list
peakfit mcmc Fits/<run_dir>
peakfit plot intensity Fits/<run_dir>
peakfit plot cest Fits/<run_dir>
peakfit gui spectrum.ft2 --peaks peaks.list --peak-axis-order indirect-direct
```

Start with [`examples/02-advanced-fitting`](examples/02-advanced-fitting/) for a
ready-to-run pseudo-ND CEST workflow.

`peakfit gui` opens the desktop application: inspect and edit peaks, fit the
complete dataset, compare model and residual, export results, and save analyses
that reopen with `peakfit gui Analysis.peakfit`. See the
[desktop application guide](docs/gui.md).

## Install

PeakFit requires Python 3.14 or newer.

```bash
git clone https://github.com/gbouvignies/PeakFit.git
cd PeakFit
uv sync --all-extras
uv run peakfit --help
```

For an installed command in the current environment:

```bash
uv pip install .
peakfit --help
```

## Fit

Use an NMRPipe spectrum (`.ft2` or `.ft3`) and a peak list:

```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --output Fits
```

`peakfit fit` validates inputs before fitting; a peak list is required.

Useful fit options:

- `--config peakfit.toml` for reproducible settings.
- `--lineshape auto|gaussian|lorentzian|pvoigt|sp1|sp2|no_apod`.
- `--refine N` for exactly `N` optimizer passes; corrections update only between
  passes. Existing commands that relied on the former `N + 1` pass behavior
  should increment `N` by one.
- `--workers -1` to use all CPUs.
- `--format json --format csv --format txt` to request output formats.

Generate a starter config:

```bash
peakfit init peakfit.toml
```

## Inputs

Peak lists can be:

- Sparky-style `.list` files with `Assignment w1 w2 ...` columns.
- Tables (`.csv`, `.json`, `.xlsx`, `.xls`) with position columns such as
  `F1_ppm`, `w1`, or `Pos F1`.

For pseudo-ND data, use the retained `--z-values` option to supply one plane
value per plane. CEST examples use B1 offset values; CPMG plotting additionally
needs `--time-t2`.

## Outputs

By default, `peakfit fit --output Fits` writes a timestamped run directory:

```text
Fits/<run_dir>/
├── scientific.npz
├── summary/report.md
└── tables/
    ├── parameters.csv
    ├── intensities.csv
    ├── clusters.csv
    └── shifts.csv
```

The scientific archive retains exact results, provenance, sample mappings and
terminal corrections. CSV tables and a short report are written by default.
`--detailed-evidence` adds observations/sample predictions for portable review
and continuation. Full-grid simulation and large legacy JSON exports are opt-in.
GUI `.peakfit` projects reference the external spectrum rather than embedding it.

See [`docs/outputs.md`](docs/outputs.md) for the exact output
contract.

## Plot

```bash
LATEST_RUN="$(find Fits -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"

peakfit plot intensity "$LATEST_RUN" --output "$LATEST_RUN/intensity_profiles.pdf"
peakfit plot cest "$LATEST_RUN" --output "$LATEST_RUN/cest_profiles.pdf"
peakfit plot cpmg "$LATEST_RUN" --time-t2 0.04 --output "$LATEST_RUN/cpmg_profiles.pdf"
```

Intensity plots are generated from `tables/intensities.csv`. CEST and CPMG plots
read the recorded amplitudes from `scientific.npz`, because their errors propagate
each peak's covariance between planes, which only the archive records
([profile-uncertainty §4](docs/profile-uncertainty.md)); the values and errors are
the desktop profile's ratios exactly. CEST `--ref` indices count points in order of
offset. CEST preserves signed normalized intensities. CPMG computes `R2eff = −ln(I/I0)/T` with
σ = σ(I/I0) / ((I/I0)·T), so only points with a positive intensity ratio can
contribute to the transformed profile. Ratios use the desktop profile's
propagation: a peak whose covariance between planes is unavailable, or whose
reference is not distinguishable from its uncertainty (σ_R/|R| > 0.1, σ_R the total
σ of the reference mean), is skipped with the reason rather than plotted with
misleading error bars.

## Check the errors against replicate planes

```bash
peakfit replicates "$LATEST_RUN"
peakfit replicates "$LATEST_RUN" --baseline   # CEST baseline method
```

The command reads the run from `scientific.npz`, never refits it and changes no error.
It names the method that ran.

**Duplicate planes**, used when the run has planes with exactly equal plane values (no
tolerance matching): for each peak, the orthonormal contrasts between copies are
standardized jointly by the covariance the stored covariance factors predict for them.
The pooled RMS of standardized contrasts is near 1 when their uncertainty is
calibrated. It is reported apart for peaks with and without the correlated-profile
flag, each with its counts of contrasts, peaks and clusters, and a warning below 50
contrasts. Excluded peaks are listed with their reason. `--ref` and `--window` belong
to the baseline method.

**CEST baseline planes**, used otherwise or with `--baseline`: far from a peak's own
offset, CEST planes are approximate replicates. The command compares their
plane-to-plane scatter (a robust SD of second differences across offsets, planes
within `--window` Hz of the peak's offset and the reference planes excluded) with the
published profile σ. A median observed / stated near 1 means the errors are
calibrated; `--verbose` lists every peak. Plane values must be offsets in Hz from the
carrier of the first spectral axis.

What each method can and cannot show, and how to read its result:
[Interpreting the empirical checks](docs/processing-quantitative-series.md#interpreting-the-empirical-checks).

## MCMC

Run MCMC after a successful fit:

```bash
peakfit mcmc "$LATEST_RUN" --peaks 2N-H --walkers 32 --steps 1000
peakfit plot mcmc "$LATEST_RUN" --output "$LATEST_RUN/mcmc_diagnostics.pdf"
```

`peakfit mcmc` reads `scientific.npz` and verifies the source for compact results;
detailed evidence can continue without it. Each run is recorded under `mcmc/`, which
`peakfit plot mcmc` reads. Results without `scientific.npz` must be refit.

## Documentation

- [`docs/outputs.md`](docs/outputs.md) - output files and formats.
- [`docs/processing-quantitative-series.md`](docs/processing-quantitative-series.md) -
  which processing PeakFit's uncertainties are validated for.
- [`docs/constraints-and-fit-steps.md`](docs/constraints-and-fit-steps.md) -
  parameter constraints and `[[fitting.steps]]`.
- [`docs/optimizers.md`](docs/optimizers.md) - the VarPro optimizer and runtime controls.
- [`docs/mcmc.md`](docs/mcmc.md) - MCMC workflow and diagnostics.
- [`docs/development.md`](docs/development.md) - contributor workflow and coding conventions.
- [`docs/architecture.md`](docs/architecture.md) - current package map and boundaries.
- [`docs/gui-design.md`](docs/gui-design.md) - agreed GUI design and scientific review milestones.
- [`AGENTS.md`](AGENTS.md) - instructions for AI-assisted maintenance.

## Development

```bash
uv sync --all-extras
uv run ruff check src tests
uv run ruff format --check src tests
uv run ty check --error-on-warning
uv run lint-imports
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest -q
uv run prek run --all-files
uv build
```

Use `uv` for project commands. Use `prek`, not `pre-commit`, for hooks.

## License

GPL-3.0-or-later.
