# PeakFit

PeakFit is a Python package and CLI for fitting lineshape models to pseudo-ND NMR
spectra.

The normal workflow is:

```bash
peakfit fit spectrum.ft2 peaks.list
peakfit mcmc Fits/<run_dir>
peakfit plot intensity Fits/<run_dir>
peakfit plot cest Fits/<run_dir>
peakfit plot spectrum --spectrum spectrum.ft2 --results Fits/<run_dir>
```

Start with [`examples/02-advanced-fitting`](examples/02-advanced-fitting/) for a
ready-to-run pseudo-3D CEST workflow.

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

`peakfit fit` validates inputs before fitting. Omitting the peak list invokes the
experimental automatic peak-picking workflow; provide a peak list for routine
analyses.

Useful fit options:

- `--config peakfit.toml` for reproducible settings.
- `--lineshape auto|gaussian|lorentzian|pvoigt|sp1|sp2|no_apod`.
- `--optimizer varpro|basin_hopping`.
- `--refine N` for cross-talk refinement iterations.
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

For pseudo-3D data, pass plane values with `--z-values`. CEST examples use B1
offset values; CPMG plotting additionally needs `--time-t2`.

## Outputs

By default, `peakfit fit --output Fits` writes a timestamped run directory:

```text
Fits/<run_dir>/
├── README.md
├── summary/
│   └── fit.json
├── tables/
│   ├── parameters.csv
│   ├── intensities.csv
│   └── shifts.csv
├── metadata/
│   └── fitting_state.pkl
```

The canonical machine-readable summary is `summary/fit.json`. The canonical
tables are CSV files under `tables/`. Markdown reports are optional and are
written only when `txt` is requested.

See [`docs/outputs.md`](docs/outputs.md) for the exact output
contract.

## Plot

```bash
LATEST_RUN="$(find Fits -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"

peakfit plot intensity "$LATEST_RUN" --output "$LATEST_RUN/intensity_profiles.pdf"
peakfit plot cest "$LATEST_RUN" --output "$LATEST_RUN/cest_profiles.pdf"
peakfit plot cpmg "$LATEST_RUN" --time-t2 0.04 --output "$LATEST_RUN/cpmg_profiles.pdf"
peakfit plot spectrum --spectrum data/pseudo3d.ft2 --results "$LATEST_RUN"
```

CEST and CPMG plots are generated from `tables/intensities.csv`. CEST preserves
signed normalized intensities. CPMG computes `R2eff` from `log(I/I0)`, so only
points with a positive intensity ratio can contribute to the transformed profile.

## MCMC

Run MCMC after a successful fit:

```bash
peakfit mcmc "$LATEST_RUN" --peaks 2N-H --walkers 32 --steps 1000
peakfit plot mcmc "$LATEST_RUN" --output "$LATEST_RUN/mcmc_diagnostics.pdf"
```

`peakfit mcmc` reads `metadata/fitting_state.pkl` and writes chain files under
`chains/` when `--save-chains` is enabled.

## Documentation

- [`docs/outputs.md`](docs/outputs.md) - output files and formats.
- [`docs/constraints-and-fit-steps.md`](docs/constraints-and-fit-steps.md) -
  parameter constraints and `[[fitting.steps]]`.
- [`docs/optimizers.md`](docs/optimizers.md) - optimizer choice and runtime controls.
- [`docs/mcmc.md`](docs/mcmc.md) - MCMC workflow and diagnostics.
- [`docs/development.md`](docs/development.md) - contributor workflow and coding conventions.
- [`docs/architecture.md`](docs/architecture.md) - current package map and boundaries.
- [`AGENTS.md`](AGENTS.md) - instructions for AI-assisted maintenance.

## Development

```bash
uv sync --all-extras
uv run ruff check src tests
uv run ruff format --check src tests
uv run ty check
uv run lint-imports
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest -q
uv run prek run --all-files
uv build
```

Use `uv` for project commands. Use `prek`, not `pre-commit`, for hooks.

## License

GPL-3.0-or-later.
