# PeakFit

Lineshape fitting for pseudo-ND NMR spectra.

## Features

- **Multiple lineshape models**: Gaussian, Lorentzian, Pseudo-Voigt, and apodization-specific models (SP1, SP2, No-Apod)
- **Automatic lineshape detection**: Detects optimal lineshape from NMRPipe processing parameters
- **Peak clustering**: Automatic grouping of overlapping peaks for simultaneous fitting
- **Modern CLI**: Intuitive command-line interface with rich terminal output
- **Configuration files**: TOML-based configuration for reproducible analyses
- **Type-safe**: Full type hints and Pydantic models for validation
- **Comprehensive testing**: Extensive test suite with synthetic data validation

## Installation

### Using uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package and project manager. Install it first:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then install PeakFit:

```bash
# Install PeakFit
uv pip install peakfit

# Or create a new project with PeakFit
uv init my-project
cd my-project
uv add peakfit
```

### Using pip

```bash
pip install peakfit
```

### Development Installation

```bash
git clone https://github.com/gbouvignies/PeakFit.git
cd PeakFit
uv sync --all-extras  # Install all dependencies including dev tools
```

## Requirements

- Python >= 3.14
- NMRPipe format spectrum files (.ft2, .ft3)

## Quick Start

### Basic Fitting

```bash
# Fit peaks in a pseudo-3D spectrum
peakfit fit spectrum.ft2 peaks.list

# Specify output directory and refinement iterations
peakfit fit spectrum.ft2 peaks.list --output Results --refine 2

# Use specific lineshape model
peakfit fit spectrum.ft2 peaks.list --lineshape pvoigt

# Fix peak positions during fitting
peakfit fit spectrum.ft2 peaks.list --fixed
```

### Using Configuration Files

```bash
# Generate a default configuration file
peakfit init config.toml

# Edit the configuration file, then run:
peakfit fit spectrum.ft2 peaks.list --config config.toml
```

Example configuration (`config.toml`):

```toml
[fitting]
lineshape = "auto"
refine_iterations = 2
fix_positions = false
max_iterations = 1000
tolerance = 1e-8

[clustering]
contour_factor = 5.0

noise_level = 100.0  # optional
exclude_planes = []

[output]
directory = "Fits"
formats = ["json", "csv"]
save_simulated = false
include_timestamp = true
```

### Input validation

`peakfit fit` automatically validates inputs before fitting and fails fast if any errors are found.

### MCMC Uncertainty Analysis

```bash
# Run MCMC sampling on fit results
peakfit mcmc Results/ --walkers 32 --steps 1000
```

### Plotting

```bash
# Generate intensity plots
peakfit plot intensity Results/ --show

# Generate CEST and CPMG profiles from fitted intensities
peakfit plot cest Results/
peakfit plot cpmg Results/ --time-t2 0.04

# Launch interactive spectrum viewer
peakfit plot spectrum --spectrum spectrum.ft2 --results Results/
```

## CLI Reference

### `peakfit fit`

Fit lineshapes to peaks in pseudo-3D NMR spectrum.

```bash
peakfit fit SPECTRUM [PEAKLIST] [OPTIONS]

Arguments:
  SPECTRUM                Path to NMRPipe spectrum file (.ft2, .ft3)
  PEAKLIST                Peak list file (.list, .csv); omit for automatic peak picking

Options:
  -z, --z-values PATH     Path to Z-dimension values file
  -o, --output PATH       Output directory [default: Fits]
  -c, --config PATH       Path to TOML configuration file
  -l, --lineshape TEXT    Lineshape: auto, gaussian, lorentzian, pvoigt, sp1, sp2
  -r, --refine INTEGER    Number of refinement iterations [default: 2]
  -t, --contour FLOAT     Contour level for segmentation
  -n, --noise FLOAT       Manual noise level
  --fixed/--no-fixed      Fix peak positions
  --jx/--no-jx            Fit J-coupling constant
  --phx/--no-phx          Fit phase correction in X
  --phy/--no-phy          Fit phase correction in Y
  -e, --exclude INTEGER   Plane indices to exclude
  -f, --format TEXT       Output format: json, csv, txt
  -w, --workers INTEGER   Parallel workers (-1 = all CPUs)
  --headless              Disable live UI
  --help                  Show this message and exit
```

### `peakfit mcmc`

Run MCMC sampling for uncertainty estimation on existing fit results.

```bash
peakfit mcmc RESULTS [OPTIONS]

Arguments:
  RESULTS                 Path to results directory from 'peakfit fit'

Options:
  --peaks TEXT            Peak names to analyze (default: all)
  -w, --walkers INTEGER   Number of MCMC walkers [default: 32]
  -s, --steps INTEGER     MCMC steps per walker [default: 1000]
  -b, --burn-in INTEGER   Burn-in steps (default: auto)
  --workers INTEGER       Parallel workers (-1 = all CPUs)
```

### `peakfit init`

Generate a default configuration file.

```bash
peakfit init [PATH] [OPTIONS]

Arguments:
  PATH    Path for new configuration file [default: peakfit.toml]

Options:
  -f, --force    Overwrite existing file
```

### `peakfit plot`

Generate plots from fitting results using subcommands.

```bash
peakfit plot [SUBCOMMAND] [RESULTS] [OPTIONS]
```

#### `peakfit plot intensity`

Plot raw fitted intensity profiles vs. plane index or Z value.

```bash
peakfit plot intensity RESULTS [--output PATH] [--show]
```

#### `peakfit plot cest`

Plot normalized CEST profiles from fitted intensities.

```bash
peakfit plot cest RESULTS [--ref INDEX] [--output PATH] [--show]
```

#### `peakfit plot cpmg`

Plot CPMG relaxation-dispersion profiles from fitted intensities.

```bash
peakfit plot cpmg RESULTS --time-t2 SECONDS [--output PATH] [--show]
```

#### `peakfit plot spectrum`

Launch interactive spectrum viewer.

```bash
peakfit plot spectrum --spectrum PATH [--results PATH]
```

#### `peakfit plot mcmc`

Generate MCMC diagnostic plots (traces and corners).

```bash
peakfit plot mcmc RESULTS [--output PATH] [--burn-in INT]
```

## Peak List Formats

### Sparky Format (`.list`)

```
# Sparky peak list
Assignment  w1   w2
Peak1  8.50  120.5
Peak2  7.80  115.3
Peak3  8.52  120.8
```

### CSV Format (`.csv`)

```csv
Assign F1,Assign F2,Pos F1,Pos F2
Peak1,Peak1,8.50,120.5
Peak2,Peak2,7.80,115.3
```

### JSON Format (`.json`)

```json
[
  { "name": "Peak1", "positions": [8.5, 120.5] },
  { "name": "Peak2", "positions": [7.8, 115.3] }
]
```

## Output Files

After fitting, PeakFit generates the following files in the output directory:

- **`README.md`** - short guide to the generated run files
- **`summary/fit.json`** - canonical machine-readable fit summary
- **`tables/parameters.csv`** - fitted model parameters
- **`tables/intensities.csv`** - per-plane fitted intensities and errors
- **`tables/shifts.csv`** - fitted chemical shifts when shift parameters are present
- **`metadata/fitting_state.pkl`** - saved state for plotting and MCMC workflows

Optional outputs:

- **`summary/report.md`** when `txt` is requested with `--format txt`
- **`simulated.ft2/ft3`** when `save_simulated = true`

## Lineshape Models

### Frequency-Domain Shapes

- **Gaussian**: `exp(-(dx²) * 4*ln(2) / FWHM²)`
- **Lorentzian**: `(0.5*FWHM)² / (dx² + (0.5*FWHM)²)`
- **Pseudo-Voigt**: `(1-η)*Gaussian + η*Lorentzian`

### Time-Domain Apodized Shapes

- **NO_APOD**: No apodization window applied
- **SP1**: Sine-bell apodization (power 1)
- **SP2**: Sine-bell apodization (power 2)

## Advanced Usage

### Fitting behavior

PeakFit fits each cluster with scipy.optimize least squares. Use `--workers` to control
parallel cluster execution; `--workers 1` is the simplest deterministic mode and
`--workers -1` uses all available CPUs.

### Excluding Planes

```bash
# Exclude specific planes from fitting
peakfit fit spectrum.ft2 peaks.list --exclude 0 --exclude 5 --exclude 10
```

### Custom Noise Level

```bash
# Set manual noise level instead of auto-detection
peakfit fit spectrum.ft2 peaks.list --noise 100.0
```

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/gbouvignies/PeakFit.git
cd PeakFit

# Install with all dependencies (recommended)
uv sync --all-extras

# Or install development dependencies only
uv sync --extra dev
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=peakfit --cov-report=html

# Run specific test file
uv run pytest tests/test_lineshapes_equivalence.py
```

### Code Quality

```bash
# Linting with Ruff
uv run ruff check src tests

# Type checking
uv run ty check

# Format code
uv run ruff format src tests

# Run pre-commit hooks
uv run pre-commit run --all-files
```

### Building the Package

```bash
# Build wheel and source distribution
uv build

# Build artifacts are in dist/
```

### Project Structure

```
src/peakfit/
├── cli/                # Modern CLI with Typer + Rich
│   ├── app.py          # Main Typer application
│   └── commands/       # Command implementations
├── engine/             # Pure computation (domain, algorithms, lineshapes)
├── fit/                # Fit workflow (validation, orchestration, outputs)
├── mcmc/               # MCMC workflows and diagnostics
├── io/                 # Input/output operations
│   ├── readers/        # File readers (Sparky, NMRPipe)
│   └── writers/        # File writers
├── plot/               # Visualization (plots + Qt spectrum viewer)
└── ui/                 # User interface (Rich + Qt viewer)
    └── console.py      # Rich console integration
```

## Plotting

PeakFit provides result plotting through `peakfit plot` subcommands:

### Intensity Profiles

```bash
# Generate intensity plots
peakfit plot intensity Fits/ --output plots.pdf

# Interactive display (limited to first 10 plots for large datasets)
peakfit plot intensity Fits/ --show
```

### CEST And CPMG Profiles

```bash
# CEST uses high-offset reference points automatically; pass --ref to override
peakfit plot cest Fits/ --output cest_profiles.pdf

# CPMG converts fitted intensities to R2eff with deterministic error propagation
peakfit plot cpmg Fits/ --time-t2 0.04 --output cpmg_profiles.pdf
```

### Interactive Spectra Viewer

```bash
# Launch Qt viewer with spectrum overlay
peakfit plot spectrum --spectrum data.ft2 --results Fits/
```

## Migration from Previous Version

The new CLI provides a more intuitive interface while maintaining all functionality:

| Old Command                                     | New Command                                               |
| ----------------------------------------------- | --------------------------------------------------------- |
| `peakfit -s spec.ft2 -l peaks.list`             | `peakfit fit spec.ft2 peaks.list`                         |
| `peakfit -s spec.ft2 -l peaks.list -o Out -r 3` | `peakfit fit spec.ft2 peaks.list --output Out --refine 3` |
| `peakfit -s spec.ft2 -l peaks.list --pvoigt`    | `peakfit fit spec.ft2 peaks.list --lineshape pvoigt`      |

## Citation

If you use PeakFit in your research, please cite:

```
[Citation information to be added]
```

## License

GPL-3.0-or-later

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

- NMRPipe file format support via [nmrglue](https://www.nmrglue.com/)
- Rich terminal output via [Rich](https://github.com/Textualize/rich)
- CLI framework via [Typer](https://typer.tiangolo.com/)
